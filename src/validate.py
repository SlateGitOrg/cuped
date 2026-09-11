"""The pre-treatment validation gate.

THE DIFFERENTIATOR LIVES HERE.

CUPED is unbiased only if the covariate is causally upstream of the assignment.
Use one measured after randomisation and the adjustment subtracts off part of
the treatment effect - and the resulting estimate is *confidently wrong*: the
interval gets tighter, the variance-reduction number goes up, every diagnostic
a generic implementation prints looks better than before. There is no failure
signal anywhere in the output. That is what makes this mistake worth building a
gate for rather than a warning.

The gate has two independent parts, and both are needed:

  DECLARATION (necessary, not sufficient). The caller must state the window in
  which the covariate was measured and that window must close at or before
  randomisation. This catches the honest mistake. It cannot catch a wrong
  declaration, because it is just metadata.

  BALANCE (empirical, and the part with teeth). Randomisation guarantees that
  anything measured before it is distributed identically across arms. So if the
  covariate differs between arms by more than chance, it is not pre-treatment -
  whatever the declaration says. This catches the dishonest or mistaken
  declaration, using the data itself.

WHAT THE GATE CANNOT DO, stated plainly because a gate whose limits are
undocumented is worse than no gate: the balance check detects post-treatment
covariates that the treatment actually MOVES. A variable measured after
randomisation but genuinely unaffected by the treatment will pass. It is still
unsafe - it can be a collider on a path between assignment and outcome - and no
amount of data will reveal that. Only the declaration, and the judgement behind
it, stands between you and that case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CovariateDeclaration:
    """What the analyst asserts about where the covariate came from.

    `measurement_window_end_days_before_randomisation` must be >= 0: the
    measurement window has to close before the coin is flipped. There is no
    default. Requiring the caller to write the number down is most of the
    value - it converts an unexamined assumption into a statement someone
    can be wrong about in review.
    """
    covariate: str
    measurement_window_start_days_before_randomisation: int
    measurement_window_end_days_before_randomisation: int
    rationale: str = ''


@dataclass(frozen=True)
class GateResult:
    covariate: str
    passed: bool
    reasons: tuple[str, ...] = field(default=())
    #: Standardised mean difference of the covariate between arms.
    balance_smd: float = 0.0
    balance_z: float = 0.0
    checks: tuple[tuple[str, bool], ...] = field(default=())


#: Significance threshold for the balance test.
BALANCE_ALPHA = 0.001

#: Minimum standardised mean difference that counts as real imbalance.
#:
#: The balance test needs BOTH a significant z AND a material SMD, and the
#: second condition is not decoration. At n = 200,000 the z-test has enough
#: power to flag an imbalance of 0.005 standard deviations - a difference with
#: no bearing on anything - so a p-value-only gate rejects perfectly valid
#: covariates as the sample grows, i.e. it gets *worse* the more data you have.
#: 0.02 SD is an order of magnitude below the conventional 0.1 "imbalance"
#: threshold, so this stays strict while remaining a statement about
#: magnitude rather than about sample size.
MIN_MATERIAL_SMD = 0.02


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def _var(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)


def _normal_sf(z: float) -> float:
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(abs(z) / math.sqrt(2))


def balance_check(
    control: list[float], treatment: list[float],
) -> tuple[bool, float, float]:
    """Return (balanced, smd, z) for a covariate across the two arms."""
    if len(control) < 2 or len(treatment) < 2:
        return True, 0.0, 0.0

    mc, mt = _mean(control), _mean(treatment)
    vc, vt = _var(control), _var(treatment)
    pooled_sd = math.sqrt((vc + vt) / 2)
    smd = (mt - mc) / pooled_sd if pooled_sd > 0 else 0.0

    se = math.sqrt(vc / len(control) + vt / len(treatment))
    z = (mt - mc) / se if se > 0 else 0.0

    significant = 2 * _normal_sf(z) < BALANCE_ALPHA
    material = abs(smd) >= MIN_MATERIAL_SMD
    return not (significant and material), smd, z


def run_gate(
    experiment, declaration: CovariateDeclaration, metric: str = 'revenue',
) -> GateResult:
    checks: list[tuple[str, bool]] = []
    reasons: list[str] = []

    # --- 1. the covariate must exist ---------------------------------------
    sample = experiment.users[0]
    exists = hasattr(sample, declaration.covariate)
    checks.append(('covariate_exists', exists))
    if not exists:
        reasons.append(f'no such field {declaration.covariate!r}')
        return GateResult(declaration.covariate, False, tuple(reasons),
                          checks=tuple(checks))

    # --- 2. it must not be the outcome -------------------------------------
    # Adjusting the outcome by itself drives the variance to zero and the
    # effect estimate with it. It looks like a spectacular result.
    not_outcome = declaration.covariate != metric
    checks.append(('covariate_is_not_the_outcome', not_outcome))
    if not not_outcome:
        reasons.append('the covariate is the outcome metric itself')

    # --- 3. the declared window must close before randomisation ------------
    start = declaration.measurement_window_start_days_before_randomisation
    end = declaration.measurement_window_end_days_before_randomisation
    window_ok = end >= 0 and start >= end
    checks.append(('declared_window_closes_before_randomisation', window_ok))
    if not window_ok:
        reasons.append(
            f'declared measurement window [-{start}d, -{end}d] does not close '
            'before randomisation'
        )

    # --- 4. the window must fit inside the pre-period ----------------------
    within = start <= experiment.pre_window_days
    checks.append(('declared_window_within_available_pre_period', within))
    if not within:
        reasons.append(
            f'declared window reaches {start}d back but only '
            f'{experiment.pre_window_days}d of pre-period exist'
        )

    # --- 5. balance: the check the declaration cannot fake ------------------
    control = [getattr(u, declaration.covariate)
               for u in experiment.arm('control')]
    treatment = [getattr(u, declaration.covariate)
                 for u in experiment.arm('treatment')]
    balanced, smd, z = balance_check(control, treatment)
    checks.append(('balanced_across_arms', balanced))
    if not balanced:
        reasons.append(
            f'covariate differs across arms (SMD {smd:+.3f}, z {z:+.1f}) - '
            'randomisation makes that impossible for anything measured '
            'before assignment, so this covariate is post-treatment '
            'regardless of what was declared'
        )

    return GateResult(
        covariate=declaration.covariate,
        passed=not reasons,
        reasons=tuple(reasons),
        balance_smd=smd,
        balance_z=z,
        checks=tuple(checks),
    )
