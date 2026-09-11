"""CUPED, and the gate that decides whether CUPED is allowed to run.

THE DIFFERENTIATOR LIVES IN `validate.py`. This module is the estimator it
guards, and is written so that the guard cannot be skipped by accident: the
only public entry point is `estimate()`, which requires a `CovariateDeclaration`
and refuses to proceed if the declaration does not clear the gate.

CUPED itself is a one-liner of statistics. Given an outcome Y and a covariate X
measured before randomisation,

    Y_adjusted = Y - theta * (X - mean(X))        theta = cov(X, Y) / var(X)

`Y_adjusted` has the same expectation as `Y` - so the treatment effect estimate
is unchanged in expectation - but variance reduced by a factor of
(1 - corr(X, Y)^2). The effect estimate is then the ordinary difference in
adjusted means.

The arithmetic is unremarkable. What makes an implementation trustworthy or
worthless is whether X is genuinely pre-treatment, and nothing in the formula
checks that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .validate import CovariateDeclaration, GateResult, run_gate
from .generator import Experiment, User


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def covariance(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)


def correlation(xs: list[float], ys: list[float]) -> float:
    denom = math.sqrt(variance(xs) * variance(ys))
    return covariance(xs, ys) / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class Estimate:
    effect: float
    standard_error: float
    low: float
    high: float
    significant: bool
    method: str
    #: Variance of the (adjusted) outcome, pooled across arms.
    outcome_variance: float

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2


@dataclass(frozen=True)
class Readout:
    naive: Estimate
    cuped: Estimate
    theta: float
    correlation: float
    gate: GateResult
    #: Whether the gate was overridden by an explicit flag.
    overridden: bool

    @property
    def variance_reduction(self) -> float:
        """Share of outcome variance removed. Equals corr^2 in expectation."""
        if self.naive.outcome_variance <= 0:
            return 0.0
        return 1.0 - self.cuped.outcome_variance / self.naive.outcome_variance

    @property
    def sample_size_saving(self) -> float:
        """Fraction of the sample CUPED makes unnecessary at equal power.

        Required n scales linearly with outcome variance, so removing a share
        `r` of the variance removes the same share of the required sample -
        which is the form the number should be reported in, because "43% less
        variance" means nothing to whoever is waiting on the result and
        "six weeks instead of eleven" means everything.
        """
        return self.variance_reduction


_Z95 = 1.959963985


def _difference_in_means(
    control: list[float], treatment: list[float], label: str,
) -> Estimate:
    effect = mean(treatment) - mean(control)
    vc, vt = variance(control), variance(treatment)
    se = math.sqrt(vc / len(control) + vt / len(treatment))
    half = _Z95 * se
    pooled = (
        (len(control) - 1) * vc + (len(treatment) - 1) * vt
    ) / max(1, len(control) + len(treatment) - 2)
    return Estimate(
        effect=effect, standard_error=se,
        low=effect - half, high=effect + half,
        significant=se > 0 and abs(effect) > half,
        method=label, outcome_variance=pooled,
    )


def compute_theta(covariates: list[float], outcomes: list[float]) -> float:
    """theta = cov(X, Y) / var(X), pooled across BOTH arms.

    Pooling matters. Estimating theta separately per arm lets the adjustment
    absorb part of the treatment effect itself, which shrinks the estimate
    toward zero - a subtle downward bias that looks like a disappointing
    result rather than a bug.
    """
    v = variance(covariates)
    return covariance(covariates, outcomes) / v if v > 0 else 0.0


class GateRejection(Exception):
    """Raised when a covariate fails pre-treatment validation.

    Deliberately an exception rather than a returned warning: a warning in a
    notebook is a warning nobody reads, and the failure mode being prevented
    produces output that looks BETTER than the correct analysis.
    """

    def __init__(self, gate: GateResult):
        self.gate = gate
        super().__init__(
            f'covariate {gate.covariate!r} rejected: '
            + '; '.join(gate.reasons)
        )


def estimate(
    experiment: Experiment,
    declaration: CovariateDeclaration,
    metric: str = 'revenue',
    override_gate: bool = False,
) -> Readout:
    """Run CUPED, subject to the gate.

    `override_gate` exists because there are legitimate cases the automated
    checks cannot clear - but it is recorded in the readout, so the override
    travels with the result rather than living in someone's shell history.
    """
    gate = run_gate(experiment, declaration, metric=metric)
    if not gate.passed and not override_gate:
        raise GateRejection(gate)

    control = experiment.arm('control')
    treatment = experiment.arm('treatment')

    def pull(users: list[User], field: str) -> list[float]:
        return [getattr(u, field) for u in users]

    all_users = list(experiment.users)
    covariates = pull(all_users, declaration.covariate)
    outcomes = pull(all_users, metric)

    theta = compute_theta(covariates, outcomes)
    cov_mean = mean(covariates)

    def adjust(users: list[User]) -> list[float]:
        return [
            getattr(u, metric) - theta * (getattr(u, declaration.covariate)
                                          - cov_mean)
            for u in users
        ]

    naive = _difference_in_means(
        pull(control, metric), pull(treatment, metric), 'naive')
    adjusted = _difference_in_means(
        adjust(control), adjust(treatment), 'cuped')

    return Readout(
        naive=naive, cuped=adjusted, theta=theta,
        correlation=correlation(covariates, outcomes),
        gate=gate, overridden=override_gate and not gate.passed,
    )
