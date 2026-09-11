"""Rank candidate covariates by achievable variance reduction.

The ranking runs the gate FIRST and ranks only what survives. Ranking by
variance reduction alone would put the post-treatment covariate at the top of
the list every time - it is the most correlated with the outcome precisely
because the treatment moved both - and a ranked list is a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cuped import correlation
from .validate import CovariateDeclaration, GateResult, run_gate


@dataclass(frozen=True)
class Candidate:
    covariate: str
    correlation: float
    #: Expected share of variance removed, = corr^2.
    expected_variance_reduction: float
    gate: GateResult

    @property
    def eligible(self) -> bool:
        return self.gate.passed


def rank(
    experiment,
    declarations: list[CovariateDeclaration],
    metric: str = 'revenue',
) -> list[Candidate]:
    """Return candidates, eligible ones first, each ordered by reduction."""
    outcomes = [getattr(u, metric) for u in experiment.users]
    out: list[Candidate] = []

    for declaration in declarations:
        gate = run_gate(experiment, declaration, metric=metric)
        if not any(name == 'covariate_exists' and ok
                   for name, ok in gate.checks):
            continue
        values = [getattr(u, declaration.covariate) for u in experiment.users]
        r = correlation(values, outcomes)
        out.append(Candidate(
            covariate=declaration.covariate,
            correlation=r,
            expected_variance_reduction=r * r,
            gate=gate,
        ))

    # Eligible first, then by reduction. Rejected candidates are RETAINED in
    # the output rather than filtered away, so the report can show what was
    # excluded and why - a silently shorter list teaches nobody anything.
    out.sort(key=lambda c: (not c.eligible, -c.expected_variance_reduction))
    return out
