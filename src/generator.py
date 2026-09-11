"""Synthetic tariff-switching experiment with planted ground truth.

The generator exists so the gate can be tested. It emits, for every user:

  * ``pre_revenue`` - revenue in the 28 days BEFORE randomisation. A valid
    covariate: correlated with the outcome, and causally upstream of the
    assignment, so conditioning on it cannot induce bias.

  * ``post_engagement`` - measured AFTER randomisation and itself affected by
    the treatment. An INVALID covariate. It is deliberately made *more*
    correlated with the outcome than ``pre_revenue`` is, because that is what
    makes the mistake attractive: the analyst picks it precisely because it
    reduces variance more, and is rewarded with a tighter interval around the
    wrong number.

  * ``signup_tenure_days`` - a valid but weakly correlated covariate, present
    so covariate ranking has something to rank.

Revenue is heavy-tailed (lognormal), which is the whole reason CUPED is worth
doing on this metric: a handful of large accounts dominate the variance and
the naive test needs an implausible sample size.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    user_id: int
    variant: str
    revenue: float
    pre_revenue: float
    post_engagement: float
    signup_tenure_days: float


@dataclass(frozen=True)
class Experiment:
    users: tuple[User, ...]
    #: The effect actually planted, in absolute revenue units per user.
    true_effect: float
    pre_window_days: int

    def arm(self, variant: str) -> list[User]:
        return [u for u in self.users if u.variant == variant]


def _lognormal(rnd: random.Random, mu: float, sigma: float) -> float:
    return math.exp(rnd.gauss(mu, sigma))


def generate(
    n_users: int = 20_000,
    true_effect: float = 1.80,
    seed: int = 20260911,
    pre_window_days: int = 28,
    covariate_strength: float = 0.85,
) -> Experiment:
    """Generate one experiment.

    ``covariate_strength`` is the share of a user's latent spending propensity
    that carries from the pre-period into the test period. Real pre/post
    correlation on retail energy revenue sits around 0.6-0.8; the variance
    reduction CUPED achieves is roughly the square of that correlation, which
    is why the method is worth the trouble here and not worth it on a metric
    with no pre-period analogue.
    """
    rnd = random.Random(seed)
    users: list[User] = []

    for i in range(n_users):
        variant = 'treatment' if rnd.random() < 0.5 else 'control'

        # Latent propensity to spend. Persists across periods - this is what
        # makes the pre-period informative at all.
        propensity = rnd.gauss(0.0, 1.0)

        pre_revenue = _lognormal(rnd, 3.0 + 0.60 * propensity, 0.30)

        # Period-specific noise, partly shared with the pre-period through
        # `propensity` and partly fresh.
        carried = covariate_strength * propensity
        fresh = math.sqrt(max(0.0, 1.0 - covariate_strength ** 2)) \
            * rnd.gauss(0.0, 1.0)
        revenue = _lognormal(rnd, 3.0 + 0.60 * (carried + fresh), 0.30)

        # THE TRAP. Engagement is measured during the test window and is moved
        # by the treatment, so it is a descendant of the assignment. It is
        # also strongly correlated with revenue, which is exactly why someone
        # reaches for it.
        treated = variant == 'treatment'
        if treated:
            revenue += true_effect
        post_engagement = (
            0.85 * revenue
            + (6.0 if treated else 0.0)
            + rnd.gauss(0.0, 2.0)
        )

        users.append(User(
            user_id=i,
            variant=variant,
            revenue=revenue,
            pre_revenue=pre_revenue,
            post_engagement=post_engagement,
            signup_tenure_days=abs(rnd.gauss(400, 250)) + 30 * propensity,
        ))

    return Experiment(
        users=tuple(users),
        true_effect=true_effect,
        pre_window_days=pre_window_days,
    )
