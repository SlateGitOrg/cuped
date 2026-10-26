"""The suite is organised around one question: does the gate earn its place?

A test that only checks CUPED reduces variance would pass against an
implementation with no gate at all - and against one using the post-treatment
covariate, which reduces variance MORE. So the bias tests carry the weight.
"""

from __future__ import annotations

import math
import unittest

from src.cuped import (
    GateRejection, Readout, compute_theta, correlation, estimate, mean,
    variance,
)
from src.generator import generate
from src.ranking import rank
from src.validate import (
    CovariateDeclaration, MIN_MATERIAL_SMD, balance_check, run_gate,
)

VALID = CovariateDeclaration(
    covariate='pre_revenue',
    measurement_window_start_days_before_randomisation=28,
    measurement_window_end_days_before_randomisation=0,
    rationale='billed revenue in the 28 days before assignment',
)

INVALID = CovariateDeclaration(
    covariate='post_engagement',
    measurement_window_start_days_before_randomisation=0,
    measurement_window_end_days_before_randomisation=-14,
    rationale='engagement during the test window',
)

# The same post-treatment covariate, but declared (wrongly) as pre-treatment.
# This is the case the declaration cannot catch and the balance test must.
MISDECLARED = CovariateDeclaration(
    covariate='post_engagement',
    measurement_window_start_days_before_randomisation=28,
    measurement_window_end_days_before_randomisation=0,
    rationale='claimed to be pre-period; it is not',
)

TENURE = CovariateDeclaration(
    covariate='signup_tenure_days',
    measurement_window_start_days_before_randomisation=28,
    measurement_window_end_days_before_randomisation=0,
)


class TestTheGate(unittest.TestCase):
    """The gate rejects post-treatment covariates."""

    def test_the_honest_mistake_is_caught_by_the_declaration(self):
        exp = generate(n_users=4_000)
        gate = run_gate(exp, INVALID)
        self.assertFalse(gate.passed)
        self.assertTrue(any('does not close before randomisation' in r
                            for r in gate.reasons))

    def test_the_MISDECLARED_covariate_is_caught_by_the_data(self):
        # The declaration is a lie and looks perfect. Only the balance test
        # can see through it. Without this the gate is metadata theatre.
        exp = generate(n_users=4_000)
        gate = run_gate(exp, MISDECLARED)
        self.assertFalse(gate.passed, 'a misdeclared covariate cleared the gate')
        self.assertTrue(
            any('post-treatment' in r for r in gate.reasons),
            f'rejected, but not for the right reason: {gate.reasons}')
        self.assertGreater(abs(gate.balance_smd), MIN_MATERIAL_SMD)

    def test_estimate_REFUSES_rather_than_warns(self):
        exp = generate(n_users=4_000)
        with self.assertRaises(GateRejection):
            estimate(exp, MISDECLARED)

    def test_the_valid_covariate_passes(self):
        exp = generate(n_users=8_000)
        gate = run_gate(exp, VALID)
        self.assertTrue(gate.passed, gate.reasons)

    def test_a_valid_covariate_still_passes_at_large_n(self):
        # A p-value-only balance gate rejects everything once n is large
        # enough, because it has power to detect imbalance of no consequence.
        # A gate that breaks as data accumulates is not a gate.
        exp = generate(n_users=60_000, seed=77)
        gate = run_gate(exp, VALID)
        self.assertTrue(
            gate.passed,
            f'valid covariate rejected at n=60,000: {gate.reasons} '
            f'(SMD {gate.balance_smd:+.4f}, z {gate.balance_z:+.1f})')

    def test_the_outcome_cannot_be_its_own_covariate(self):
        exp = generate(n_users=2_000)
        self_ref = CovariateDeclaration(
            covariate='revenue',
            measurement_window_start_days_before_randomisation=28,
            measurement_window_end_days_before_randomisation=0)
        gate = run_gate(exp, self_ref)
        self.assertFalse(gate.passed)

    def test_a_window_longer_than_the_pre_period_is_rejected(self):
        exp = generate(n_users=2_000, pre_window_days=28)
        too_long = CovariateDeclaration(
            covariate='pre_revenue',
            measurement_window_start_days_before_randomisation=90,
            measurement_window_end_days_before_randomisation=0)
        self.assertFalse(run_gate(exp, too_long).passed)

    def test_a_missing_covariate_is_rejected_not_crashed_on(self):
        exp = generate(n_users=500)
        missing = CovariateDeclaration(
            covariate='not_a_field',
            measurement_window_start_days_before_randomisation=28,
            measurement_window_end_days_before_randomisation=0)
        gate = run_gate(exp, missing)
        self.assertFalse(gate.passed)

    def test_the_override_is_recorded_in_the_readout(self):
        # Overriding must be possible - and must leave a mark.
        exp = generate(n_users=4_000)
        readout = estimate(exp, MISDECLARED, override_gate=True)
        self.assertTrue(readout.overridden)
        self.assertFalse(readout.gate.passed)

    def test_override_does_not_mark_a_clean_run_as_overridden(self):
        exp = generate(n_users=4_000)
        readout = estimate(exp, VALID, override_gate=True)
        self.assertFalse(readout.overridden)


class TestWhyTheGateMatters(unittest.TestCase):
    """Prove the mistake is both harmful AND attractive.

    Either half alone is unconvincing: harmful-but-obvious needs no gate, and
    attractive-but-harmless is not worth blocking.
    """

    def test_the_bad_covariate_BIASES_the_estimate(self):
        exp = generate(n_users=30_000, true_effect=1.80, seed=5)
        good = estimate(exp, VALID)
        bad = estimate(exp, MISDECLARED, override_gate=True)

        good_error = abs(good.cuped.effect - exp.true_effect)
        bad_error = abs(bad.cuped.effect - exp.true_effect)
        self.assertLess(good_error, 0.15)
        self.assertGreater(
            bad_error, 1.0,
            f'post-treatment adjustment gave {bad.cuped.effect:.3f} against a '
            f'true effect of {exp.true_effect} - if this is close, the '
            f'generator is not planting a real trap')

    def test_the_bad_covariate_EXCLUDES_the_truth_from_its_interval(self):
        # Biased is bad. Biased and confident is what ships the wrong change.
        exp = generate(n_users=30_000, true_effect=1.80, seed=5)
        bad = estimate(exp, MISDECLARED, override_gate=True)
        self.assertFalse(
            bad.cuped.low <= exp.true_effect <= bad.cuped.high,
            'the biased interval happened to cover the truth')

    def test_the_bad_covariate_LOOKS_better_on_every_diagnostic(self):
        exp = generate(n_users=30_000, seed=5)
        good = estimate(exp, VALID)
        bad = estimate(exp, MISDECLARED, override_gate=True)

        self.assertGreater(bad.variance_reduction, good.variance_reduction)
        self.assertLess(bad.cuped.half_width, good.cuped.half_width)
        self.assertGreater(abs(bad.correlation), abs(good.correlation))


class TestTheEstimator(unittest.TestCase):

    def test_unbiased_across_many_simulations(self):
        # The central claim. One run says nothing; the spread across runs is
        # what shows the estimator is centred on the truth.
        #
        # The bar is the estimator's OWN standard error, not a round number.
        # Revenue is heavy-tailed, so a single run's error has sd ~0.5 and any
        # fixed threshold is either unfalsifiable or flaky depending on how
        # many simulations happen to be run. Three standard errors of the mean
        # is a bar that tightens automatically as the simulation count rises.
        true_effect = 1.80
        errors = []
        for seed in range(300):
            exp = generate(n_users=3_000, true_effect=true_effect, seed=seed)
            errors.append(estimate(exp, VALID).cuped.effect - true_effect)
        se = math.sqrt(variance(errors) / len(errors))
        self.assertLess(
            abs(mean(errors)), 3 * se,
            f'mean error {mean(errors):+.4f} over {len(errors)} simulations '
            f'is {abs(mean(errors)) / se:.1f} standard errors from zero')

    def test_cuped_agrees_with_naive_in_expectation(self):
        # Same expectation, smaller variance. If CUPED shifted the answer it
        # would not be variance reduction, it would be a different estimator.
        naive_errors, cuped_errors = [], []
        for seed in range(200):
            exp = generate(n_users=3_000, true_effect=1.80, seed=seed)
            r = estimate(exp, VALID)
            naive_errors.append(r.naive.effect - 1.80)
            cuped_errors.append(r.cuped.effect - 1.80)
        self.assertLess(abs(mean(naive_errors) - mean(cuped_errors)), 0.06)
        self.assertLess(variance(cuped_errors), variance(naive_errors))

    def test_variance_reduction_is_real_and_material(self):
        exp = generate(n_users=30_000, seed=3)
        r = estimate(exp, VALID)
        self.assertGreater(r.variance_reduction, 0.30)
        self.assertLess(r.cuped.half_width, r.naive.half_width)

    def test_variance_reduction_tracks_the_squared_correlation(self):
        exp = generate(n_users=40_000, seed=9)
        r = estimate(exp, VALID)
        self.assertAlmostEqual(
            r.variance_reduction, r.correlation ** 2, delta=0.04)

    def test_a_useless_covariate_reduces_almost_nothing(self):
        # CUPED is not free magic. Stating where it does not help is part of
        # knowing the method.
        exp = generate(n_users=20_000, seed=4)
        r = estimate(exp, TENURE)
        self.assertLess(r.variance_reduction, 0.15)

    def test_theta_is_pooled_not_per_arm(self):
        # A per-arm theta absorbs part of the effect. Check the pooled theta
        # against the value computed from all users directly.
        exp = generate(n_users=10_000, seed=8)
        r = estimate(exp, VALID)
        expected = compute_theta(
            [u.pre_revenue for u in exp.users],
            [u.revenue for u in exp.users])
        self.assertAlmostEqual(r.theta, expected, places=9)

    def test_zero_effect_is_not_manufactured(self):
        # Judged against the interval the estimator itself reports, rather
        # than a hand-picked tolerance. An A/A run must cover zero and must
        # not call a winner.
        exp = generate(n_users=40_000, true_effect=0.0, seed=6)
        r = estimate(exp, VALID)
        self.assertLessEqual(r.cuped.low, 0.0)
        self.assertGreaterEqual(r.cuped.high, 0.0)
        self.assertFalse(r.cuped.significant)

    def test_the_saving_is_reported_as_sample_size(self):
        exp = generate(n_users=20_000, seed=3)
        r = estimate(exp, VALID)
        self.assertAlmostEqual(r.sample_size_saving, r.variance_reduction)


class TestRanking(unittest.TestCase):

    def test_the_post_treatment_covariate_is_not_recommended(self):
        # It would rank FIRST on variance reduction. Ranking has to gate
        # before it sorts, or the ranking recommends the bug.
        exp = generate(n_users=10_000)
        ranked = rank(exp, [VALID, MISDECLARED, TENURE])
        self.assertTrue(ranked[0].eligible)
        self.assertEqual(ranked[0].covariate, 'pre_revenue')

    def test_rejected_candidates_are_shown_not_hidden(self):
        exp = generate(n_users=10_000)
        ranked = rank(exp, [VALID, MISDECLARED, TENURE])
        names = [c.covariate for c in ranked]
        self.assertIn('post_engagement', names)
        bad = next(c for c in ranked if c.covariate == 'post_engagement')
        self.assertFalse(bad.eligible)
        self.assertTrue(bad.gate.reasons)

    def test_the_rejected_one_really_would_have_won(self):
        exp = generate(n_users=10_000)
        ranked = rank(exp, [VALID, MISDECLARED, TENURE])
        bad = next(c for c in ranked if c.covariate == 'post_engagement')
        good = next(c for c in ranked if c.covariate == 'pre_revenue')
        self.assertGreater(bad.expected_variance_reduction,
                           good.expected_variance_reduction)

    def test_eligible_candidates_are_ordered_by_reduction(self):
        exp = generate(n_users=10_000)
        ranked = [c for c in rank(exp, [TENURE, VALID]) if c.eligible]
        self.assertEqual([c.covariate for c in ranked],
                         ['pre_revenue', 'signup_tenure_days'])


class TestTheGateHasAFalsePositiveRate(unittest.TestCase):
    """The balance test is a hypothesis test, so it is wrong sometimes.

    Pretending otherwise would be the same species of error the gate exists to
    catch. This is measured, bounded, and is the reason `override_gate` is part
    of the API rather than an afterthought.
    """

    def test_valid_covariates_are_rejected_at_about_the_nominal_rate(self):
        rejected = sum(
            0 if run_gate(generate(n_users=20_000, seed=s), VALID).passed
            else 1
            for s in range(300)
        )
        rate = rejected / 300
        # Nominal alpha is 0.001; the SMD floor makes the gate strictly more
        # conservative than that. 2% leaves room for simulation noise while
        # still failing loudly if the gate becomes trigger-happy.
        self.assertLess(
            rate, 0.02,
            f'the gate rejected {rejected}/300 valid covariates ({rate:.1%}) - '
            f'a gate that cries wolf gets overridden by habit, which is worse '
            f'than not having one')


class TestNumerics(unittest.TestCase):

    def test_balance_check_on_identical_arms(self):
        balanced, smd, z = balance_check([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertTrue(balanced)
        self.assertAlmostEqual(smd, 0.0)

    def test_balance_check_survives_a_tiny_sample(self):
        balanced, _, _ = balance_check([1.0], [2.0])
        self.assertTrue(balanced)

    def test_correlation_of_a_variable_with_itself(self):
        xs = [1.0, 4.0, 2.0, 8.0, 3.0]
        self.assertAlmostEqual(correlation(xs, xs), 1.0, places=9)

    def test_variance_of_a_constant_is_zero(self):
        self.assertEqual(variance([5.0] * 10), 0.0)

    def test_theta_is_zero_when_the_covariate_is_constant(self):
        self.assertEqual(compute_theta([2.0] * 5, [1.0, 5.0, 2.0, 9.0, 3.0]),
                         0.0)


if __name__ == '__main__':
    unittest.main()
