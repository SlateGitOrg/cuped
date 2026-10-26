"""The 60-second artefact: the saving, and the mistake that fakes a bigger one.

Run: python -m src.demo
"""

from __future__ import annotations

from .cuped import GateRejection, estimate
from .generator import generate
from .ranking import rank
from .validate import CovariateDeclaration

PRE_REVENUE = CovariateDeclaration(
    covariate='pre_revenue',
    measurement_window_start_days_before_randomisation=28,
    measurement_window_end_days_before_randomisation=0,
    rationale='billed revenue in the 28 days before assignment',
)

TENURE = CovariateDeclaration(
    covariate='signup_tenure_days',
    measurement_window_start_days_before_randomisation=28,
    measurement_window_end_days_before_randomisation=0,
    rationale='account age at assignment',
)

POST_ENGAGEMENT = CovariateDeclaration(
    covariate='post_engagement',
    measurement_window_start_days_before_randomisation=28,
    measurement_window_end_days_before_randomisation=0,
    rationale='DECLARED as pre-period. It is not.',
)

BAR = '=' * 74
RULE = '-' * 74

exp = generate(n_users=30_000, true_effect=1.80, seed=5)

print()
print('  CUPED - variance reduction, and the gate that keeps it honest')
print(f'  {BAR}')
print(f'  Tariff-switching experiment, {len(exp.users):,} customers, '
      f'revenue metric.')
print(f'  True effect planted by the generator: '
      f'{exp.true_effect:.2f} per customer.\n')

good = estimate(exp, PRE_REVENUE)

# --- the saving ------------------------------------------------------------
print('  WHAT CUPED BUYS')
print(f'  {RULE}')
print(f'  {"estimator":<16}{"effect (GBP)":<15}{"95% interval":<24}interval width')
for label, est in (('naive', good.naive), ('CUPED', good.cuped)):
    print(f'  {label:<16}{est.effect:<15.3f}'
          f'{f"[{est.low:+.3f}, {est.high:+.3f}]":<24}'
          f'{est.high - est.low:.3f}')
print(f'\n    correlation with the pre-period covariate  '
      f'{good.correlation:.3f}')
print(f'    variance removed                          '
      f'{good.variance_reduction:.1%}')

WEEKS = 11
saved = WEEKS * (1 - good.sample_size_saving)
print(f'\n    Required sample scales with variance, so removing '
      f'{good.variance_reduction:.0%} of it')
print(f'    removes {good.variance_reduction:.0%} of the sample: an '
      f'{WEEKS}-week test becomes a {saved:.0f}-week one.')
print('    Sample size, not variance, is what a team schedules around.\n')

# --- the trap --------------------------------------------------------------
print('  THE MISTAKE THE GATE EXISTS TO STOP')
print(f'  {RULE}')
bad = estimate(exp, POST_ENGAGEMENT, override_gate=True)
print('  The same analysis using `post_engagement` - measured DURING the')
print('  test window, and moved by the treatment:\n')
print(f'  {"":<22}{"pre_revenue (valid)":<24}post_engagement (invalid)')
rows = [
    ('correlation', f'{good.correlation:.3f}', f'{bad.correlation:.3f}'),
    ('variance removed', f'{good.variance_reduction:.1%}',
     f'{bad.variance_reduction:.1%}'),
    ('interval width', f'{good.cuped.high - good.cuped.low:.3f}',
     f'{bad.cuped.high - bad.cuped.low:.3f}'),
    ('estimated effect', f'{good.cuped.effect:.3f}',
     f'{bad.cuped.effect:.3f}'),
    ('true effect', f'{exp.true_effect:.3f}', f'{exp.true_effect:.3f}'),
]
for name, a, b in rows:
    print(f'  {name:<22}{a:<24}{b}')

covers = bad.cuped.low <= exp.true_effect <= bad.cuped.high
print(f'\n    The invalid covariate wins on EVERY diagnostic an analyst looks')
print(f'    at: higher correlation, more variance removed, a tighter')
print(f'    interval. And its answer is wrong by GBP '
      f'{abs(bad.cuped.effect - exp.true_effect):.2f}, with a 95% interval')
print(f'    of [{bad.cuped.low:+.3f}, {bad.cuped.high:+.3f}] that '
      f'{"covers" if covers else "DOES NOT COVER"} the truth.')
print('    Nothing in the output says anything is wrong. That is the point.\n')

# --- the gate --------------------------------------------------------------
print('  WHAT THE GATE SAYS')
print(f'  {RULE}')
gate = bad.gate
try:
    estimate(exp, POST_ENGAGEMENT)
    print('  ...it let it through. That is a bug.')
except GateRejection as rejection:
    print('  estimate(exp, post_engagement)')
    print(f'    -> GateRejection: {rejection.gate.covariate}')
    for reason in rejection.gate.reasons:
        print(f'       {reason}')
print()
print('  Check-by-check:')
for name, ok in gate.checks:
    print(f'    [{"PASS" if ok else "FAIL"}] {name}')
print()
print('  Note the declaration is IDENTICAL to the valid one - a 28-day')
print('  pre-period window. The metadata check cannot see the problem.')
print('  The balance test can: randomisation makes it impossible for a')
print('  genuinely pre-treatment variable to differ across arms.\n')

# --- ranking ---------------------------------------------------------------
print('  COVARIATE RANKING')
print(f'  {RULE}')
print(f'  {"covariate":<22}{"corr":<9}{"var. removed":<16}status')
for candidate in rank(exp, [PRE_REVENUE, POST_ENGAGEMENT, TENURE]):
    status = 'eligible' if candidate.eligible else 'REJECTED by gate'
    print(f'  {candidate.covariate:<22}{candidate.correlation:<9.3f}'
          f'{candidate.expected_variance_reduction:<16.1%}{status}')
print('\n    Ranking gates before it sorts. On raw variance reduction the')
print('    rejected covariate comes first - so a ranking that skipped the')
print('    gate would not just permit the mistake, it would recommend it.\n')

# --- limits ----------------------------------------------------------------
print('  WHAT THE GATE CANNOT DO')
print(f'  {RULE}')
print('  The balance test detects post-treatment covariates that the')
print('  treatment MOVES. One measured after randomisation but genuinely')
print('  unaffected by the treatment passes - and is still unsafe, because')
print('  it can sit on a path between assignment and outcome. No quantity')
print('  of data reveals that case. The declaration, and the judgement')
print('  behind it, is the only thing standing there.\n')
