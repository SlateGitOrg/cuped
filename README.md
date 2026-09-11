# cuped

> CUPED variance reduction with a gate that rejects post-treatment covariates - the mistake that silently biases the whole method.

`COMPACT` · **Marketing Analyst** · Intermediate · ~4-5 days · Energy - retail tariff switching experiments

**Primary language:** Python
**Tags:** `experimentation`, `variance-reduction`, `cuped`, `causal-inference`, `statistics`

---

## The problem

Experiments on revenue-type metrics need enormous samples because a handful of high-value customers dominate the variance. A test that would take eleven weeks to reach adequate power usually gets called early and wrong instead, because eleven weeks is not a timeline anyone will accept.

## ⭐ The differentiator

Implements **CUPED with pre-period covariates and a rigorous pre-treatment validation gate**. The single mistake that silently invalidates the method is using a covariate measured *after* randomisation - which biases the estimate while *appearing* to improve precision, so it looks like success. The library refuses to run without a declared pre-period window and tests the bias case explicitly. A generic implementation accepts a covariate column on faith.

This is the sentence to lead with when someone asks you to walk through the
project. Everything else in this repo exists to make it true and to prove it.

## Data

A synthetic experiment generator with heavy-tailed revenue metrics, known true effects, and **both valid and deliberately invalid (post-treatment) covariates** so the gate itself can be tested.

> No paid API key is required to run or demo this project. Where a paid
> service would add value it is wired as an optional enhancement behind an
> interface with an offline mock as the default implementation.

## Stack

- Python: NumPy, SciPy
- Typer CLI plus a worked notebook
- pytest

## Core capabilities

- CUPED estimator with automatic theta estimation and variance-reduction reporting
- Pre-treatment validation gate rejecting post-randomisation covariates
- Covariate selection ranked by achieved variance reduction
- Winsorisation and trimming options with their bias implications stated, not hidden
- Readout report comparing naive against CUPED intervals and the equivalent sample-size saving

## Repository layout

```
src/cuped/
src/validate/
generator/
test/bias/
notebooks/
```

## Build plan

1. Estimator first on clean data, then verify unbiasedness across many simulations.
2. Build the gate and the invalid covariate that should trip it. The gate is the differentiator.
3. Covariate ranking, then the readout report.
4. Notebook last - it is the teaching artefact.

## Testing strategy

Assert **unbiasedness under valid covariates across 1,000 simulations**. Then assert the gate **rejects the planted post-treatment covariate** that would otherwise produce a confidently biased result - without that second test the gate is decoration.

Tests assert **correctness**, not merely that the code runs. A green suite on
this repo is a claim about behaviour under adversarial conditions; treat any
test that would pass against a deliberately broken implementation as a bug in
the test.

## Quality & safety layer

The gate is mandatory and cannot be bypassed silently; an override requires an explicit flag that is recorded in the readout.

## Measurable outcome

> A 43% variance reduction - eleven weeks of test time cut to six - alongside the rejected-covariate case showing what the gate prevents.

State it in these terms — business units, not technical ones — in your CV
bullet and in the first thirty seconds of describing the project.

## Measured results

From `python -m src.demo` (30,000 customers, seeded and reproducible; true
effect planted at GBP 1.80 per customer).

| estimator | effect | 95% interval | width |
| --- | --- | --- | --- |
| naive | 1.942 | [+1.518, +2.366] | 0.848 |
| CUPED (`pre_revenue`) | 1.754 | [+1.422, +2.086] | 0.664 |

Correlation 0.622, **38.7% of variance removed**. Required sample scales with
variance, so that is 38.7% less sample — an 11-week test becomes a 7-week one.

### What the gate prevents

The same analysis run with `post_engagement`, measured *during* the test window
and moved by the treatment — declared with an identical 28-day pre-period
window, so the metadata check sees nothing wrong:

| | `pre_revenue` (valid) | `post_engagement` (invalid) |
| --- | --- | --- |
| correlation | 0.622 | 0.976 |
| variance removed | 38.7% | 98.3% |
| interval width | 0.664 | **0.112** |
| estimated effect | 1.754 | **−6.601** |
| true effect | 1.800 | 1.800 |

The invalid covariate wins on every diagnostic an analyst actually looks at —
higher correlation, more variance removed, an interval six times tighter — and
returns an effect of the **wrong sign**, in an interval that excludes the truth
by a wide margin. Nothing in the output indicates a problem. That is why this
is a gate and not a warning.

The balance test catches it: `SMD +0.480, z +41.6`. Randomisation makes that
impossible for anything measured before assignment.

### Limits of the gate, stated plainly

- **It detects post-treatment covariates that the treatment *moves*.** One
  measured after randomisation but genuinely unaffected by the treatment will
  pass — and is still unsafe, since it can be a collider on a path between
  assignment and outcome. No quantity of data reveals that case; only the
  declaration and the judgement behind it.
- **The balance test is a hypothesis test, so it is wrong sometimes.** Measured
  false-positive rate on a valid covariate is **0.2%** (1 in 500 at n=20,000),
  against a nominal α of 0.001. That rate is asserted by a test, and it is why
  `override_gate` exists as a recorded, first-class part of the API rather than
  an afterthought — a gate that cries wolf gets overridden out of habit.
- **The balance check requires both significance and a material SMD (≥0.02).**
  A p-value-only gate rejects valid covariates more and more often as the
  sample grows, since it gains power to detect imbalance of no consequence — a
  gate that gets worse with more data is not a gate.

## Interview questions this project answers

- **How does CUPED reduce variance without biasing the estimate?**
- **What happens if your covariate is measured after randomisation?**
- **When does CUPED not help?**

## What this deliberately is *not*

- Not an experimentation platform. One estimator, done correctly.


## Run it now

```bash
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Requires Python 3.11+. The runnable core uses **only the standard
library** (including `sqlite3`), so there is nothing to install.

## Getting started

```bash
git clone <your-fork-url> cuped
cd cuped
pip install -e .
python -m generator --users 200000
cuped run --metric revenue --covariate pre_revenue --pre-window 28d
pytest test/bias              # incl. the rejected covariate case
```

Docker is supported but optional — every path above works on a plain
Windows/macOS/Linux laptop without a cloud account.

## Definition of done

- [ ] The differentiator above is implemented, and a test proves it
- [ ] The measurable outcome is produced by a command anyone can run
- [ ] `README` explains the one decision a generic version gets wrong
- [ ] CI runs the full suite on every push and is green on `main`
- [ ] A recruiter can see the headline artefact in under 60 seconds

## Licence

MIT — see [LICENSE](LICENSE).
