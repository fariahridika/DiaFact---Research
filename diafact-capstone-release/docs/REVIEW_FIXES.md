# Reviewer-Response Analysis — TEHI 2026 Paper

All findings below were **computed from your own `exp_v2` artifacts**, not inferred.
Every published number was reproduced exactly before any new claim was made.

Reproduction check (exact match to the paper):

| Quantity | Recomputed | Published |
|---|---|---|
| XGBoost P / R / F1 @0.5 | 0.3412 / 0.4203 / 0.3766 | identical |
| LightGBM, CatBoost @0.5 | identical | identical |
| Alignment mean (std / shap) | 0.7524 / 0.6605 | identical |
| Proximity, sparsity, validity | identical | identical |

---

## FINDING 1 — The alignment metric does not match Equation 1, and is broken

**This is the most serious issue in the paper, and it is in your favour to fix.**

`cf.ipynb`, cell 12, computes:

```
Alignment = Σ|SHAP_f| over CHANGED modifiable features
            ────────────────────────────────────────────
            Σ|SHAP_f| over ALL modifiable features
```

The paper's Equation 1 says something completely different:
`|F_changed ∩ F_top-6| / |F_changed|`.

### Why the implemented version is invalid

The denominator is **constant for a given patient** — it does not depend on the
counterfactual at all. The numerator only ever **grows** as more features change.
Therefore **changing more features mechanically raises the score**. Verified on
patient 460:

| features changed | score |
|---|---|
| top 1 | 0.4241 |
| top 3 | 0.6841 |
| top 6 | 0.8378 |
| **all 12** | **1.0000** |

A counterfactual that changes *everything* scores a perfect 1.0. The metric is
anti-correlated with sparsity **by construction**.

Empirically: Spearman ρ(n_changed, alignment) = **+0.560, p = 2.9e-11**.

### Consequence

Standard DiCE changes 9.65 features on average; SHAP-guided changes 5.88. The
paper's "uncomfortable finding" — that Standard DiCE scores *higher* alignment —
is **an artifact of the metric rewarding more changes**, not a property of the
search. The most-praised passage of the paper is reporting a bug.

### The honest fix (important — do not simply switch to Eq. 1)

Equation 1 as written is **also** confounded, just in the opposite direction
(its denominator `|F_changed|` *penalises* changing more), which is why it
flatters your method. Swapping to it would replace one biased metric with
another that happens to favour your thesis — a reviewer will catch this.

| Metric | standard | shap-guided | ρ with n_changed |
|---|---|---|---|
| As-implemented (published) | 0.7524 | 0.6605 | **+0.560** confounded |
| Paper's Eq. 1 (top-6 of 16) | 0.4167 | 0.7125 | **−0.622** confounded |
| Eq. 1 over modifiable only | 0.4938 | 0.8335 | **−0.792** confounded |
| **SHAP-mass concentration** | **0.7803** | **0.8117** | **+0.162 (n.s.)** neutral |

**Recommended metric — sparsity-neutral concentration ratio:**

```
Alignment = Σ|SHAP_f| over changed features
            ──────────────────────────────────────────────────────
            Σ of the |F_changed| largest |SHAP_f| among modifiable
```

"Of the SHAP mass you *could* have captured by changing this many features
optimally, how much did you actually capture?" This is in [0,1], is not gameable
by changing more or fewer features, and is genuinely non-circular.

**Under this metric: 0.8117 vs 0.7803, paired Wilcoxon p = 0.368 — not significant.**

### What this means for the paper's story

You cannot claim better alignment. You **can** claim something cleaner and still
novel:

> SHAP guidance reaches the same target prediction with **40% less movement** and
> **39% fewer variables changed**, while the changes it makes are **equally well
> targeted** (alignment statistically indistinguishable, p = 0.37).

That is a parsimony result, not an alignment result — and it is fully supported.

---

## FINDING 2 — Statistical tests: your headline claims hold up

Paired Wilcoxon on per-patient means (n = 20), bootstrap CIs (20,000 resamples):

| Metric | Standard | SHAP-guided | Δ | 95% CI | p |
|---|---|---|---|---|---|
| L1 proximity | 179.72 | 107.89 | −40.0% | [12.42, 131.70] | **0.0020** |
| Sparsity | 0.8042 | 0.4903 | −39.0% | [0.282, 0.344] | **0.00009** |
| Alignment (neutral) | 0.7803 | 0.8117 | +0.031 | [−0.015, +0.088] | 0.368 n.s. |

**Add these to Table 3.** The two headline numbers are statistically solid.

---

## FINDING 3 — Derived-feature inconsistency is near-universal (**RERUN COMPLETE — fixed**)

The paper says DiCE "*sometimes* treats derived features as independent."
Measured across all 120 counterfactuals:

| | physically inconsistent |
|---|---|
| Standard DiCE | **60 / 60 (100%)** |
| SHAP-guided DiCE | **57 / 60 (95%)** |

Breakdown (standard): `bmi_age` violated in 56, `bp_ratio` in 54, `glucose_bmi`
in 38, `pulse_pressure` in 12.

Worse — **weight and BMI move incoherently** (height is fixed, so they must move
together) in **33%** of standard and **70%** of SHAP-guided counterfactuals.
Your own case study contains one: patient 460, weight 54.0 → 52.0 (down) while
BMI 22.49 → 27.17 (up). Impossible.

### Impact on the validity claim

Repairing consistency (recompute derived features from raw, re-predict):

| | validity as published | after repair |
|---|---|---|
| Standard | 98.3% | **90.0%** |
| SHAP-guided | 98.3% | **75.0%** |

Robust to how `cardio_risk` is handled (88.3% / 70.0% under the stricter variant).

**The 98.3% validity depends on physically impossible feature combinations.**
SHAP-guided is hurt more because, changing fewer features, it leans harder on
"free" derived-feature movement to cross the boundary.

The good news: measured on genuinely actionable raw variables after repair, your
advantage **grows** — proximity −46.5%, sparsity −42.0%.

**Fix:** regenerate counterfactuals with DiCE searching *only* raw variables and
derived features recomputed inside the prediction loop. This makes inconsistency
impossible by construction.

### ✅ RERUN DONE — and the corrected results are stronger than the published ones

I regenerated all counterfactuals this way (`rerun_outputs/`). DiCE now varies only
the 6 raw actionable variables (`glucose, weight, systolic_bp, diastolic_bp,
pulse_rate, hypertensive`); `bmi`, `bp_ratio`, `pulse_pressure`, `bmi_age`,
`glucose_bmi`, `cardio_risk` are recomputed from them at every prediction.
SHAP importance of derived features is credited back to their raw parents to rank
what SHAP-guided DiCE may touch (top-3).

| | Standard | SHAP-guided | Δ | 95% CI | p |
|---|---|---|---|---|---|
| **Validity** | **100%** | **100%** | — | — | — |
| Proximity (L1) | 19.53 | **8.12** | **−58.5%** | [7.33, 16.33] | **<0.00001** |
| Sparsity | 0.708 | **0.379** | **−46.9%** | [0.279, 0.388] | **0.00009** |
| Alignment (neutral) | 0.845 | **0.932** | **+10.6%** | [0.030, 0.152] | **0.00169** |

Every headline claim improves, and **all three are now statistically significant
in your favour** — including alignment, which the broken metric had inverted:

* Validity rises to **100% for both methods** (vs 98.3% published, which depended
  on impossible profiles, and vs 90%/75% when the old CFs are repaired).
* Proximity advantage grows from 40.0% to **58.5%**.
* Sparsity advantage grows from 39.0% to **46.9%**.
* **Alignment now favours SHAP-guided, significantly** (p = 0.0017) — on a
  sparsity-neutral metric, so this is a real effect, not a normalisation artifact.

Two caveats to disclose honestly:
1. **Coverage cost.** SHAP-guided produced 58/60 counterfactuals; patients 437 and
   672 yielded only 2 of 3 requested. Restricting to 3 features occasionally makes
   the target unreachable. Report as 96.7% coverage.
2. **Residual confound.** ρ(sparsity, concentration) = −0.292 (p = 0.0013) — much
   smaller than the discarded metrics (+0.56 / −0.62 / −0.79) but not exactly zero.
   State it rather than claim perfect neutrality.

### Replacement case study — patient 460 (use this; it is far stronger)

Original: 60-year-old woman, 54.0 kg, BMI 22.49 (implied height 1.55 m),
glucose 11.3 mmol/L, BP 145/87, pulse 84.

**SHAP-guided (consistent):**
- **lower fasting glucose 11.3 → 7.0 mmol/L — a single change**
- or glucose 11.3 → 7.3 with weight 54 → 53 kg
- or glucose 11.3 → 7.0 with weight 54 → 53 kg

**Standard (consistent):** all three counterfactuals change **5 variables**, and
every one of them sets `hypertensive` 1 → 0 — i.e. "stop having a hypertension
diagnosis," which is not an action a patient can take. One also asks her to
**gain** weight (54.0 → 55.1 kg) while lowering diabetes risk.

This is a much cleaner illustration of the paper's thesis than the original
(mis-stated) case study: one actionable, clinically sensible instruction versus
five, one impossible and one counterproductive. It also gives you a concrete
finding for the Discussion: `hypertensive` is a **diagnosis flag, not a lifestyle
lever**, and arguably should be frozen alongside age and gender in future work.

---

## FINDING 4 — Sparsity denominator is wrong in the text

The paper says sparsity is over "the ten features it actually treated as
continuously searchable" and reports "8 of the 10."

The code divides by **12**: `sparsity = n_changed / 12`, giving 9.65/12 → 5.88/12.

The stated justification is also false. The paper claims `hypertensive` and
`cardio_risk` "essentially never varied":

| feature | standard | shap-guided |
|---|---|---|
| hypertensive | changed in 22% | 13% |
| cardio_risk | changed in 28% | 38% |

That is not "essentially never." **Correct the text to 12 features and delete the
justification**, or re-derive the metric over 10 and re-run. The ratios (−39%) are
unaffected; only the "8 of 10" phrasing is wrong.

---

## FINDING 5 — Case study (patient 460) has factual errors

| Paper says | Actual (index 460) |
|---|---|
| weight 52 kg | **54.0 kg** |
| diastolic BP 96 mmHg | **87.0** — 96 is the *counterfactual* value, not the original |
| SHAP-guided "targeted only her three highest-impact features" | **6 features changed** (adds `bp_ratio`, `bmi_age`, `glucose_bmi`) |
| Standard "touching 8 variables" | **9 variables** |

The "three features" claim was produced by silently dropping the three derived
features that also moved. Also, in that counterfactual `glucose_bmi` = 45.15
while `glucose × bmi` = 8.3 × 21.41 = 177.7 — internally contradictory.

**This section must be rewritten against the actual data.** It is the most
checkable part of the paper.

---

## FINDING 6 — Severe multicollinearity (explains two of your own results)

Correlations on the training set, and VIFs:

| pair | r |
|---|---|
| family_hypertension ↔ family_risk | **0.992** |
| hypertensive ↔ cardio_risk | **0.937** |
| bp_ratio ↔ pulse_pressure | 0.865 |
| weight ↔ bmi | 0.852 |
| age ↔ bmi_age | 0.846 |
| glucose ↔ glucose_bmi | 0.829 |

`pulse_pressure` = systolic − diastolic is an **exact linear combination** →
VIF = ∞ (also for systolic_bp, diastolic_bp). Other VIFs: glucose_bmi 52.5,
bmi_age 38.7, family_risk 62.5.

Because `family_diabetes`, `cardiovascular_disease` and `stroke` were rejected by
Boruta, `family_risk` collapses to ≈ `family_hypertension` and `cardio_risk`
collapses to ≈ `hypertensive` — you are carrying **near-duplicate columns**.

This directly explains two things the paper reports but does not connect:
1. **A3 ablation** — removing engineered features *improves* metrics. Of course:
   they are redundant.
2. **"Raw weight outranks BMI"** (0.896 vs 0.453) — this is SHAP splitting credit
   between two variables correlated at r = 0.85, not a physiological insight.
   **Soften this claim**; it is an artifact, not a finding.

---

## FINDING 7 — Threshold tuning (**RERUN COMPLETE**) — the model choice is refuted

AutoGluon reproduced its published metrics **exactly** (P 0.7083, R 0.2464,
F1 0.3656, AUC 0.8858, AUC-PR 0.4882, Brier 0.0438), confirming a clean load.

| model | setting | thr | P | R | **F1** |
|---|---|---|---|---|---|
| XGBoost | @0.50 (published) | 0.500 | 0.3412 | 0.4203 | 0.3766 |
| XGBoost | F1-optimal | 0.505 | 0.3452 | 0.4203 | 0.3791 |
| LightGBM | F1-optimal | 0.155 | 0.3058 | 0.5362 | 0.3895 |
| CatBoost | F1-optimal | 0.155 | 0.2835 | 0.5217 | 0.3673 |
| **AutoGluon** | **F1-optimal** | **0.300** | **0.6222** | **0.4058** | **0.4912** |

**Recall-matched comparison** — every model forced to XGBoost's own recall (0.4203):

| model | thr | recall | **precision** | F1 |
|---|---|---|---|---|
| XGBoost | 0.510 | 0.4203 | 0.3452 | 0.3791 |
| LightGBM | 0.285 | 0.4203 | 0.3152 | 0.3602 |
| CatBoost | 0.295 | 0.4203 | 0.3187 | 0.3625 |
| **AutoGluon** | **0.280** | **0.4203** | **0.5577** | **0.4793** |

### The claim does not survive

At **identical recall**, AutoGluon delivers precision **0.5577 vs XGBoost's 0.3452**
— a 62% improvement, and it wins on every threshold-independent metric too.
Tuned F1: **0.4912 vs 0.3791**. Even tuned *LightGBM* (0.3895) beats the 0.3766
the paper used to justify XGBoost.

**"XGBoost caught more of the minority class" is an artifact of leaving the
threshold at 0.5.** It cannot be defended and must be removed.

This also resolves FINDING 8: AutoGluon at matched recall cuts false positives
roughly in half (precision 0.35 → 0.56), so the precision objection is answerable
— just not while claiming XGBoost was the better classifier.

The defensible version is *not* "XGBoost had the best F1" but:

> AutoGluon is the stronger classifier at every operating point: at matched recall
> (0.42) it reaches precision 0.558 against XGBoost's 0.345. Its stacked ensemble
> is nonetheless incompatible with exact SHAP TreeExplainer, which the prescriptive
> stage requires, so we use XGBoost as the base model and accept a measurable loss
> of precision as the price of exact Shapley attribution.

That is honest, quantified, and survives review. The current framing does not.

You now have the number to put on that trade-off — use it rather than hide it.

---

## FINDING 8 — Precision cost is never discussed

XGBoost @0.5 has precision **0.3412** — about **two-thirds of the patients handed
a prescription are false positives**. The whole prescriptive apparatus sits on
this number and the paper never engages it. Add a sentence to Discussion naming
the cost (unnecessary anxiety, unwarranted lifestyle burden) as the price of
recall-first screening.

---

## Corrected results table — drop-in replacement for Table 3

Physically-consistent counterfactuals, sparsity-neutral alignment, paired
Wilcoxon on per-patient means (n = 20), bootstrap CIs (20,000 resamples):

| Strategy | Validity | Coverage | Proximity (L1) ↓ | Sparsity ↓ | Alignment ↑ |
|---|---|---|---|---|---|
| Standard DiCE | 100% | 60/60 | 19.53 | 0.708 | 0.845 |
| SHAP-Guided DiCE | 100% | 58/60 | **8.12** | **0.379** | **0.932** |
| Difference | — | −3.3% | **−58.5%** | **−46.9%** | **+10.6%** |
| *p* (paired Wilcoxon) | — | — | **<0.00001** | **0.00009** | **0.00169** |
| 95% CI of difference | — | — | [7.33, 16.33] | [0.279, 0.388] | [0.030, 0.152] |

Verified: 0/118 of these counterfactuals contain a derived-feature or weight-BMI
inconsistency, versus 100%/95% in the published set.

---

## What to change in the paper — priority order

**Both reruns are done — no Kaggle needed. Everything below is editing.**

1. **Rebuild Section 4.3 + the Discussion paragraph around it.** Replace the
   broken metric with the concentration ratio, fix Equation 1 to match the code,
   and delete the "standard DiCE wins alignment" narrative — with consistent
   counterfactuals SHAP-guided wins alignment significantly (p = 0.0017).
2. **Swap in the corrected Table 3** above, including p-values and CIs.
3. **Rewrite the model-selection justification** (Section 4.1 + Discussion
   limitation 1) around the recall-matched AutoGluon comparison.
4. **Rewrite the patient-460 case study** using the new single-change result.
5. **Move the derived-feature issue** out of "limitations" and into the method as
   a solved problem; report validity 100% and coverage 58/60.
6. **Fix the sparsity denominator** (12, not 10) and delete the false
   "essentially never varied" claim.
7. **Add a multicollinearity paragraph**; soften the weight-vs-BMI claim.
8. Add the precision-cost sentence, data/code availability, seeds, versions.

---

## Files

**`rerun_outputs/`** — all generated on your saved artifacts:
- `table_threshold_tuned.csv`, `table_recall_matched.csv` — Part 1
- `cf_standard_consistent.csv`, `cf_shap_guided_consistent.csv` — new CFs
- `cf_metrics_consistent.csv`, `table_cf_consistent.csv` — per-CF metrics
- `probs.npy` — test-set probabilities for all four models

**`kaggle_rerun.py`** — the same pipeline, path-parameterised, if you want to
reproduce on Kaggle or hand it to a co-author.

**Reproducibility note for the paper:** these runs used Python 3.12,
AutoGluon 1.5.0, xgboost 3.1.3, dice-ml (genetic method), seed 42. The
AutoGluon artifacts were trained on Linux/Python 3.12.13 and load on
Windows/3.12.12 only after `pathlib.PosixPath = pathlib.WindowsPath`; published
metrics reproduced exactly, confirming the load is faithful.
