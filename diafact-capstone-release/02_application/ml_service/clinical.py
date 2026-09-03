"""
clinical.py — single source of truth for feature engineering, validation and
clinical constraints in DiaFact.

Every formula here is copied verbatim from the training pipeline
(`exp_v2/run/pre-processing.ipynb`). If a formula changes there, it must change
here, or the served model receives inputs drawn from a different distribution
than the one it was fitted on.

Training-time definitions (DO NOT "simplify"):
    bp_ratio       = systolic_bp / (diastolic_bp + 1)      <- the +1 is not optional
    pulse_pressure = systolic_bp - diastolic_bp
    bmi_age        = bmi * age
    glucose_bmi    = glucose * bmi
    cardio_risk    = hypertensive + cardiovascular_disease + stroke
    family_risk    = family_diabetes + family_hypertension
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

# Exact Boruta-selected order the XGBoost model was fitted on.
FEATURES = [
    "age", "gender", "pulse_rate", "systolic_bp", "diastolic_bp",
    "glucose", "weight", "bmi", "hypertensive", "bp_ratio",
    "pulse_pressure", "bmi_age", "glucose_bmi", "cardio_risk",
    "family_hypertension", "family_risk",
]

# Raw clinical variables an intervention can actually act on.
# `hypertensive` is deliberately excluded: it records a diagnosis, so setting it
# to zero describes a change in clinical status, not an action a patient can take.
RAW_ACTIONABLE = ["glucose", "weight", "systolic_bp", "diastolic_bp", "pulse_rate"]

# Derived features are never searched directly; they are recomputed from their
# parents at every prediction. This maps each derived feature to the raw
# variables it is built from, used to credit SHAP importance back to parents.
DERIVED_PARENTS = {
    "bmi":            ["weight"],
    "bmi_age":        ["weight"],
    "glucose_bmi":    ["glucose", "weight"],
    "bp_ratio":       ["systolic_bp", "diastolic_bp"],
    "pulse_pressure": ["systolic_bp", "diastolic_bp"],
    "cardio_risk":    [],  # parents are all fixed (diagnosis flags)
}

# Raw inputs the caller must supply.
REQUIRED_INPUTS = [
    "age", "gender", "pulse_rate", "systolic_bp", "diastolic_bp",
    "glucose", "weight", "height", "hypertensive",
    "family_diabetes", "family_hypertension",
    "cardiovascular_disease", "stroke",
]


# ---------------------------------------------------------------------------
# Input validation — physiological plausibility
# ---------------------------------------------------------------------------
# Wide survivability envelopes. Anything outside these is a data-entry error,
# not a patient, and is rejected rather than silently scored.
INPUT_RANGES = {
    "age":            (1.0, 120.0,  "years"),
    "pulse_rate":     (25.0, 220.0, "bpm"),
    "systolic_bp":    (60.0, 260.0, "mmHg"),
    "diastolic_bp":   (30.0, 160.0, "mmHg"),
    "glucose":        (1.0, 40.0,   "mmol/L"),
    "weight":         (20.0, 300.0, "kg"),
    "height":         (100.0, 230.0, "cm"),
}

BINARY_INPUTS = [
    "gender", "hypertensive", "family_diabetes",
    "family_hypertension", "cardiovascular_disease", "stroke",
]


class ValidationError(ValueError):
    """Raised when input fails clinical plausibility checks."""

    def __init__(self, errors):
        self.errors = errors if isinstance(errors, list) else [errors]
        super().__init__("; ".join(self.errors))


def validate_inputs(raw: dict) -> dict:
    """Coerce and validate a raw input payload. Returns cleaned floats."""
    errors = []

    missing = [k for k in REQUIRED_INPUTS if k not in raw or raw[k] is None or raw[k] == ""]
    if missing:
        raise ValidationError([f"Missing required field(s): {', '.join(missing)}"])

    clean = {}

    for key, (lo, hi, unit) in INPUT_RANGES.items():
        try:
            v = float(raw[key])
        except (TypeError, ValueError):
            errors.append(f"{key} must be a number")
            continue
        if v != v:  # NaN
            errors.append(f"{key} must be a number, not NaN")
            continue
        if not (lo <= v <= hi):
            errors.append(f"{key}={v:g} is outside the plausible range {lo:g}-{hi:g} {unit}")
            continue
        clean[key] = v

    for key in BINARY_INPUTS:
        try:
            v = int(raw[key])
        except (TypeError, ValueError):
            errors.append(f"{key} must be 0 or 1")
            continue
        if v not in (0, 1):
            errors.append(f"{key} must be 0 or 1, got {v}")
            continue
        clean[key] = v

    if errors:
        raise ValidationError(errors)

    # Cross-field checks
    if clean["systolic_bp"] <= clean["diastolic_bp"]:
        raise ValidationError(
            [f"systolic_bp ({clean['systolic_bp']:g}) must exceed "
             f"diastolic_bp ({clean['diastolic_bp']:g})"]
        )

    bmi = clean["weight"] / (clean["height"] / 100.0) ** 2
    if not (8.0 <= bmi <= 80.0):
        raise ValidationError(
            [f"weight/height give an implausible BMI of {bmi:.1f}; check both values"]
        )

    return clean


# ---------------------------------------------------------------------------
# Feature engineering — must mirror training exactly
# ---------------------------------------------------------------------------

def engineer(clean: dict) -> dict:
    """Build the 16-feature model row from validated raw inputs."""
    height_m = clean["height"] / 100.0
    bmi = clean["weight"] / (height_m ** 2)

    return {
        "age":                 clean["age"],
        "gender":              float(clean["gender"]),
        "pulse_rate":          clean["pulse_rate"],
        "systolic_bp":         clean["systolic_bp"],
        "diastolic_bp":        clean["diastolic_bp"],
        "glucose":             clean["glucose"],
        "weight":              clean["weight"],
        "bmi":                 bmi,
        "hypertensive":        float(clean["hypertensive"]),
        # +1 in the denominator: matches training, prevents divide-by-zero
        "bp_ratio":            clean["systolic_bp"] / (clean["diastolic_bp"] + 1.0),
        "pulse_pressure":      clean["systolic_bp"] - clean["diastolic_bp"],
        "bmi_age":             bmi * clean["age"],
        "glucose_bmi":         clean["glucose"] * bmi,
        "cardio_risk":         float(clean["hypertensive"]
                                    + clean["cardiovascular_disease"]
                                    + clean["stroke"]),
        "family_hypertension": float(clean["family_hypertension"]),
        "family_risk":         float(clean["family_diabetes"]
                                    + clean["family_hypertension"]),
    }


def rebuild_from_raw(raw_rows, ctx):
    """
    Reconstruct full 16-feature frames from raw actionable variables only.

    `ctx` carries the patient's fixed context (age, gender, height, diagnosis
    flags). This is what makes a feature-inconsistent counterfactual impossible:
    bmi, bp_ratio, pulse_pressure, bmi_age and glucose_bmi are always recomputed
    from whatever the search proposed, never carried over or varied directly.
    """
    import pandas as pd

    df = raw_rows.reset_index(drop=True)
    X = pd.DataFrame(index=range(len(df)), columns=FEATURES, dtype=float)

    for f in RAW_ACTIONABLE:
        X[f] = df[f].astype(float).values

    X["age"] = ctx["age"]
    X["gender"] = ctx["gender"]
    X["hypertensive"] = ctx["hypertensive"]
    X["family_hypertension"] = ctx["family_hypertension"]
    X["family_risk"] = ctx["family_risk"]
    X["cardio_risk"] = ctx["cardio_risk"]

    h2 = ctx["height_m"] ** 2
    X["bmi"] = X["weight"] / h2
    X["bp_ratio"] = X["systolic_bp"] / (X["diastolic_bp"] + 1.0)
    X["pulse_pressure"] = X["systolic_bp"] - X["diastolic_bp"]
    X["bmi_age"] = X["bmi"] * ctx["age"]
    X["glucose_bmi"] = X["glucose"] * X["bmi"]

    return X[FEATURES].astype(float)


# ---------------------------------------------------------------------------
# Counterfactual search constraints
# ---------------------------------------------------------------------------
# Direction of clinical improvement for each actionable variable. A
# recommendation may only move a variable the "safe" way: -1 means only
# decrease is allowed, 0 means either direction is acceptable.
#
# This closes the gap the paper flags as a limitation: without it, the search
# happily lowers predicted risk by *raising* blood pressure, which happened in
# 75% of unconstrained and 52% of SHAP-guided counterfactuals.
MONOTONIC_DIRECTION = {
    "glucose":      -1,
    "systolic_bp":  -1,
    "diastolic_bp": -1,
    "pulse_rate":   -1,
    "weight":        0,   # underweight patients should not be told to lose more
}


def clinical_bounds(ctx: dict, current: dict) -> dict:
    """
    Per-patient search bounds: clinician-plausible targets, not survivability
    limits, and never worse than where the patient already is.

    Targets follow standard guidance:
      glucose      -> down toward the 5.5 mmol/L non-diabetic fasting range
      systolic BP  -> down toward 120 (age-relaxed to 140 over 65)
      diastolic BP -> down toward 80
      pulse        -> down toward 60-90
      weight       -> only downward, and never below BMI 18.5 (underweight)
    """
    age = ctx["age"]
    h2 = ctx["height_m"] ** 2

    sys_floor = 110.0 if age <= 65 else 120.0
    dia_floor = 70.0
    weight_floor = max(18.5 * h2, 30.0)   # never recommend underweight

    bounds = {
        "glucose":      [3.9, current["glucose"]],
        "systolic_bp":  [sys_floor, current["systolic_bp"]],
        "diastolic_bp": [dia_floor, current["diastolic_bp"]],
        "pulse_rate":   [55.0, current["pulse_rate"]],
        "weight":       [weight_floor, current["weight"]],
    }

    # A bound may not invert if the patient is already at or below target.
    for f, (lo, hi) in bounds.items():
        if lo > hi:
            bounds[f] = [hi, hi]

    # If a patient is already underweight, weight is not a lever at all.
    if current["weight"] <= weight_floor:
        bounds["weight"] = [current["weight"], current["weight"]]

    return bounds


def violates_direction(original: dict, cf: dict, tol: float = 1e-6) -> bool:
    """True if any change moves a variable in the clinically wrong direction."""
    for f, d in MONOTONIC_DIRECTION.items():
        if d == -1 and cf[f] > original[f] + tol:
            return True
    return False
