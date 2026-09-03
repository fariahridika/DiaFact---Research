"""
DiaFact ML Service — Flask API

Serves calibrated XGBoost risk, exact SHAP attribution, and consistency-safe
SHAP-guided counterfactual recommendations, matching the published method.

Design notes that are easy to get wrong, and were:

*   Feature formulas live in `clinical.py` and mirror the training notebook
    exactly. `bp_ratio` uses (diastolic + 1) and `cardio_risk` is built from
    hypertensive + cardiovascular_disease + stroke. Getting these wrong feeds
    the model values it never saw during training.

*   The counterfactual search ranges over RAW variables only. Every engineered
    feature is recomputed from them before each prediction, so a physically
    impossible profile -- losing weight while gaining BMI -- cannot be
    produced. Letting DiCE vary `bmi` directly and patching it up afterwards
    does not work: it drops validity to 75-90%.

*   `hypertensive` is frozen. It records a diagnosis, so flipping it to 0 is
    not an action a patient can take.

*   Recommendations are direction-constrained. Without this the search happily
    lowers predicted risk by *raising* blood pressure.

Run:  python app.py         (artifacts must exist -- see prepare_artifacts.py)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

import dice_ml
import shap
import xgboost as xgb
from dice_ml import Dice

# DiCE writes tqdm progress bars straight to stderr on every request, which
# turns the service log into noise. Neutralise them before Dice is constructed.
try:
    import tqdm as _tqdm
    from functools import partialmethod
    _tqdm.tqdm.__init__ = partialmethod(_tqdm.tqdm.__init__, disable=True)
except Exception:                                              # noqa: BLE001
    pass

from clinical import (FEATURES, RAW_ACTIONABLE, DERIVED_PARENTS,
                      MONOTONIC_DIRECTION, ValidationError,
                      clinical_bounds, engineer, rebuild_from_raw,
                      validate_inputs, violates_direction)

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("diafact.ml")

app = Flask(__name__)

# CORS restricted to the dev frontend by default; override in production.
_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
CORS(app, resources={r"/*": {"origins": [o.strip() for o in _origins.split(",") if o.strip()]}})

# Reject oversized bodies outright (this endpoint needs well under 8 KB).
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

# DiCE's genetic search is not thread-safe and is CPU-bound; serialise it.
_cf_lock = threading.Lock()
CF_TIMEOUT_S = float(os.environ.get("CF_TIMEOUT_S", "45"))


# ---------------------------------------------------------------------------
# Artifact loading — no pickles on the serving path
# ---------------------------------------------------------------------------
def _require(path, hint):
    if not os.path.exists(path):
        raise SystemExit(
            f"\nMissing artifact: {path}\n{hint}\n"
            "Run:  python prepare_artifacts.py --source \"<path to>/Final Results/exp_v2\"\n"
        )
    return path


log.info("loading artifacts from %s", ART)
_require(os.path.join(ART, "model.json"),
         "The model must be exported from the published experiment.")

booster = xgb.Booster()
booster.load_model(os.path.join(ART, "model.json"))

with open(os.path.join(ART, "meta.json")) as fh:
    META = json.load(fh)

with open(os.path.join(ART, "calibration.json")) as fh:
    _cal = json.load(fh)
CAL_X = np.asarray(_cal["x"], dtype=float)
CAL_Y = np.asarray(_cal["y"], dtype=float)

if META["features"] != FEATURES:
    raise SystemExit("Artifact feature order does not match clinical.py FEATURES.")

BG = pd.read_csv(os.path.join(ART, "dice_background.csv"))

OPS = META["operating_points"]
DEFAULT_OP = META.get("default_operating_point", "balanced")
THRESHOLD = float(OPS[DEFAULT_OP]["threshold"])

log.info("model loaded | features=%d | default op=%s thr=%.4f",
         len(FEATURES), DEFAULT_OP, THRESHOLD)


# The published model was fitted with early stopping. Predicting with every
# stored tree instead of the selected range silently changes the output
# (AUC 0.8182 rather than the published 0.8203), so the range is pinned.
ITER_RANGE = tuple(META.get("iteration_range", (0, booster.num_boosted_rounds())))


def raw_proba(X: pd.DataFrame) -> np.ndarray:
    """Uncalibrated positive-class probability from the booster."""
    d = xgb.DMatrix(X[FEATURES].astype(float).values, feature_names=FEATURES)
    return booster.predict(d, iteration_range=ITER_RANGE)


def calibrate(p: np.ndarray) -> np.ndarray:
    """Apply the isotonic map fitted on out-of-fold predictions."""
    return np.interp(np.asarray(p, dtype=float), CAL_X, CAL_Y, left=CAL_Y[0], right=CAL_Y[-1])


def calibrated_proba(X: pd.DataFrame) -> np.ndarray:
    return calibrate(raw_proba(X))


# SHAP on the raw margin, as in the paper. TreeExplainer is exact for XGBoost.
explainer = shap.TreeExplainer(booster)
log.info("SHAP TreeExplainer ready")


# ---------------------------------------------------------------------------
# Consistency-safe model wrapper for DiCE
# ---------------------------------------------------------------------------
class ConsistentModel:
    """
    Presents the classifier to DiCE as if it took only the raw actionable
    variables. Derived features are rebuilt from them on every call, so the
    search cannot produce an internally contradictory patient.
    """

    classes_ = np.array([0, 1])

    def __init__(self, ctx, threshold):
        self.ctx = ctx
        self.threshold = float(threshold)

    def fit(self, X, y):
        return self

    def _full(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=RAW_ACTIONABLE)
        return rebuild_from_raw(X, self.ctx)

    def _rescale(self, p):
        """
        Map the operating threshold onto 0.5, monotonically.

        DiCE decides "has the class flipped?" against a hard-wired 0.5, but our
        operating point is 0.174. Without this the search returns profiles that
        cross 0.5 and every one of them is then rejected by the stricter real
        check, yielding zero recommendations. The transform is order-preserving,
        so it changes which points count as flipped, not their ranking.
        """
        t = self.threshold
        p = np.asarray(p, dtype=float)
        return np.where(p <= t,
                        0.5 * p / max(t, 1e-9),
                        0.5 + 0.5 * (p - t) / max(1.0 - t, 1e-9))

    def true_proba(self, X):
        return calibrated_proba(self._full(X))

    def predict_proba(self, X):
        p = self._rescale(self.true_proba(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.true_proba(X) >= self.threshold).astype(int)


def shap_raw_ranking(shap_row: dict) -> list:
    """
    Rank raw actionable variables by importance, crediting each derived
    feature's |phi| back to the raw parents it was computed from. Without this
    the ranking ignores that e.g. glucose also drives glucose_bmi.
    """
    agg = {r: 0.0 for r in RAW_ACTIONABLE}
    for feat, phi in shap_row.items():
        a = abs(float(phi))
        if feat in agg:
            agg[feat] += a
        elif feat in DERIVED_PARENTS:
            parents = [p for p in DERIVED_PARENTS[feat] if p in agg]
            for p in parents:
                agg[p] += a / len(parents)
    return [f for f, _ in sorted(agg.items(), key=lambda kv: -kv[1])]


def _changes(original: dict, cf_row: pd.Series, ctx) -> dict:
    out = {}
    for f in RAW_ACTIONABLE:
        o, n = float(original[f]), float(cf_row[f])
        if abs(n - o) > 1e-3:
            out[f] = round(n, 2)
    return out


def generate_counterfactuals(clean, eng, shap_row, threshold, total_cfs=3):
    """
    Produce both strategies: unconstrained (Standard DiCE over all five raw
    variables) and SHAP-guided (top-3 by parent-credited |SHAP|).

    Returns (payload, diagnostics).
    """
    ctx = {
        "age": eng["age"],
        "gender": eng["gender"],
        "height_m": clean["height"] / 100.0,
        "hypertensive": eng["hypertensive"],
        "family_hypertension": eng["family_hypertension"],
        "family_risk": eng["family_risk"],
        "cardio_risk": eng["cardio_risk"],
    }
    current = {f: float(eng[f]) for f in RAW_ACTIONABLE}
    bounds = clinical_bounds(ctx, current)

    ranking = shap_raw_ranking(shap_row)
    query = pd.DataFrame([current])[RAW_ACTIONABLE].astype(float)

    cmodel = ConsistentModel(ctx, threshold)
    data = dice_ml.Data(dataframe=BG, continuous_features=list(RAW_ACTIONABLE),
                        outcome_name="diabetic")
    engine = Dice(data, dice_ml.Model(model=cmodel, backend="sklearn"), method="genetic")

    strategies = {
        "standard": list(RAW_ACTIONABLE),
        "shap_guided": ranking[:3],
    }

    out, diag = {}, {}
    for name, vary in strategies.items():
        # A variable pinned to a single value by clinical bounds is not a lever.
        vary = [f for f in vary if bounds[f][0] < bounds[f][1] - 1e-9]
        if not vary:
            out[name], diag[name] = [], {"requested": total_cfs, "returned": 0,
                                         "reason": "no actionable headroom"}
            continue

        pr = {f: bounds[f] for f in vary}
        try:
            with _cf_lock:
                t0 = time.time()
                res = engine.generate_counterfactuals(
                    query, total_CFs=total_cfs, desired_class="opposite",
                    features_to_vary=vary, permitted_range=pr,
                    verbose=False,
                )
                elapsed = time.time() - t0
            cf_df = res.cf_examples_list[0].final_cfs_df
        except Exception as exc:                       # noqa: BLE001
            log.warning("DiCE failed for %s: %s", name, exc)
            out[name], diag[name] = [], {"requested": total_cfs, "returned": 0,
                                         "reason": "search failed"}
            continue

        cands, dropped = [], {"direction": 0, "not_valid": 0, "no_change": 0}
        if cf_df is not None:
            for _, row in cf_df.iterrows():
                ch = _changes(current, row, ctx)
                if not ch:
                    dropped["no_change"] += 1
                    continue

                proposed = dict(current)
                proposed.update(ch)

                if violates_direction(current, proposed):
                    dropped["direction"] += 1
                    continue

                full = rebuild_from_raw(pd.DataFrame([proposed])[RAW_ACTIONABLE], ctx)
                new_p = float(calibrated_proba(full)[0])
                if new_p >= threshold:
                    dropped["not_valid"] += 1
                    continue

                n_ch = len(ch)
                cands.append({
                    "changes": ch,
                    "new_risk_score": round(new_p, 4),
                    "new_risk_percent": round(new_p * 100, 1),
                    "n_changes": n_ch,
                    "effort": "Easy" if n_ch <= 1 else "Moderate" if n_ch <= 2 else "Hard",
                    "alignment": round(alignment_score(ch, shap_row, ranking), 4),
                })

        # Prefer fewer changes, then larger risk reduction.
        cands.sort(key=lambda c: (c["n_changes"], c["new_risk_score"]))

        # DiCE's genetic search frequently converges on the same profile more
        # than once. Offering a patient the identical instruction three times
        # is noise, so keep the best occurrence of each distinct plan.
        seen, unique = set(), []
        for c in cands:
            sig = tuple(sorted(c["changes"].items()))
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(c)
        cands = unique

        for i, c in enumerate(cands, 1):
            c["cf_index"] = i

        out[name] = cands
        diag[name] = {
            "requested": total_cfs,
            "returned": len(cands),
            "varied": vary,
            "dropped": dropped,
            "seconds": round(elapsed, 2),
        }

    return out, diag


def alignment_score(changes: dict, shap_row: dict, ranking: list) -> float:
    """
    SHAP-CF Alignment Score (paper Eq. 1): the SHAP mass actually captured,
    over the most any counterfactual changing this many features could have
    captured. Sparsity-neutral -- it cannot be gamed by changing more or fewer
    variables, and equals 1.0 exactly when the search picked the k most
    important levers available to it.
    """
    agg = {r: 0.0 for r in RAW_ACTIONABLE}
    for feat, phi in shap_row.items():
        a = abs(float(phi))
        if feat in agg:
            agg[feat] += a
        elif feat in DERIVED_PARENTS:
            parents = [p for p in DERIVED_PARENTS[feat] if p in agg]
            for p in parents:
                agg[p] += a / len(parents)

    k = len(changes)
    if k == 0:
        return 0.0
    got = sum(agg.get(f, 0.0) for f in changes)
    best = sum(sorted(agg.values(), reverse=True)[:k])
    return float(got / best) if best > 0 else 0.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.errorhandler(ValidationError)
def _handle_validation(exc):
    return jsonify({"error": "validation_failed", "details": exc.errors}), 400


@app.errorhandler(413)
def _handle_too_large(_):
    return jsonify({"error": "request body too large"}), 413


@app.errorhandler(Exception)
def _handle_unexpected(exc):
    # Log the detail, return a generic message: internals are not the caller's.
    log.exception("unhandled error: %s", exc)
    return jsonify({"error": "internal error"}), 500


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": {
            "features": FEATURES,
            "raw_actionable": RAW_ACTIONABLE,
            "frozen": ["hypertensive", "age", "gender",
                       "family_hypertension", "family_risk", "cardio_risk"],
            "calibrated": True,
            "generated_utc": META["generated_utc"],
        },
        "operating_points": OPS,
        "default_operating_point": DEFAULT_OP,
        "performance": META["held_out_metrics_calibrated"],
    })


@app.post("/predict")
def predict():
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "body must be a JSON object"}), 400

    clean = validate_inputs(payload)          # raises ValidationError -> 400
    eng = engineer(clean)
    row = pd.DataFrame([eng])[FEATURES].astype(float)

    op_name = payload.get("operating_point", DEFAULT_OP)
    if op_name not in OPS:
        return jsonify({"error": f"unknown operating_point '{op_name}'",
                        "valid": list(OPS)}), 400
    threshold = float(OPS[op_name]["threshold"])

    p_raw = float(raw_proba(row)[0])
    p_cal = float(calibrate(np.array([p_raw]))[0])
    is_high = p_cal >= threshold

    sv = explainer.shap_values(row)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv).reshape(-1)
    shap_row = {f: round(float(sv[i]), 5) for i, f in enumerate(FEATURES)}

    cfs, diag = ({}, {})
    if is_high:
        cfs, diag = generate_counterfactuals(clean, eng, shap_row, threshold)

    op = OPS[op_name]
    return jsonify({
        "risk_score":   round(p_cal, 4),
        "risk_percent": round(p_cal * 100, 1),
        "risk_label":   "High" if is_high else "Low",
        "risk_raw_uncalibrated": round(p_raw, 4),

        "operating_point": {
            "name": op_name,
            "threshold": threshold,
            # What a positive flag actually means, from held-out data. The tool
            # prescribes on the back of this number, so it is shown, not hidden.
            "expected_precision": round(op["test"]["precision"], 3),
            "expected_recall":    round(op["test"]["recall"], 3),
            "flag_rate":          round(op["test"]["flag_rate"], 3),
        },

        "shap_values":    shap_row,
        "shap_ranking":   shap_raw_ranking(shap_row),
        "model_features": {k: round(float(v), 4) for k, v in eng.items()},

        "counterfactuals":             cfs.get("shap_guided", []),
        "counterfactuals_unconstrained": cfs.get("standard", []),
        "cf_diagnostics":              diag,

        "disclaimer": (
            "Decision support only. This estimates a model's prediction, not a "
            "diagnosis, and recommendations are model-level targets that have "
            "not been validated against clinical outcomes."
        ),
    })


if __name__ == "__main__":
    port = int(os.environ.get("ML_PORT", "5001"))
    # Bind loopback by default: this service has no authentication of its own
    # and must not be exposed directly to a network.
    host = os.environ.get("ML_HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False, threaded=True)
