"""
prepare_artifacts.py — build the serving artifacts for DiaFact from the
published experiment, with provenance checks.

Run once before starting the service:

    python prepare_artifacts.py --source "<path to>/Final Results/exp_v2"

What it does, and why:

1.  Loads the *published* XGBoost model (the one Table 1 of the paper reports)
    and refuses to continue unless it reproduces the published test metrics.
    The previously shipped model was trained on a different dataset entirely
    (12.19% prevalence vs DiaHealth's 6.33%, mean weight 75kg vs 53.6kg) and
    scored AUC 0.674 / precision 0.111 on the real DiaHealth test set.

2.  Fits an isotonic calibrator on *out-of-fold* predictions. The model was
    trained on BorderlineSMOTE-resampled data balanced 50/50, so its raw
    predict_proba is badly inflated relative to the true 6.34% prevalence.
    Showing a patient that inflated number as "your risk" is a safety problem,
    not a cosmetic one. Calibration is monotonic, so it changes no ranking and
    no AUC -- only the number on screen.

3.  Chooses the operating threshold on those same out-of-fold predictions.
    The test set is never used for any fitting decision; it is scored once, at
    the end, as a held-out estimate.

4.  Writes pickle-free artifacts. Unpickling is arbitrary code execution, which
    is not an acceptable load path for a clinical tool. The model goes out in
    XGBoost's native JSON format and the calibrator as interpolation knots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "artifacts")

# Published DiaHealth test-set metrics for the XGBoost model (paper Table 1).
PUBLISHED = {"auc_roc": 0.8203, "f1": 0.3766, "precision": 0.3412, "recall": 0.4203}
TOL = 5e-3

sys.path.insert(0, HERE)
from clinical import FEATURES, RAW_ACTIONABLE  # noqa: E402


def log(msg=""):
    print(msg, flush=True)


def load_source(source):
    import joblib

    ck = os.path.join(source, "checkpoints")
    splits = joblib.load(os.path.join(ck, "nb01_splits.pkl"))
    bundle = joblib.load(os.path.join(ck, "nb02_xgb_diahealth.pkl"))
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    return splits, model


def best_iteration_range(model):
    """
    The published model was fitted with early stopping (best_iteration=609 of
    1076 requested). `Booster.predict` uses every stored tree unless told
    otherwise, which silently changes the predictions -- AUC 0.8182 instead of
    the published 0.8203. Serving must use the same tree range the sklearn
    wrapper does.
    """
    bi = getattr(model, "best_iteration", None)
    n = model.get_booster().num_boosted_rounds()
    return (0, int(bi) + 1) if bi is not None else (0, int(n))


def verify(model, X_test, y_test, iter_range):
    from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                                 recall_score, average_precision_score,
                                 brier_score_loss)
    import xgboost as xgb

    Xd = X_test[FEATURES].astype(float)
    p = model.predict_proba(Xd)[:, 1]

    # The served path must agree with the sklearn path to floating-point noise.
    d = xgb.DMatrix(Xd.values, feature_names=FEATURES)
    p_serve = model.get_booster().predict(d, iteration_range=iter_range)
    drift = float(np.abs(p_serve - p).max())
    log(f"  serving path vs sklearn path: max |diff| = {drift:.2e} "
        f"(iteration_range={iter_range})")
    if drift > 1e-6:
        raise SystemExit("ABORT: serving prediction path diverges from the published one.")

    yp = (p >= 0.5).astype(int)
    got = {
        "auc_roc":   roc_auc_score(y_test, p),
        "auc_pr":    average_precision_score(y_test, p),
        "f1":        f1_score(y_test, yp),
        "precision": precision_score(y_test, yp),
        "recall":    recall_score(y_test, yp),
        "brier":     brier_score_loss(y_test, p),
    }
    log("  provenance check (raw model @0.5):")
    ok = True
    for k, want in PUBLISHED.items():
        d = abs(got[k] - want)
        flag = "OK " if d < TOL else "FAIL"
        if d >= TOL:
            ok = False
        log(f"    {flag} {k:10s} got {got[k]:.4f}  published {want:.4f}  diff {d:.5f}")
    if not ok:
        raise SystemExit(
            "\nABORT: this model does not reproduce the published metrics.\n"
            "You are pointing at the wrong artifacts. The correct model is\n"
            "  <exp_v2>/checkpoints/nb02_xgb_diahealth.pkl\n"
        )
    return got, p


def oof_calibration(model, X_train, y_train, n_estimators, seed=42, folds=5):
    """
    Out-of-fold probabilities using the published hyperparameters, with
    BorderlineSMOTE applied *inside* each fold (never across the split).
    """
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import BorderlineSMOTE
    from xgboost import XGBClassifier

    params = model.get_params()
    for drop in ("callbacks", "early_stopping_rounds", "eval_metric"):
        params.pop(drop, None)
    params["n_jobs"] = -1
    # Match the deployed model's effective capacity. Early stopping cannot be
    # replicated without a validation split inside each fold, so we fix the
    # tree count at the value early stopping chose.
    params["n_estimators"] = int(n_estimators)

    X = X_train[FEATURES].astype(float).reset_index(drop=True)
    y = pd.Series(y_train).reset_index(drop=True).astype(int)

    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        sm = BorderlineSMOTE(random_state=seed)
        Xr, yr = sm.fit_resample(Xtr, ytr)
        m = XGBClassifier(**params)
        m.fit(Xr, yr, verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        log(f"    fold {i}/{folds} done")
    return oof, y.values


def _at(y, p, t):
    from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
    yp = (p >= t).astype(int)
    return {
        "threshold": float(t),
        "precision": float(precision_score(y, yp, zero_division=0)),
        "recall":    float(recall_score(y, yp, zero_division=0)),
        "f1":        float(f1_score(y, yp, zero_division=0)),
        "f2":        float(fbeta_score(y, yp, beta=2, zero_division=0)),
        "flag_rate": float(yp.mean()),
    }


def pick_operating_points(y, p):
    """
    Three named operating points, all chosen on out-of-fold data.

    A single hard-coded 0.5 cutoff is the wrong default for a screening tool at
    6.3% prevalence: it optimises nothing in particular and, on this model,
    trades away recall that matters more than the precision it buys. Missing a
    preventable diagnosis costs more than an unnecessary follow-up test, so the
    default here maximises F2 (recall weighted twice precision). The clinician
    can select a stricter point when the follow-up is expensive or invasive.
    """
    grid = np.unique(np.round(np.linspace(0.02, 0.95, 466), 4))
    rows = [_at(y, p, t) for t in grid]

    screening = max(rows, key=lambda r: r["f2"])
    balanced = max(rows, key=lambda r: r["f1"])
    # Highest recall still reaching 0.60 precision. Isotonic calibration puts
    # long plateaus in the threshold sweep, so maximising precision directly
    # tends to land back on the balanced point; anchoring on a precision target
    # and taking the most sensitive point that clears it keeps them distinct.
    eligible = [r for r in rows if r["precision"] >= 0.60 and r["recall"] > 0]
    strict = max(eligible, key=lambda r: r["recall"]) if eligible else \
        max(rows, key=lambda r: (r["precision"], r["recall"]))

    return {"screening": screening, "balanced": balanced, "confirmatory": strict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default=r"D:/thesis paper/TEHI 2026/exp_v2/exp_v2/Final Results/exp_v2",
        help="path to the published exp_v2 results folder",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 70)
    log("DiaFact artifact preparation")
    log("=" * 70)
    log(f"source: {args.source}")

    log("\n[1/5] loading published artifacts")
    splits, model = load_source(args.source)
    X_train, y_train = splits["X_train"], splits["y_train"].astype(int)
    X_test, y_test = splits["X_test"], splits["y_test"].astype(int)
    log(f"  train {X_train.shape}  test {X_test.shape}  "
        f"test prevalence {y_test.mean() * 100:.2f}%")

    log("\n[2/5] verifying model provenance against the paper")
    iter_range = best_iteration_range(model)
    raw_metrics, p_test_raw = verify(model, X_test, y_test, iter_range)

    log("\n[3/5] fitting isotonic calibration on out-of-fold predictions")
    oof_raw, y_oof = oof_calibration(model, X_train, y_train,
                                     n_estimators=iter_range[1], seed=args.seed)

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof_raw, y_oof)
    oof_cal = iso.predict(oof_raw)

    from sklearn.metrics import brier_score_loss, roc_auc_score
    log(f"  OOF Brier  raw {brier_score_loss(y_oof, oof_raw):.5f}"
        f"  ->  calibrated {brier_score_loss(y_oof, oof_cal):.5f}")
    log(f"  OOF AUC    raw {roc_auc_score(y_oof, oof_raw):.5f}"
        f"  ->  calibrated {roc_auc_score(y_oof, oof_cal):.5f}  (must be ~equal)")
    log(f"  mean predicted risk  raw {oof_raw.mean() * 100:.1f}%"
        f"  ->  calibrated {oof_cal.mean() * 100:.2f}%"
        f"   (true prevalence {y_oof.mean() * 100:.2f}%)")

    log("\n[4/5] selecting operating points on out-of-fold data")
    ops = pick_operating_points(y_oof, oof_cal)
    for name, r in ops.items():
        log(f"  {name:13s} thr {r['threshold']:.4f}  P {r['precision']:.3f}"
            f"  R {r['recall']:.3f}  F1 {r['f1']:.3f}  F2 {r['f2']:.3f}"
            f"  flags {r['flag_rate'] * 100:.1f}%")

    # Held-out estimate, scored once, at each operating point.
    from sklearn.metrics import average_precision_score
    p_test_cal = iso.predict(p_test_raw)
    log("  held-out test performance at those thresholds:")
    for name, r in ops.items():
        t = _at(y_test, p_test_cal, r["threshold"])
        r["test"] = t
        log(f"  {name:13s} thr {t['threshold']:.4f}  P {t['precision']:.3f}"
            f"  R {t['recall']:.3f}  F1 {t['f1']:.3f}  flags {t['flag_rate'] * 100:.1f}%")

    # Default to `balanced`. This tool does not merely flag risk, it issues
    # lifestyle prescriptions, so a false positive costs a patient real effort
    # and anxiety rather than one cheap confirmatory test. `screening` catches
    # more (test recall 0.81) but flags 30% of everyone at 0.17 precision.
    default_op = "balanced"
    thr = ops[default_op]["threshold"]
    held_out = {
        "auc_roc": float(roc_auc_score(y_test, p_test_cal)),
        "auc_pr":  float(average_precision_score(y_test, p_test_cal)),
        "brier":   float(brier_score_loss(y_test, p_test_cal)),
    }
    log(f"  test AUC-ROC {held_out['auc_roc']:.4f}  AUC-PR {held_out['auc_pr']:.4f}")
    log(f"  test Brier  raw {raw_metrics['brier']:.5f} -> calibrated {held_out['brier']:.5f}")

    log("\n[5/5] writing pickle-free artifacts")

    # Slice the booster to exactly the trees early stopping selected. Saving
    # the full 660 and relying on an iteration_range at serve time would leave
    # SHAP explaining trees the prediction never uses, so the attributions
    # would not sum to the score being explained.
    model_path = os.path.join(OUT_DIR, "model.json")
    sliced = model.get_booster()[iter_range[0]:iter_range[1]]
    sliced.save_model(model_path)
    log(f"  booster sliced to {sliced.num_boosted_rounds()} trees "
        f"(from {model.get_booster().num_boosted_rounds()})")

    # The sliced artifact must still reproduce the published metrics.
    import xgboost as _xgb
    _b = _xgb.Booster(); _b.load_model(model_path)
    _p = _b.predict(_xgb.DMatrix(X_test[FEATURES].astype(float).values,
                                 feature_names=FEATURES))
    _d = float(np.abs(_p - p_test_raw).max())
    log(f"  saved-artifact vs published predictions: max |diff| = {_d:.2e}")
    if _d > 1e-6:
        raise SystemExit("ABORT: saved model does not match the published one.")

    cal = {
        "x": [float(v) for v in iso.X_thresholds_],
        "y": [float(v) for v in iso.y_thresholds_],
    }
    with open(os.path.join(OUT_DIR, "calibration.json"), "w") as fh:
        json.dump(cal, fh)

    bg = X_train[RAW_ACTIONABLE].astype(float).copy()
    bg["diabetic"] = np.asarray(y_train)
    bg.to_csv(os.path.join(OUT_DIR, "dice_background.csv"), index=False)

    ref = X_train[FEATURES].astype(float)
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": os.path.abspath(args.source),
        "seed": args.seed,
        "features": FEATURES,
        "iteration_range": [0, int(iter_range[1] - iter_range[0])],
        "trees_saved": int(iter_range[1] - iter_range[0]),
        "raw_actionable": RAW_ACTIONABLE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_prevalence": float(np.asarray(y_train).mean()),
        "test_prevalence": float(np.asarray(y_test).mean()),
        "operating_threshold": thr,
        "default_operating_point": default_op,
        "operating_points": ops,
        "published_metrics_raw_at_0.5": {k: float(v) for k, v in raw_metrics.items()},
        "held_out_metrics_calibrated": held_out,
        "feature_reference": {
            c: {
                "min": float(ref[c].min()), "max": float(ref[c].max()),
                "mean": float(ref[c].mean()), "p05": float(ref[c].quantile(0.05)),
                "p95": float(ref[c].quantile(0.95)),
            } for c in FEATURES
        },
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    log(f"  wrote {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        log(f"    {f:24s} {size / 1024:8.1f} KB")
    log("\nDone.")


if __name__ == "__main__":
    main()
