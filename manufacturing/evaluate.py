"""
The evaluation protocol for the ACT-Africa bearing challenge.

There is exactly one protocol here, and it is the same one used for the result in
the lecture:

    Leave-one-bearing-out over the 12 real + healthy bearings.
    Train on 11, predict the 1 held out, repeat for all 12.
    Pool the out-of-fold predictions and score them once.
    Metric: balanced accuracy, per window.

Everything you report must come from `lobo_score`. Nothing else is comparable to
the lecture, to the baselines, or to the other teams.

    from evaluate import lobo_score, verify_protocol, report, BASELINES

THE PROTOCOL GATE
-----------------
A number that did not come out of `lobo_score` is not a valid hackathon result, and
Slide 1 must show `verify_protocol` -> PASSED. A submission without it is invalid,
not merely marked down.

`train_test_split` is INVALID here. One bearing produces ~600 near-duplicate windows,
so a random split puts the same physical bearing in training and testing; the model
recognises the bearing, not the fault. Measured: 0.863 that way, 0.554 done properly.

LEAKAGE, AND WHAT THIS FILE CANNOT CHECK
----------------------------------------
Any parameter learned from data - a mean, a standard deviation, a PCA basis, a
whitening matrix, a feature selector, a tuned threshold - must be fitted on the
eleven TRAINING bearings inside each fold. Put the step inside a sklearn Pipeline
and `lobo_score` handles it, because it calls clone(model).fit(X_train, y_train)
once per fold.

    # CORRECT                                    # WRONG - PCA has seen the held-out bearing
    Pipeline([("s", StandardScaler()),           Xw = PCA(64).fit_transform(X[pool])
              ("p", PCA(64)),                    lobo_score(probe(), Xw, y[pool], b[pool])
              ("c", LogisticRegression())])

The guards below reject the wrong BEARINGS. They cannot reject the wrong PIPELINE:
by the time `lobo_score` receives an already-transformed array, the leakage happened
upstream. On this dataset the leaked version scores 0.543 against 0.564 for the
correct one, so leakage here makes the number slightly WORSE - you cannot detect it
from the score. Only reading the code finds it, which is why the protocol is judged
on your slide rather than on your number.

Why leave-one-bearing-out, and not a train/test split? Because the question is
"will this work on the next bearing?", and with only four bearings per class a
single held-out set is a lottery - the score swings between 0.07 and 0.97
depending on which bearings you happen to hold out. Rotating through all twelve
removes that luck.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (balanced_accuracy_score, f1_score, confusion_matrix)

CLASS_NAMES = ["Healthy", "Outer-race fault", "Inner-race fault"]

# The 12 bearings the protocol runs over: 4 healthy, plus 8 with real damage from
# accelerated lifetime tests (4 outer-race, 4 inner-race). Per the Paderborn
# documentation, seven of the eight are fatigue pitting; KA15 is plastic
# deformation with indentations. The 4 bearings whose damage was machined in
# (KA01, KA03, KI01, KI03) are NOT evaluation folds - see `extra_train` below.
POOL = {0: ["K001", "K002", "K003", "K004"],
        1: ["KA04", "KA15", "KA16", "KA22"],
        2: ["KI04", "KI14", "KI16", "KI17"]}
POOL_BEARINGS = sorted(sum(POOL.values(), []))

# The 4 bearings whose damage was machined in. They are NEVER an evaluation fold.
# They may enter training only through the `extra_train` argument of lobo_score.
EXTRA_BEARINGS = ["KA01", "KA03", "KI01", "KI03"]


def _check_pool(bearing, where):
    """Reject anything that is not exactly the 12 evaluation bearings.

    Passing the whole dataset instead of `pool_mask(...)` used to run silently and
    return a number roughly 0.10 too low, because the machined bearings were never
    predicted yet still counted in the metric. It now stops immediately.
    """
    bearing = np.asarray(bearing)
    present = set(map(str, np.unique(bearing)))
    expected = set(POOL_BEARINGS)

    unexpected = sorted(present - expected)
    missing = sorted(expected - present)

    if unexpected:
        machined = [b for b in unexpected if b in EXTRA_BEARINGS]
        hint = ""
        if machined:
            hint = ("\n  {} are the machined-damage bearings. They are never an "
                    "evaluation fold."
                    "\n  Select the pool first, and pass them as extra training data if you "
                    "want them:"
                    "\n      pool = pool_mask(bearing)"
                    "\n      lobo_score(model, X[pool], y[pool], bearing[pool],"
                    "\n                 extra_train=(X[~pool], y[~pool], bearing[~pool]))"
                    .format(machined))
        raise ValueError(
            f"{where}: unexpected bearings in the evaluation set: {unexpected}"
            f"\n  The protocol runs over exactly these 12: {POOL_BEARINGS}"
            f"\n  You probably forgot `pool = pool_mask(bearing)`.{hint}")

    if missing:
        raise ValueError(
            f"{where}: the evaluation pool is incomplete - missing {missing}."
            f"\n  All 12 bearings must be present, or the score is not comparable to"
            f"\n  the reference results or to other teams: {POOL_BEARINGS}")

# Measured with a linear probe: standardise, then logistic regression with
# class_weight="balanced". Same protocol, same metric, same unit.
BASELINES = {
    "raw envelope window":              0.322,
    "chance":                           0.333,
    "MOMENT frozen, 512-d":             0.554,
    "MOMENT + machined bearings in training": 0.589,
    "classical features, 17-d":         0.634,
}


def verify_protocol(bearing, recording, verbose=True):
    """Check the split is honest before you trust any number that comes out of it.

    Asserts that every one of the 12 folds holds out a whole bearing, and that no
    recording is ever cut across a fold boundary. Run it, show the output, and put
    it in your presentation - "what did you hold out" is the first thing you are
    judged on.
    """
    bearing, recording = np.asarray(bearing), np.asarray(recording)
    _check_pool(bearing, "verify_protocol")          # raises on anything unexpected
    problems = []

    for b in POOL_BEARINGS:
        held = bearing == b
        if not held.any():
            continue
        # a recording must belong to exactly one bearing, or folds overlap
        shared = set(recording[held]) & set(recording[~held])
        if shared:
            problems.append(f"{b}: {len(shared)} recording(s) also appear outside "
                            f"this bearing - folds would overlap")

    if verbose:
        if problems:
            print("PROTOCOL CHECK: FAILED")
            for p in problems:
                print("  -", p)
        else:
            print("PROTOCOL CHECK: PASSED")
            print(f"  12 folds, one bearing held out each time")
            print(f"  no recording spans a fold boundary")
            print(f"  {len(bearing)} windows over {len(set(recording))} recordings")
    return not problems


def lobo_score(model, X, y, bearing, extra_train=None, verbose=True):
    """Leave-one-bearing-out over the 12 pool bearings. This is the official score.

    model        an unfitted sklearn estimator or Pipeline; cloned for each fold
    X, y         features and labels for the pool windows
    bearing      bearing code per window, used as the fold key
    extra_train  optional (X_extra, y_extra, bearing_extra) added to EVERY training
                 fold and never tested on. Bearing codes are REQUIRED and are checked:
                 only the 4 machined bearings may be added. This is the only permitted
                 route for them; they become extra training material, not a separate
                 experiment.

    X, y and bearing must cover exactly the 12 evaluation bearings. Pass anything
    else and this raises ValueError rather than returning a number that is not
    comparable to the reference results.

    Returns a dict with the pooled balanced accuracy (the headline), macro F1, the
    per-bearing recalls, and the out-of-fold predictions.
    """
    X, y, bearing = np.asarray(X), np.asarray(y), np.asarray(bearing)
    _check_pool(bearing, "lobo_score")               # raises on anything unexpected

    if extra_train is not None:
        if len(extra_train) != 3:
            raise ValueError(
                "lobo_score: extra_train must be (X_extra, y_extra, bearing_extra)."
                "\n  Bearing codes are required so that the extra training data can be"
                "\n  checked - without them there is no way to tell whether it came from"
                "\n  the evaluation bearings, which would leak the held-out fold into"
                "\n  training. Use:"
                "\n      extra_train=(X[~pool], y[~pool], bearing[~pool])")
        Xe, ye, be = extra_train
        Xe, ye, be = np.asarray(Xe), np.asarray(ye), np.asarray(be)

        if not (len(Xe) == len(ye) == len(be)):
            raise ValueError(f"lobo_score: extra_train lengths disagree - "
                             f"{len(Xe)} rows, {len(ye)} labels, {len(be)} bearing codes")

        seen = set(map(str, np.unique(be)))
        leaked = sorted(seen & set(POOL_BEARINGS))
        if leaked:
            raise ValueError(
                f"lobo_score: extra_train contains evaluation bearings {leaked}."
                f"\n  Those windows would appear in training while the same bearing is"
                f"\n  being tested, which inflates the score. Pass only the machined"
                f"\n  bearings: extra_train=(X[~pool], y[~pool], bearing[~pool])")
        foreign = sorted(seen - set(EXTRA_BEARINGS))
        if foreign:
            raise ValueError(
                f"lobo_score: extra_train contains bearings that are not permitted as"
                f" extra training data: {foreign}."
                f"\n  Only the machined-damage bearings may be added: {EXTRA_BEARINGS}")

    oof = np.full(len(y), -1, dtype=int)
    per = {}

    for b in POOL_BEARINGS:
        held = bearing == b
        Xtr, ytr = X[~held], y[~held]
        if extra_train is not None:
            Xtr = np.vstack([Xtr, Xe])
            ytr = np.concatenate([ytr, ye])
        pred = clone(model).fit(Xtr, ytr).predict(X[held])
        oof[held] = pred
        per[b] = float((pred == y[held]).mean())

    assert (oof >= 0).all(), "internal error: some windows were never predicted"

    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y, oof)),
        "macro_f1": float(f1_score(y, oof, average="macro")),
        "per_bearing": per,
        "oof": oof,
    }
    if verbose:
        v = np.array(list(per.values()))
        print(f"balanced accuracy  {out['balanced_accuracy']:.3f}      "
              f"(baseline 0.554 · chance 0.333)")
        print(f"macro F1           {out['macro_f1']:.3f}")
        print(f"per-bearing recall  {(v > 0.60).sum()} above 0.60 · "
              f"{(v < 0.15).sum()} below 0.15 · {((v >= 0.15) & (v <= 0.60)).sum()} in between")
    return out


def report(y_true, oof, per_bearing=None, title="result"):
    """Confusion matrix and per-bearing table - the two things you must present."""
    cm = confusion_matrix(y_true, oof, labels=[0, 1, 2])
    print(f"\n{title}")
    print(f"balanced accuracy {balanced_accuracy_score(y_true, oof):.4f}\n")
    print(f"{'':20s}" + "".join(f"{n[:12]:>14s}" for n in CLASS_NAMES) + "     recall")
    for i, name in enumerate(CLASS_NAMES):
        rec = cm[i].sum() and cm[i, i] / cm[i].sum()
        print(f"{name:20s}" + "".join(f"{v:>14d}" for v in cm[i]) + f"{rec:>11.3f}")
    print("rows = truth, columns = prediction")

    if per_bearing:
        print("\nper-bearing recall, worst last - this is the table that matters")
        for b, v in sorted(per_bearing.items(), key=lambda kv: -kv[1]):
            bar = "#" * int(round(v * 30))
            print(f"  {b:6s} {v:.3f}  {bar}")
        vals = np.array(list(per_bearing.values()))
        print(f"\n  mean {vals.mean():.3f}   sd {vals.std():.3f}   "
              f"worst {vals.min():.3f} on {min(per_bearing, key=per_bearing.get)}")
    return cm


def pool_mask(bearing):
    """Boolean mask selecting the 12 protocol bearings from the full dataset."""
    return np.isin(np.asarray(bearing), POOL_BEARINGS)
