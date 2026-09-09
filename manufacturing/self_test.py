"""
Self-test for the evaluation protocol.  Run it any time; it needs no internet.

    python self_test.py

It checks two things a beginner could otherwise get wrong silently, plus the five
reference values. If every line says OK, the protocol in this pack is behaving as
documented.
"""
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from evaluate import (lobo_score, verify_protocol, pool_mask,
                      POOL_BEARINGS, EXTRA_BEARINGS, BASELINES)

failures = []


def check(label, fn):
    try:
        print(f"  OK    {label:52s} {fn()}")
    except AssertionError as exc:
        failures.append(label)
        print(f"  FAIL  {label:52s} {exc}")
    except Exception as exc:
        failures.append(label)
        print(f"  ERROR {label:52s} {type(exc).__name__}: {exc}")


def probe():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=42))])


d = np.load("data/challenge.npz", allow_pickle=True)
X, y, bearing, recording = d["embeddings"], d["y"], d["bearing"], d["recording"]
pool = pool_mask(bearing)

print("\nACT-Africa bearing challenge - protocol self-test\n")

# ---------------------------------------------------------------- the pool itself
def _composition():
    assert sorted(set(map(str, bearing[pool]))) == POOL_BEARINGS, "pool contents wrong"
    assert sorted(set(map(str, bearing[~pool]))) == sorted(EXTRA_BEARINGS), "extras wrong"
    assert int(pool.sum()) == 7209, f"expected 7209 pool windows, got {int(pool.sum())}"
    return f"{len(POOL_BEARINGS)} evaluation + {len(EXTRA_BEARINGS)} machined"


def _protocol_passes():
    assert verify_protocol(bearing[pool], recording[pool], verbose=False)
    return "12 folds, no recording crosses a fold"


check("pool composition is the documented 12 + 4", _composition)
check("verify_protocol passes on the correct pool", _protocol_passes)


# ------------------------------------------- the mistake a beginner actually makes
def _raises(fn, must_mention):
    try:
        fn()
    except ValueError as exc:
        assert must_mention in str(exc), f"message does not mention {must_mention!r}"
        return "ValueError raised"
    raise AssertionError("no ValueError - the machined bearings were accepted silently")


check("verify_protocol REJECTS all 16 bearings",
      lambda: _raises(lambda: verify_protocol(bearing, recording, verbose=False), "pool_mask"))
check("lobo_score REJECTS all 16 bearings",
      lambda: _raises(lambda: lobo_score(probe(), X, y, bearing, verbose=False), "pool_mask"))

_missing = pool & (bearing != "KI17")
check("lobo_score REJECTS an incomplete pool",
      lambda: _raises(lambda: lobo_score(probe(), X[_missing], y[_missing],
                                         bearing[_missing], verbose=False), "incomplete"))
# --- extra_train: the only route by which the machined bearings may enter training ---
def _extra_accepted():
    r = lobo_score(probe(), X[pool], y[pool], bearing[pool],
                   extra_train=(X[~pool], y[~pool], bearing[~pool]), verbose=False)
    assert abs(r["balanced_accuracy"] - 0.589) < 0.001, r["balanced_accuracy"]
    return f"machined bearings accepted, {r['balanced_accuracy']:.4f}"


check("extra_train ACCEPTS the machined bearings", _extra_accepted)
check("extra_train REJECTS evaluation bearings",
      lambda: _raises(lambda: lobo_score(probe(), X[pool], y[pool], bearing[pool],
                                         extra_train=(X[pool], y[pool], bearing[pool]),
                                         verbose=False), "evaluation bearings"))
check("extra_train REJECTS a tuple without bearing IDs",
      lambda: _raises(lambda: lobo_score(probe(), X[pool], y[pool], bearing[pool],
                                         extra_train=(X[~pool], y[~pool]),
                                         verbose=False), "bearing_extra"))


# ---------------------------------------------------------------- reference values
_cache = {}


def _ref(name, key, expected, extra=False):
    def f():
        if key not in _cache:
            _cache[key] = lobo_score(
                probe(), d[key][pool], y[pool], bearing[pool],
                extra_train=(d[key][~pool], y[~pool], bearing[~pool]) if extra else None,
                verbose=False)["balanced_accuracy"]
        got = _cache[key]
        assert abs(got - expected) < 0.001, f"expected {expected}, got {got:.4f}"
        return f"{got:.4f}"
    return f


print()
check("reference: raw envelope        = 0.322", _ref("envelope", "envelope", 0.322))
check("reference: classical 17-d      = 0.634", _ref("classical", "classical", 0.634))
check("reference: MOMENT frozen       = 0.554", _ref("embeddings", "embeddings", 0.554))


def _extra():
    r = lobo_score(probe(), X[pool], y[pool], bearing[pool],
                   extra_train=(X[~pool], y[~pool], bearing[~pool]),
                   verbose=False)["balanced_accuracy"]
    assert abs(r - 0.589) < 0.001, f"expected 0.589, got {r:.4f}"
    return f"{r:.4f}"


check("reference: MOMENT + machined   = 0.589", _extra)


def _dict():
    for k, v in [("raw envelope window", 0.322), ("chance", 0.333),
                 ("MOMENT frozen, 512-d", 0.554),
                 ("MOMENT + machined bearings in training", 0.589),
                 ("classical features, 17-d", 0.634)]:
        assert BASELINES[k] == v, f"{k}: shipped {BASELINES[k]}, measured {v}"
    return f"{len(BASELINES)} entries agree with the measurements"


check("shipped BASELINES match what the code produces", _dict)

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}\n")
    sys.exit(1)
print("All protocol checks passed.\n")
