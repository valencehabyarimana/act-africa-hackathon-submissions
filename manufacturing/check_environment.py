"""
Run this first. It needs no internet once the required packages are installed.

Windows:
    py check_environment.py

macOS / Linux:
    python3 check_environment.py

If every line says OK, you are ready. If not, the message tells you what to do.
Do not start modelling until this passes.
"""
import pathlib
import subprocess
import sys
import time

ok = True
HERE = pathlib.Path(__file__).parent


def check(label, fn):
    global ok
    try:
        detail = fn()
        print(f"  OK    {label:34s} {detail}")
    except Exception as exc:
        ok = False
        print(f"  FAIL  {label:34s} {exc}")


def _python():
    v = sys.version_info
    if v < (3, 9):
        raise RuntimeError("Python 3.9+ required; Python 3.11 is recommended")
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if v > (3, 13):
        detail += " (newer than workshop-tested range; Python 3.11 recommended if problems occur)"
    elif (v.major, v.minor) == (3, 11):
        detail += " (recommended)"
    return detail


def _imp(name):
    def f():
        m = __import__(name)
        return getattr(m, "__version__", "?")
    return f


def _data():
    import numpy as np
    d = np.load(HERE / "data/challenge.npz", allow_pickle=True)
    assert d["embeddings"].shape == (9624, 512), d["embeddings"].shape
    return f"{d['embeddings'].shape[0]} windows, 16 bearings"


def _protocol():
    import numpy as np
    from evaluate import verify_protocol, pool_mask
    d = np.load(HERE / "data/challenge.npz", allow_pickle=True)
    m = pool_mask(d["bearing"])
    assert verify_protocol(d["bearing"][m], d["recording"][m], verbose=False)
    return f"{int(m.sum())} windows in the 12-bearing pool"


def _fit():
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    d = np.load(HERE / "data/challenge.npz", allow_pickle=True)
    t0 = time.time()
    Pipeline([
        ("s", StandardScaler()),
        ("c", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]).fit(d["embeddings"][:4000], d["y"][:4000])
    dt = time.time() - t0
    return f"{dt:.1f} s test fit; full notebook runs multiple LOBO folds"


def _jupyter():
    """Test Jupyter through THIS Python interpreter, not the shell PATH.

    On Windows, Python can be installed and `py -m notebook` can work even when
    jupyter.exe is not on PATH. Checking `jupyter ...` directly therefore creates
    a false FAIL. Using sys.executable also guarantees that Jupyter belongs to the
    same Python environment whose NumPy/scikit-learn packages were checked above.
    """
    commands = [
        [sys.executable, "-m", "notebook", "--version"],
        [sys.executable, "-m", "jupyterlab", "--version"],
    ]
    errors = []
    for cmd in commands:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            errors.append("timeout")
            continue
        if r.returncode == 0:
            ver = (r.stdout or r.stderr or "?").strip().splitlines()
            ver = ver[-1].strip() if ver else "?"
            name = "notebook" if "notebook" in cmd else "jupyterlab"
            return f"{name} {ver} via {pathlib.Path(sys.executable).name}"
        errors.append((r.stderr or r.stdout or "not installed").strip().splitlines()[-1:])
    raise RuntimeError(f"not available in this Python -> {sys.executable} -m pip install notebook")


def _cwd():
    here, cwd = HERE.resolve(), pathlib.Path.cwd().resolve()
    if here != cwd:
        raise RuntimeError(f"run this from the pack folder -> cd {here}")
    return str(cwd)


print("\nACT-Africa bearing challenge - environment check\n")
check("python 3.9 or newer", _python)
for mod, pipname in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
]:
    check(f"{pipname} installed", _imp(mod))
check("data readable", _data)
check("protocol check passes", _protocol)
check("a model actually fits", _fit)
check("jupyter launchable", _jupyter)
check("working directory correct", _cwd)

print()
if ok:
    print("=" * 60)
    print("ENVIRONMENT STATUS: READY")
    print("=" * 60)
    print(f"Launch from THIS folder with:\n  {sys.executable} -m notebook starter.ipynb")
    print(f"Protocol self-test:\n  {sys.executable} self_test.py\n")
else:
    print("=" * 60)
    print("ENVIRONMENT STATUS: NOT READY")
    print("=" * 60)
    print("Fastest fix: install the supplied requirements into this Python:")
    print(f"  {sys.executable} -m pip install -r requirements.txt\n")
    print("Windows participants may instead double-click install_windows.bat.")
    print("macOS/Linux participants may run: bash install_mac_linux.sh\n")
    print("If only the working-directory check fails, open the terminal in the folder")
    print("that contains starter.ipynb and run the checker again.\n")
    print("If data readable fails after NumPy is OK, re-extract/copy the pack.")
    print("If that does not fix it, ask a facilitator for the offline rescue environment.\n")

sys.exit(0 if ok else 1)
