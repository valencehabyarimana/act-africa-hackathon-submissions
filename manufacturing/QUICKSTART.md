# Quickstart

### For anyone who has not done this before

You do not need a machine-learning background to take part or to contribute something
useful to your team. This page gets you to a valid scored submission in about fifteen
minutes.

Read this instead of `README.md` if you want the shortest route to getting started.

---

## What the problem is, in plain terms

A bearing is part of a rotating machine that allows a shaft to spin. When a bearing starts
to fail, it can produce small repeated impacts that are captured by a vibration sensor.

This pack contains measurements from sixteen bearings.

**Every window in the pack is labelled.** There is no hidden test set and nothing is
withheld from you. What differs between the bearings is how the damage arose:

| bearing group | how many | how the damage arose |
|---|---:|---|
| healthy | 4 | undamaged (K001–K004) |
| real damage | 8 | developed during accelerated lifetime tests |
| machined damage | 4 | introduced deliberately in a laboratory — KA01, KA03, KI01, KI03 |

Your challenge uses twelve bearings: the four healthy bearings plus the eight with real
damage.

The evaluation is simple but demanding: hold out one bearing completely, train on the
other eleven, and test whether the model can diagnose the bearing it has never seen.
Repeat this twelve times so that every evaluation bearing is held out once.

The four machined-damage bearings are never evaluation folds. They are additional
training material that you may use or ignore.

This is the central question of the challenge:

> **Can a model diagnose a bearing it has never seen?**

---

## Step 1 — check your laptop

### Windows — easiest route

Double-click **`install_windows.bat`**. It uses the Python already available on your
machine, installs the packages in `requirements.txt`, and runs the environment check.

If you prefer the terminal, open one in this folder and run:

```bat
py -m pip install -r requirements.txt
py check_environment.py
```

If `py` is not recognised but `python` is, replace `py` with `python`.

### macOS / Linux

```bash
python3 -m pip install -r requirements.txt
python3 check_environment.py
```

Or run:

```bash
bash install_mac_linux.sh
```

Every check should say `OK`, and the final line should show:

```text
ENVIRONMENT STATUS: READY
```

If any line says `FAIL`, ask a facilitator for help before continuing.

**Recommended workshop Python:** 3.11. Python 3.9–3.13 has been workshop-tested.
A different version is acceptable only if `check_environment.py` reaches `READY`.

---

## Step 2 — open the notebook

Launch Jupyter through the same Python interpreter you used above. This avoids Windows
PATH problems where Python works but `jupyter.exe` cannot be found.

**Windows:**

```bat
py -m notebook starter.ipynb
```

**macOS / Linux:**

```bash
python3 -m notebook starter.ipynb
```

A page will open in your browser containing explanatory text and grey code cells.

Click a code cell and press **Shift + Enter** to run it. Work down the notebook in order.
Do not skip cells because later cells depend on earlier ones.

### If Step 2 does not work

**Notebook/Jupyter is missing** — install the supplied requirements and rerun the checker:

```bat
py -m pip install -r requirements.txt
py check_environment.py
```

On macOS/Linux, use `python3` instead of `py`.

**`ModuleNotFoundError: No module named 'evaluate'`** — Jupyter was started from the
wrong folder. Close it (`Ctrl+C` in the terminal), open a terminal in the folder containing
`starter.ipynb`, and launch it again from there.

`check_environment.py` checks the required packages, data, protocol, model fit, Jupyter,
and working directory. If it ends with `ENVIRONMENT STATUS: READY`, continue.

---

## Step 3 — get a valid result

Run everything down to the end of **section 3**.

It takes about two minutes because the model is repeatedly trained and evaluated across
the twelve leave-one-bearing-out folds. It is not stuck.

You will see three main reference results, all evaluated using the same 12-bearing LOBO
protocol:

- **0.322** — envelope reference, approximately at chance level
- **0.554** — frozen MOMENT foundation-model reference from the lecture
- **0.634** — 17 handcrafted vibration features, the strongest of the three references

**Chance balanced accuracy = 0.333.**

Your goal is not simply to obtain a bigger number. It is to improve unseen-bearing
diagnosis **and explain why your change works**.

**You now have a valid reference result.** Whatever else happens during the session,
your team has something valid to analyse and present. Everyone should aim to reach this
point by about 15:15.

### One rule decides whether your result counts

> **Only a number produced by `lobo_score` is a hackathon result. Your first slide must
> show `verify_protocol` → `PASSED`. A team that cannot show it has no valid score,
> however high the number looks.**

The tempting shortcut is `train_test_split`, which is common in machine-learning
tutorials. **It is invalid for this challenge and will not be accepted.**

Each physical bearing produces many related windows. A random window-level split can
therefore place windows from the **same physical bearing** in both training and testing.
The model can then exploit bearing-specific information instead of demonstrating
generalization to a completely unseen bearing.

For this dataset, a stratified random split produces approximately:

```text
balanced accuracy = 0.863
```

That number is much higher than the correct LOBO result, but it answers a different and
easier question. It is therefore **not a valid hackathon result**.

There is another important leakage trap. If you fit a scaler, PCA, whitening transform,
or feature selector on the **whole dataset before LOBO evaluation**, information from the
held-out bearing has already influenced the preprocessing.

Put all data-dependent preprocessing steps **inside** the `Pipeline`, where they are
fitted only on the eleven training bearings within each fold.

The supplied `probe()` already follows this pattern correctly.

---

## Step 4 — now you have time to think

Two observations in the notebook are more interesting than simply obtaining a higher
score.

### 1. Handcrafted features outperform the frozen MOMENT representation here

Using the same classifier and the same 12-fold LOBO evaluation, the 17 handcrafted
features achieve **0.634 balanced accuracy**, compared with **0.554** for the frozen
MOMENT embeddings.

This result applies to this specific unseen-bearing evaluation. It should not be
interpreted as a general conclusion that handcrafted features are better than foundation
models.

The handcrafted features contain bearing-specific engineering information, while MOMENT
provides a general 512-dimensional time-series representation.

Why does that difference matter here?

That is a useful question for your team to investigate.

### 2. The pooled score hides strong bearing-to-bearing variation

Look at section 4 of the notebook.

Seven bearings score above **0.60**, while five score below **0.15**, with nothing in
between.

The model therefore does not perform moderately well on every bearing. It succeeds on
some bearings and fails almost completely on others.

The pooled balanced accuracy of **0.554** hides this strong bearing-to-bearing variation.

If your team can explain which bearings fail, how they fail, and what those bearings have
in common, you have understood one of the main lessons of the challenge.

---

## Useful things to do that are not modelling

Every team needs these, and none requires a machine-learning background:

- **Read the per-bearing table.** Which bearings fail? What do the failing bearings have
  in common? This is central to the diagnosis part of the judging.

- **Measure runtime.** How long does your method take? Would it work offline on a normal
  laptop in a plant? This matters for deployment.

- **Try Track B.** Take your phone, find a fan, air conditioner, water pump, or another
  rotating machine, and record about a minute of it. `track_b_your_machine.ipynb` walks
  you through the analysis. It requires no fault labels.

- **Prepare the three slides.** Start early. A clear explanation of what you tested,
  what failed, and what you learned is more valuable than presenting only your highest
  score.

---

## Words you will hear this afternoon

| word | what it means here |
|---|---|
| **window** | 256 milliseconds of vibration; one row of the data |
| **recording** | one continuous measurement, containing about 30 windows |
| **embedding** | 512 numbers produced by the frozen foundation model to describe a window |
| **label** | the target class: 0 healthy, 1 outer-race fault, 2 inner-race fault |
| **balanced accuracy** | the average recall across the three classes, giving each class equal importance |
| **leakage** | allowing information from the test bearing to influence model fitting or preprocessing |
| **held out** | deliberately excluded from training so that it can be used for evaluation |
| **leave-one-bearing-out (LOBO)** | hold out one bearing, train on the other eleven, test on the held-out bearing, and repeat for all twelve |
| **reference result** | a supplied result for comparison: 0.322, 0.554, and 0.634 in this challenge |

---

## If you get stuck

Ask a facilitator.

If something has not worked after about five minutes, asking for help is usually more
productive than spending an hour on an environment or setup problem.