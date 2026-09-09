# Unseen Bearing Fault Diagnosis Challenge

## ACT-Africa Hackathon 2026

### Can a model diagnose a bearing it has never seen?

This project investigates **cross-bearing generalisation** for vibration-based bearing fault diagnosis.

Rather than randomly splitting vibration windows into training and test sets, the study evaluates whether a model trained on a collection of bearings can correctly diagnose faults on a **completely unseen physical bearing**.

The central experimental question is therefore:

> **Does the learned representation capture transferable fault characteristics, or does it primarily learn bearing-specific signatures?**

---

## 1. Problem

Machine-learning models for predictive maintenance can achieve high performance when training and testing data contain observations from the same physical assets.

However, this can produce an overly optimistic estimate of real-world performance.

In deployment, a model may encounter a bearing that was never present during training.

This project therefore treats the **bearing itself as the unit of generalisation**.

---

## 2. Evaluation Strategy: Leave-One-Bearing-Out

The dataset contains vibration windows originating from **12 bearings**.

We use **Leave-One-Bearing-Out (LOBO)** evaluation.

For every fold:

```text
              TRAINING
       ┌─────────────────────┐
       │ Bearing 1            │
       │ Bearing 2            │
       │ ...                  │
       │ Bearing 11           │
       └─────────────────────┘
                  │
                  ▼
             MODEL TRAINING
                  │
                  ▼
              TESTING
       ┌─────────────────────┐
       │ Bearing 12           │
       │ completely unseen    │
       └─────────────────────┘
```

The process is repeated until every bearing has served as the unseen test bearing.

This prevents windows originating from the same physical bearing from appearing in both training and testing.

---

## 3. Representations Investigated

The study compares three different ways of representing vibration signals:

| Representation               | Balanced Accuracy |
| ---------------------------- | ----------------: |
| Raw envelope                 |             0.323 |
| Frozen MOMENT embeddings     |             0.554 |
| Classical vibration features |             0.634 |

The comparison provides an important result:

**A foundation-model representation is not automatically the best representation for an industrial time-series generalisation problem.**

Classical vibration features provide stronger initial cross-bearing generalisation than either raw envelope inputs or frozen MOMENT embeddings.

---

## 4. Progressive Model Development

The experiments then investigate whether generalisation can be improved without changing the fundamental evaluation protocol.

The development path is:

```text
MOMENT embeddings
       │
       ▼
Stronger regularisation
       │
       ▼
Additional machined-damage training data
       │
       ▼
Feature combination
       │
       ▼
Non-linear SVM
       │
       ▼
Hyperparameter tuning
```

The resulting balanced accuracy improves from:

**0.554 → 0.772**

This corresponds to an absolute improvement of **0.218** under the same unseen-bearing evaluation principle.

---

## 5. The More Important Finding

The headline score does not tell the entire story.

Performance varies substantially between physical bearings.

Some bearings generalise well, while others remain extremely difficult to diagnose.

For example, the baseline MOMENT representation achieves strong performance on several bearings but performs extremely poorly on others, with the weakest bearing achieving approximately **0.003 balanced accuracy**.

Even after improving the overall model, bearing-specific failures remain.

This reveals a deeper problem:

> **The challenge is not simply classification. It is domain generalisation across physical bearings.**

---

## 6. Why This Matters

A model can achieve a strong pooled score while still failing catastrophically on particular physical assets.

Therefore, evaluating only:

* accuracy,
* F1,
* or pooled balanced accuracy

can hide important deployment risks.

The LOBO protocol exposes these failures by asking the model to generalise across **physical bearing domains**, rather than merely across randomly sampled signal windows.

---

## 7. Research Insight

The experiments lead to a shift in the research question.

### Instead of asking:

> How can we obtain a higher classification score?

### We ask:

> **Why does the model generalise to some bearings but fail on others?**

This points toward investigation of:

* bearing-specific domain shift,
* representation transferability,
* operating-condition differences,
* fault morphology,
* feature stability,
* and the relationship between learned representations and physical asset variation.

---

## 8. Reproducibility

The main analysis is provided in the accompanying Jupyter notebook.

The notebook contains the experimental pipeline, feature extraction, model training, LOBO evaluation, and performance analysis.

Where possible, preprocessing and model fitting are performed within the training portion of each LOBO fold to prevent information from the unseen bearing entering the training process.

---

## 9. Key Takeaway

### **Generalisation is the challenge—not merely classification.**

The experiments demonstrate that:

**Random splits can overestimate performance.**

**LOBO reveals the real cross-bearing challenge.**

**Representation choice strongly affects transferability.**

**Model optimisation can substantially improve performance.**

**But pooled performance can still conceal severe bearing-specific failures.**

The next step is therefore not simply to optimise the score further, but to **understand and reduce the domain shift between physical bearings**.

---

## Citation / Attribution

Developed for the **ACT-Africa 2026 Hackathon — Unseen Bearing Fault Diagnosis Challenge**.
# Unseen-Bearing Fault Diagnosis Challenges: MOMENT Foundation Model and Classical Vibration Features

### Manufacturing & Infrastructure thematic group · Lecture 5 hackathon
Naol Dessalegn Dejene, Pusan National University, LAB. Intelligent Additive Manufacturing


**Pack v1.1** · reference results and evaluation protocol unchanged from v1.0; see
`CHANGELOG.md`. Facilitators: read `FACILITATOR_RUNBOOK.md`.

---

## Start here

```bash
py check_environment.py         # Windows (use python3 on macOS/Linux)
python self_test.py             # optional: verifies the protocol itself, ~40 s
py -m notebook starter.ipynb    # Windows (python3 -m notebook on macOS/Linux)
```

`lobo_score` and `verify_protocol` accept **only** the 12 evaluation bearings. Hand them the
whole dataset and they raise a `ValueError` telling you to apply `pool_mask()` — they will
not quietly return a number that is not comparable to anyone else's.

New to Python or machine learning? Read `QUICKSTART.md` instead of this file.

If `check_environment.py` reports a `FAIL`, see **Troubleshooting** at the end of this file
before doing anything else.

---

## The protocol gate — read this before you write any code

> **A number that did not come out of `lobo_score` is not a hackathon result.**
> **Your first slide must show `verify_protocol` → `PASSED`.**
> **A submission without it is invalid, not merely marked down.**

This is not a formality. It is the whole subject of the lecture, applied to your own work.

### Why `train_test_split` is invalid here

Every tutorial you have read splits data with `train_test_split`. On this dataset that is
not a weaker evaluation, it is a **broken** one, and results from it will not be accepted.

One bearing produces roughly 600 windows, and windows from the same recording are
near-duplicates of each other. A random split therefore puts windows from the *same
physical bearing* on both sides of the line. The model does not have to learn what an
outer-race fault looks like; it only has to recognise which bearing it is hearing.

We measured it. A stratified 75/25 random split of the frozen-model embeddings scores

```
balanced accuracy 0.863
```

Against 0.554 for the same features under the correct protocol. That 0.863 is not a good
result. It is the number this challenge exists to teach you to distrust, and it is roughly
what the 99%-accuracy bearing papers are reporting.

### The leakage rule, stated once

**Any parameter learned from data must be fitted on the eleven training bearings inside
each fold — never on the complete dataset.** That includes means and standard deviations,
PCA and whitening bases, feature selection, class-balance weights, and any threshold you
tune.

The correct pattern is to put the step inside the `Pipeline`, because `lobo_score` calls
`clone(model).fit(X_train, y_train)` per fold:

```python
# CORRECT - the PCA is fitted on the 11 training bearings, applied to the held-out one
model = Pipeline([("scale", StandardScaler()),
                  ("pca",   PCA(n_components=64)),
                  ("clf",   LogisticRegression(max_iter=2000, class_weight="balanced"))])
r = lobo_score(model, X[pool], y[pool], bearing[pool])

# WRONG - the PCA has already seen the held-out bearing before any fold is run
Xw = PCA(n_components=64).fit_transform(StandardScaler().fit_transform(X[pool]))
r  = lobo_score(probe(), Xw, y[pool], bearing[pool])          # invalid
```

`lobo_score` **cannot detect the wrong version** — by the time it receives `Xw`, the
leakage has already happened upstream. Note also that on this dataset the leaked version
scores **0.543** against **0.564** for the correct one: leakage here makes the number
slightly *worse*, so you cannot spot it by looking at the score. Only reading the code
finds it. That is why the protocol is judged on your slide, not on your number.

---

## The question

A rotating machine — a borehole pump, a maize mill, a generator — fails from the bearings
outward. Vibration carries the signature long before it is audible. Every year the
literature reports another model classifying bearing faults at 99% accuracy, and almost none
survive contact with a plant, because they are validated on the *same physical bearings*
they were trained on.

> **Can a model diagnose a bearing it has never seen?**

The lecture answered it for one model: **0.554** for the frozen foundation model. Your
objective is not simply to push that number up.

> **Can you improve unseen-bearing diagnosis while keeping the LOBO evaluation correct —
> and explain why your change works?**

A team that raises the score and cannot account for it has done less than a team that
explains the five failing bearings.

---

## One protocol, everywhere

**Leave-one-bearing-out over twelve bearings.** Train on eleven, predict the one held out,
repeat for all twelve, pool the predictions, score once. Balanced accuracy, per window.

This is the identical protocol behind the lecture result, so every number you produce is
directly comparable to the **"Generalization to Unseen Bearings"** slide, to the baselines
below, and to every other team.

The evaluation pool contains **four healthy bearings and eight bearings with real damage
obtained from accelerated lifetime tests**. 7,209 windows, 240 recordings.

| | bearings | documented condition (Paderborn) |
|---|---|---|
| Healthy | K001, K002, K003, K004 | undamaged; run in for 50 h, 19 h, 1 h and 5 h respectively before testing |
| Outer-race | KA04, KA16, KA22 | fatigue: pitting, single point (levels 1, 2, 1) |
| Outer-race | KA15 | **plastic deformation: indentations**, single point, level 1 |
| Inner-race | KI04, KI14, KI17 | fatigue: pitting, single point, level 1 |
| Inner-race | KI16 | fatigue: pitting, single point, level 3 |

Seven of the eight damaged bearings are fatigue pitting; **KA15 is plastic deformation with
indentations**, a different mechanism. Damage severity also varies. Neither fact is a
nuisance — both are covariates you can use when you try to explain which bearings fail.

**There is no hidden test set, and that is deliberate.** With four bearings per class, a
single held-out split is a lottery — we measured all 64 ways of holding out one bearing per
class and the score lands anywhere between **0.070 and 0.968** depending purely on which
three you pick. Rotating through all twelve removes that luck. It is also exactly what the
lecture did, which is why the numbers match.

**This is a transparent LOBO development benchmark for learning. It is not a blinded final
test set.** You can look at the score while you change your model, which means the final
number carries some selection bias by construction — the same bias any researcher has when
they tune against a cross-validation score. Nothing removes that in an afternoon.

What makes your result trustworthy is that the search is visible. Score with `lobo_score`
and nothing else, do not rebuild the folds, and report what you tried and why you chose the
configuration you did — including the things that did not work.

### Three reference results

Linear probe — standardise, then logistic regression with `class_weight="balanced"`. All
measured under the protocol above, on the same twelve bearings.

| reference | balanced accuracy | macro F1 |
|---|---|---|
| envelope reference — raw envelope window | 0.322 | 0.322 |
| chance | 0.333 | — |
| **foundation-model reference — MOMENT frozen, 512-d** | **0.554** | 0.555 |
| MOMENT + the 4 machined bearings added to training | 0.589 | 0.589 |
| **classical-feature reference — 17-d handcrafted** | **0.634** | 0.638 |

**0.634 is the strongest reference here, not 0.554.** 0.554 is the foundation-model result
from the lecture and it is the one to compare a foundation-model approach against; the
classical features score higher. Switching feature sets clears 0.554 in one line, so
"beating 0.554" is not, by itself, an achievement — say which reference you are comparing
against and why.

The gap holds across four classifier heads: logistic regression 0.634 vs 0.554, strong
regularisation 0.681 vs 0.568, LDA 0.665 vs 0.528, and random forest 0.681 for the classical
features (the embedding case did not finish, so treat it as unmeasured). The 17 classical
features encode decades of bearing physics — BPFO at 3.053× shaft rate, BPFI at 4.947×, four
harmonics of each. The foundation model was never told what a bearing is.

Why that happens on this dataset is an open question and the most interesting thing in the
challenge. Explaining it is worth more than moving any of these numbers.

---

## What is in the box

```
data/challenge.npz           9,624 windows · 16 bearings · fully labelled
starter.ipynb                Track A, work through it in order
track_b_your_machine.ipynb   Track B, needs a phone and a machine
evaluate.py                  the protocol: lobo_score, verify_protocol, report
self_test.py                 asserts the protocol guards and the reference values
byo_machine.py               Track B preprocessing
check_environment.py         run this first
FACILITATOR_RUNBOOK.md       session timing, protocol gate, troubleshooting (facilitators)
demo/                        a simulated recording, so Track B runs without a phone
```

29 MB. Everything runs on a laptop CPU with no internet.

The pack contains **16 bearings**: 4 healthy, 8 with real damage from accelerated lifetime
tests, and 4 with damage machined in a laboratory. The 12 in the evaluation pool are the
4 healthy plus the 8 real-damage bearings.

| array | width | what it is |
|---|---|---|
| `embeddings` | 512 | the frozen foundation model's description of the window |
| `classical` | 17 | RMS, std, peak, peak-to-peak, kurtosis, skewness, crest / shape / clearance factor, plus envelope-spectrum energy at four harmonics of BPFO and BPFI |
| `envelope` | 512 | the demodulated window itself, before any model touches it |

Plus `y`, `bearing`, `recording`, `origin`, `condition`.

**A recording is one continuous measurement**, about 30 windows. Windows from one recording
are near-duplicates — never split them across a fold. `verify_protocol()` checks this.

### The four machined-damage bearings

KA01, KA03, KI01, KI03 had their damage machined in — by electric discharge machining or an
electric engraver — rather than developed in an accelerated lifetime test. They are
**never an evaluation fold**. They are 2,415 extra labelled windows you may add
to every training fold via `extra_train`, which leaves the protocol unchanged and the number
comparable:

```python
lobo_score(model, X[pool], y[pool], bearing[pool],
           extra_train=(X[~pool], y[~pool], bearing[~pool]))
```

The bearing codes are required and are checked — only the four machined bearings may be
added, and passing any evaluation bearing raises a `ValueError`. Doing this is worth
**+0.035** to the frozen model. Whether you can do better with them is a real question.

---

## Track A — improve unseen-bearing diagnosis, and explain it

Three classes: healthy, outer-race, inner-race.

### Rules

1. **CPU only.** No GPU, no fine-tuning the encoder.
2. **Score with `lobo_score`.** Any other protocol is not comparable.
3. **Do not rebuild the folds.**
4. **Fit every preprocessing parameter on the training bearings only.**

> **Leakage rule.** Any preprocessing parameter learned from data — a mean, a standard
> deviation, a PCA basis, a whitening matrix — must be fitted on the **training bearings
> only**, inside the fold, unless you explicitly declare your method as target adaptation
> and report it separately from your LOBO number. The supplied `probe()` already does this
> correctly: the `StandardScaler` sits inside the `Pipeline`, so `clone(model).fit(X_train,
> y_train)` fits it on the eleven training bearings and merely applies it to the held-out
> one.

One tempting idea this rules out: standardising each bearing using that bearing's own
windows. Inside a fold that uses the held-out bearing's own distribution before predicting
it — transductive, not inductive. It is also useless here, measurably: because each bearing
carries exactly one class, centring per bearing removes the class signal along with the
nuisance variation. Under this protocol it scores **0.3333**, exactly chance, with every
healthy bearing at recall 1.000 and every damaged bearing at 0.000. If you want target
adaptation, declare it and report it separately.

5. **Your notebook must run top to bottom**, and produce the number you report.
6. **Report what you tried**, including what did not work, and why you chose your final
   configuration.

### The per-bearing table is the real result

Seven bearings score above 0.60. Five score below 0.15. **Nothing in between.** The model
does not work moderately well everywhere — it works on some bearings and fails on others,
and 0.554 describes none of them. A team reporting only the pooled number has hidden its
most interesting finding.

### Ideas worth the afternoon

- **Strong regularisation.** 512 dimensions from 11 training bearings. C = 0.001 gives 0.578.
- **Reduce nuisance variation the safe way.** If between-bearing variation is hurting you,
  put the transform — a PCA, a whitening step — inside the `Pipeline`, where `clone().fit()`
  will fit it on the eleven training bearings for you.
- **Combine embeddings with classical features.** The physics plus whatever it misses.
- **A reject option.** Abstain when uncertain, report coverage alongside accuracy. A model
  that says "I don't know" on the five hard bearings is more deployable than one that
  guesses.
- **Fix one bearing.** Take KA15 at 0.005 and work out what would have to be true.

---

## Track B — bring your own machine

**Optional, no labels required, and the more interesting of the two.**

Record any rotating machine you can reach with a phone at 200–400 Hz. Run the same envelope
preprocessing, fit a Gaussian to normal operation, score everything else by Mahalanobis
distance. No fault labels anywhere — the only realistic starting point for almost every
machine in the world, and where an African condition-monitoring dataset would begin.

See `track_b_your_machine.ipynb`. Judged on the quality of the question and the honesty of
the evaluation, never on accuracy.

---

## What we are looking for, in this order

0. **A valid protocol — a gate, not a criterion.** `verify_protocol` → `PASSED` on your
   first slide, and every number you report produced by `lobo_score`. Without this the
   submission is **invalid** and is not scored against the criteria below. This is not a
   deduction; there is nothing to deduct from. If you are unsure whether something you did
   counts as leakage, ask a facilitator before 16:30 rather than after.
1. **A defensible search.** Make your experimental process visible: what you changed, what
   else you tried, why you chose this one.
2. **A diagnosis, not a score.** Which bearings fail, which class they fail into, and what
   that means physically.
3. **Deployability.** CPU, offline, false-alarm rate, and what a false alarm costs.
4. **Accuracy.** Last, deliberately.

---

## Presenting

Five minutes, three slides, Wednesday at 11:00. What you held out · the per-bearing table
and confusion matrix · what deployment would cost. Then the pooled number, once, at the end.

If your change did not help, say so and say why. That is a result.

---

## Troubleshooting

`check_environment.py` tests for all of these and names the fix. Run it first; run it again
after any fix.

| what you see | what it means | fix |
|---|---|---|
| ``Jupyter command `jupyter-notebook` not found`` | Jupyter is not installed | `py -m pip install -r requirements.txt` (Windows) or `python3 -m pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'evaluate'` | Jupyter was started outside the pack folder | Ctrl+C, `cd` into the folder holding `starter.ipynb`, relaunch |
| `ModuleNotFoundError: No module named 'sklearn'` | missing packages | `py -m pip install -r requirements.txt` (Windows) or `python3 -m pip install -r requirements.txt` |
| `FileNotFoundError: data/challenge.npz` | the pack did not unzip fully, or wrong folder | re-copy from the USB; check `data/` is 28 MB |
| `ValueError: unexpected bearings in the evaluation set` | you forgot `pool = pool_mask(bearing)` | the error message contains the exact line to use |
| `ValueError: extra_train contains evaluation bearings` | you passed pool bearings as extra training data | `extra_train=(X[~pool], y[~pool], bearing[~pool])` |
| section 3 seems frozen | it is fitting 12 folds × 3 feature sets | wait ~2 min; it is not stuck |

Nothing in this pack needs internet after you have copied it. If a fix requires `pip` and
you have no connection, a facilitator has an offline wheel folder on the USB.

---

## Data and attribution

Paderborn University KAt Data Center bearing dataset — a 6203 rolling bearing in a modular
test rig.

- Original data: <https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter>

### Licence

**CC BY-NC 4.0** — <https://creativecommons.org/licenses/by-nc/4.0/>. Non-commercial
academic use is permitted; attribution is required. For commercial use, contact the dataset
authors.

**This pack is a modified derivative**, not the original: band-pass filtered (1–10 kHz,
4th-order Butterworth), Hilbert-envelope demodulated with DC removed, decimated 64 kHz →
2 kHz, windowed at 16,384 raw samples with 50% overlap, then encoded with MOMENT-1-small,
frozen. Applying the envelope *before* decimating is the step that matters — bearing impacts
excite resonances at 2–20 kHz and naive downsampling destroys them.

### Required attribution

Put this on every team slide and anything published afterwards:

> Christian Lessmeier et al., KAt-DataCenter,
> mb.uni-paderborn.de/kat/forschung/bearing-datacenter, Chair of Design and Drive
> Technology, Paderborn University. Data licensed CC BY-NC 4.0. Modified for ACT-Africa 2026.

And cite: Lessmeier, C., Kimotho, J. K., Zimmer, D., Sextro, W. (2016). *Condition
monitoring of bearing damage in electromechanical drive systems by using motor current
signals of electric motors: a benchmark data set for data-driven classification.* European
Conference of the PHM Society, Bilbao.
