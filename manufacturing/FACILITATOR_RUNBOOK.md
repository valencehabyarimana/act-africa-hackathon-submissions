# Facilitator runbook — one page

**Lecture 5 : Unseen-Bearing Fault Diagnosis Challenges**

Pack v1.1 · Naol Dessalegn Dejene, Intelligent Additive Manufacturing Lab., Pusan National University, South Korea

---

## Before the room fills

- **USB sticks with the pack, one per table.** 29 MB × 40 people over hotel wifi will not
  work. The Zenodo DOI (`10.5281/zenodo.22155588`, QR on the hackathon slide) is the
  backup, not the plan.
- **An offline wheel folder on each USB** for `numpy pandas scikit-learn scipy notebook`.
- Confirm your own laptop runs `py check_environment.py` on Windows / `python3 check_environment.py` on macOS/Linux → all `OK`, in the room, on the
  projector. Participants copy what they see you do.

---

## The one thing to say out loud at 14:30

> A number that did not come out of `lobo_score` is not a result.
> Slide 1 must show `verify_protocol` → **PASSED**.
> Without it your submission is invalid — not marked down, not scored.

Then say why, in one sentence: *a random `train_test_split` puts the same bearing in
training and testing and scores 0.863, and that is exactly the 99%-accuracy result the
lecture told you to distrust.*

Say it again at the 15:15 checkpoint and at the 16:30 call. Three times is not too many.
This is the single most likely way the session goes wrong: one team presents 0.863 on
Wednesday, the room believes it, and the lesson inverts on stage.

---

## Timing

| time | what happens | your job |
|---|---|---|
| **14:30–14:40** | everyone runs `check_environment.py` | Nobody opens `starter.ipynb` until `check_environment.py` ends with `ENVIRONMENT STATUS: READY`. Walk the room. Resolve every `FAIL` now — a broken environment found at 16:00 costs a team the session. |
| **14:40–15:15** | Section 3 only, nothing else | Hold them there. The goal is a first valid LOBO result for every team, not a good one. Anyone who has not run `verify_protocol` has not started. |
| **15:15** | **checkpoint — reconvene, 5 min** | Confirm every team has a scored result. Put the per-bearing table on the projector and read it together: 7 above 0.60, 5 below 0.15, nothing between. This is the content of the afternoon; teams that skip to tuning will miss it. |
| **15:20–16:30** | modelling, Track B, analysis | Circulate and ask "what did you hold out?" every time. Catch leakage now, not on stage. |
| **16:30** | **hard stop on modelling** | Slides start. A team that begins at 16:30 presents better than one that begins at 17:25. Enforce this. |
| **17:30** | close | Remind them: three slides, five minutes, Wednesday 11:00. Attribution line on every slide deck. |

---

## Assign non-modelling roles at 14:30

Do this explicitly per team, by name. Do not hope teams self-organise — the people with no
ML background will otherwise sit out the whole afternoon, and their tasks score higher than
accuracy does.

- **Per-bearing analyst.** Which bearings fail, which class they fail into, what the failing
  ones have in common. This is judging criterion 2 and outranks accuracy.
- **Timing / deployability.** How long does it run, on what hardware, offline? What is the
  false-alarm rate, and what does a false alarm cost a maintenance team? Criterion 3.
- **Track B.** A phone, a fan or pump in the hotel, one minute of normal and one minute of
  something different. No labels, no training data, and the track most likely to outlive the
  workshop.
- **Slide writer.** Starts at 16:30 with `TEAM_SLIDES_TEMPLATE.md` already open.

---

## Troubleshooting — the two that will actually happen

**1. ``Jupyter command `jupyter-notebook` not found``**
Jupyter is missing from the active Python environment. Install `requirements.txt`, then re-run `check_environment.py`. On Windows, `install_windows.bat` is the fastest route.
This is the most common failure and it lands on the barest Python installs — the
participants least able to debug it.

**2. `ModuleNotFoundError: No module named 'evaluate'`**
Jupyter was launched from the wrong folder. Ctrl+C, `cd` into the folder containing
`starter.ipynb`, relaunch. `check_environment.py` now catches both of these before they
happen — which is why nobody opens a notebook until it says `Ready`.

Everything else: `README.md` → **Troubleshooting**.

---

## Leakage: what to look for while circulating

Ask to see the code, not the number. Three questions catch nearly everything:

1. **"Show me the line that produced this number."** It must be `lobo_score(...)`.
2. **"Where is your scaler fitted?"** Inside the `Pipeline` is correct. `fit_transform` on
   the whole pool before `lobo_score` is leakage — and note it scores 0.543 against 0.564
   done properly, so it makes the number *worse*. You cannot spot this from the score, only
   from the code.
3. **"Did you touch the folds?"** `POOL_BEARINGS` is fixed. Rebuilt folds are invalid.

The guards in `evaluate.py` reject the wrong *bearings* — full 16-bearing sets, evaluation
bearings smuggled in via `extra_train`, incomplete pools. They cannot reject the wrong
*pipeline*. That gap is yours to cover, and it is covered by asking question 2.

---

## Reference numbers (all reproduced under the protocol)

| reference | balanced accuracy |
|---|---|
| raw envelope | 0.3224 |
| chance | 0.3333 |
| **MOMENT frozen, 512-d** | **0.5537** |
| MOMENT + 4 machined bearings in training | 0.5891 |
| **classical, 17-d** | **0.6344** |
| *invalid* — random `train_test_split` | *0.863* |

**0.6344 is the strongest reference, not 0.5537.** Switching feature sets clears the
foundation-model number in one line, so "beating 0.554" is not by itself an achievement.
Make teams say which reference they are comparing against.

Runtimes on one CPU core, no internet: `starter.ipynb` ≈ **107 s** end to end;
`track_b_your_machine.ipynb` ≈ **4 s**; `self_test.py` ≈ 40 s. If a team says it is stuck in
section 3, it is not — it is fitting 12 folds × 3 feature sets.

---

## What a good presentation looks like

Not the highest number. A team that explains why KA15 sits at 0.005 while K003 sits at 1.000
has done better engineering than a team reporting 0.65 it cannot account for. The strongest
thing anyone can say on Wednesday is: *"We expected X. We got Y. Here is what we think the
difference means."*
