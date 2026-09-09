# Your three slides

Five minutes. Three slides. Copy this, fill it in, delete the italics.

Do not open with your score. Open with what you held out — it is the first thing you are
judged on and the first thing you will be asked.

> ## ⚠ Slide 1 is a validity gate
>
> **A submission whose Slide 1 does not show `verify_protocol` → `PASSED`, with every
> reported number produced by `lobo_score`, is invalid.** It is not marked down; it is not
> scored at all. Results from `train_test_split` or any rebuilt fold structure are invalid
> for the same reason — windows from one bearing land on both sides of the split, so the
> model recognises the bearing rather than the fault (about 0.863, against 0.554 done
> properly).
>
> If you are unsure whether something you did counts as leakage, ask a facilitator **before
> 16:30**. Asking is free. Finding out on stage is not.

---

## Slide 1 — What we held out  *(required)*

**Team:** ____________  **Track:** A / B

*One picture or one table showing your split. Which bearings trained, which were held out,
at what level.*

- Protocol: leave-one-bearing-out over the 12 bearings, unchanged
- **`verify_protocol` output: PASSED**  ← paste the actual output, not a tick box
- **Every number on these slides came from `lobo_score`:** yes / no
- **Every learned preprocessing step was fitted inside the fold:** yes / no
  *(scaler, PCA, whitening, feature selection, tuned thresholds — inside the `Pipeline`)*
- What we changed from the reference: _______________
- What else we tried, including what failed: _______________
- We chose this configuration because: _______________
- Our pooled balanced accuracy: **____**
  (references: envelope 0.322 · chance 0.333 · MOMENT 0.554 · classical 0.634)

---

## Slide 2 — The per-bearing table, and the confusion matrix

*Paste the confusion matrix. Then one sentence per class — not a description of the
numbers, an explanation of them.*

|  | pred. healthy | pred. outer | pred. inner |
|---|---|---|---|
| **true healthy** | | | |
| **true outer** | | | |
| **true inner** | | | |

- Healthy: _______________
- Outer-race: _______________
- Inner-race: _______________

**Bearings above 0.60:** ____    **below 0.15:** ____    **in between:** ____

**The failure that matters most is:** _______________
**and we think it happens because:** _______________

*Candidate explanations to investigate — not answers. Any of these could produce the errors
you see, and the confusion matrix alone does not tell you which:*

- *insufficient fault-frequency information surviving the preprocessing*
- *overlapping representations for the two fault classes*
- *damage morphology and severity — KA15 is plastic deformation, not pitting; KI16 is level 3*
- *signal-to-noise ratio, or the amount of run-in on the healthy bearings*
- *decision threshold and class balance*
- *model bias, or domain shift between bearings for some other reason*

*BPFO sits at 3.053× shaft rate and BPFI at 4.947×, so a frequency argument is testable —
go and test it rather than asserting it. Say which explanation you checked, how, and what
you found. "We could not distinguish these two explanations" is a legitimate answer.*

---

## Slide 3 — What it would cost to deploy

- Runs in ______ seconds on ______ *(CPU / GPU, what machine)*
- Needs internet: yes / no
- False-alarm rate on healthy recordings: ______ %
- If a maintenance team acted on every alert, that is about ______ wasted call-outs
  per ______ machines per month
- **We would / would not put this in front of an engineer today, because:** _______________

**Pooled balanced accuracy:** ______  *(mention it here, once, at the end)*

---

## If it did not work

Say so, and say why. Explaining a failure correctly scores above reporting a number you
cannot account for. The single best thing you can say tomorrow is:

> *"We expected X. We got Y. Here is what we think the difference means."*
