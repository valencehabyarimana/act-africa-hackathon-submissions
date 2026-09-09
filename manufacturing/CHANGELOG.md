# Changelog
## v1.1 — 25 August 2026

Version 1.1 improves participant setup, protocol validation, leakage prevention,
documentation, and workshop usability. There is no change to the challenge data,
evaluation protocol, calculations, or reference results from v1.0.

`data/` and `demo/` are byte-identical to v1.0.
 All reference results and the 0.070 – 0.968 single-split range reproduce unchanged.

### Reference results

| Reference | Balanced accuracy |
|---|---:|
| Envelope / raw-window reference | 0.322 |
| Chance | 0.333 |
| Frozen MOMENT reference | 0.554 |
| MOMENT + 4 machined-damage bearings in training | 0.589 |
| Classical 17-feature reference | 0.634 |

### Fixed

- **`check_environment.py`** — improved environment verification and participant-facing
  failure messages. The checker now verifies that Jupyter Notebook can actually be
  launched and checks that the participant is working from the correct directory.

- **`QUICKSTART.md`** — removed the previous six/four contradiction concerning the
  machined-damage bearings. There are four: KA01, KA03, KI01, and KI03. The documentation
  now also states clearly that every window in the package is labelled.

- **`README.md` and `starter.ipynb`** — replaced the fixed "slide 19" reference with the
  descriptive "Generalization to Unseen Bearings" slide title so the documentation
  remains correct if the lecture deck is renumbered.

- **`starter.ipynb`** — changed `report(...)` to `_ = report(...)` in two cells to prevent
  the returned confusion matrix from appearing as an additional bare `array([[...]])`
  below the formatted output. The calculation is unchanged. A missing notebook cell
  `id` was also added for clean nbformat 4.5 validation.

### Added

- **Protocol gate** — stated consistently in `README.md`, `QUICKSTART.md`,
  `starter.ipynb`, `evaluate.py`, and `TEAM_SLIDES_TEMPLATE.md`. Only results produced
  through `lobo_score` are valid challenge results. Each team must show
  `verify_protocol` → `PASSED`. A submission without a valid protocol is considered
  invalid rather than simply receiving a lower score.

- **Random-split warning** — `train_test_split` is explicitly documented as invalid for
  this challenge. A stratified random split produces a measured balanced accuracy of
  0.863 because windows from the same physical bearing can occur in both training and
  test sets. This result is therefore not comparable with the LOBO benchmark.

- **Leakage guidance** — added worked correct/incorrect examples showing that any
  data-dependent preprocessing, including scaling, PCA, whitening, and feature selection,
  must be fitted only on the eleven training bearings inside each LOBO fold. A global-fit
  example scores 0.543 compared with 0.564 when performed correctly, demonstrating that
  leakage cannot reliably be detected from the final score alone.

- **Troubleshooting guidance** — added a troubleshooting table to `README.md` and
  additional setup guidance to `QUICKSTART.md` based on failures observed during
  pre-workshop testing.

- **`FACILITATOR_RUNBOOK.md`** — added facilitator timing, the protocol-gate procedure,
  troubleshooting guidance, non-modelling role suggestions, leakage questions to ask
  teams, and reference results.

- **Environment setup files** — added `requirements.txt`, `install_windows.bat`, and
  `install_mac_linux.sh` to simplify participant installation and environment setup.

### Terminology

Terminology for KA01, KA03, KI01, and KI03 has been standardised throughout the package
and lecture materials as "machined damage", "machined-damage bearings", or
"machined defects", replacing mixed wording such as "artificial",
"artificially damaged", and "machine-created".

- `README.md` — changed "The four artificially damaged bearings" to
  "The four machined-damage bearings".

- `starter.ipynb` — standardised the relevant labels, headings, and output text to use
  "machined" terminology.

- Lecture deck — standardised the relevant slide text and speaker notes from
  "artificial damage" to "machined damage".

These terminology changes do not alter the data, calculations, evaluation protocol,
or reference results.

### Environment robustness

- Jupyter detection now uses the active Python interpreter
  (`sys.executable -m notebook`) instead of depending on `jupyter.exe` being available
  on `PATH`. This fixes the Windows false-failure observed during testing.

- QUICKSTART, README, and the facilitator runbook now use interpreter-based Jupyter
  launch commands and document Windows `py` versus macOS/Linux `python3`.

- The environment checker now ends with an explicit
  `ENVIRONMENT STATUS: READY` / `NOT READY` gate.

