# MedSigLIP Chest X-ray Classification and Audit

## Project Description

This project is a medical-imaging machine-learning application that classifies
chest X-ray images into four categories: `COVID`, `NORMAL`, `PNEUMONIA`, and `TB`.
It was developed for the ACT-Africa 2026 Hackathon by Group 2, Health.

The project combines Google's pretrained MedSigLIP model with a lightweight
classifier and a Streamlit interface. In addition to displaying predictions, the
application evaluates how reliable those predictions are by examining model
performance, possible data leakage, shortcut learning, calibration, zero-shot
classification, and common failure cases.

This is an educational and research demonstration. It is not a clinical
diagnostic system and must not be used to make medical decisions.

## Project Architecture

```text
Chest X-ray images
  |
  v
MedSigLIP vision encoder
  |
  v
Image embeddings
  |
  v
Lightweight classifier head
  |
  v
Quantized TensorFlow Lite model
  |
  v
Streamlit application
```

The MedSigLIP encoder is used during the model-development workflow to create
image embeddings. A small classifier is trained on those embeddings and exported
as a quantized TensorFlow Lite model. The Streamlit app uses the saved evaluation
bundle and the lightweight classifier to present the results and audit findings.

## Main Features

- Four-class chest X-ray classification.
- Accuracy, confusion matrix, ROC, precision-recall, and t-SNE visualizations.
- Near-duplicate analysis between training and test images.
- Shortcut tests using blurred images, border pixels, and image metadata.
- Zero-shot classification with MedSigLIP text prompts.
- Confidence calibration and overconfident-error analysis.
- Failure gallery and interactive prediction explorer.
- TFLite model size, accuracy, and latency evaluation.

## Dataset

The project uses the **Nigeria Chest X-ray Dataset**, organized into the four
classes used by the application: COVID, NORMAL, PNEUMONIA, and TB.

Original Kaggle dataset:

https://www.kaggle.com/datasets/aminumusa/nigeria-chest-x-ray-dataset

Please review the dataset's terms and licensing conditions before using or
redistributing the images. This repository contains derived evaluation assets and
low-resolution thumbnails, not the original dataset.

## Repository Contents

```text
Health/
  app.py                                      # Streamlit application
  requirements.txt                            # Application dependencies
  ACT_2026_Foundation_MedSigLIP_Workshop_.ipynb
                 # Model-development notebook
  assets/
    meta.json                                 # Experiment metadata and metrics
    results.npz                               # Saved embeddings and scores
    cxr_classifier_quant.tflite              # Quantized classifier
    thumbs/                                   # Test-image thumbnails
    plots/                                    # Evaluation visualizations
  .streamlit/config.toml                     # Streamlit configuration
```

## Run the Application

From the repository root:

```bash
source .venv/bin/activate
pip install -r Health/requirements.txt
streamlit run Health/app.py
```

The application expects the model and evaluation files in `Health/assets/`.

## Deployment

The app can be deployed with Streamlit Community Cloud using `Health/app.py` as
the application file and `Health/requirements.txt` as the dependency file. The
deployed application uses the lightweight TFLite runtime and does not load the
full MedSigLIP model.

## Research Context

The audit is motivated by research showing that high performance in medical-image
classification can sometimes come from shortcuts or dataset artifacts rather than
the intended clinical signal.

Reference:

DeGrave, Janizek & Lee. _AI for radiographic COVID-19 detection selects shortcuts
over signal._ Nature Machine Intelligence, 2021.
