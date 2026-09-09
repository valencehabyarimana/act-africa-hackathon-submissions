# ACT-Africa 2026 Temperature Prediction Notebook

## Overview

This Google Colab notebook develops and evaluates machine-learning models for predicting temperature, with `Temp(oC)` configured as the target variable. It compares an optimized Random Forest regressor with a TensorFlow/Keras artificial neural network (ANN), using a dedicated training file and an external test file. 

The workflow supports reproducible experimentation, hyperparameter optimisation, model evaluation, visualisation, and persistence of trained models and supporting artifacts. 

## Objectives

The notebook is designed to:

- Load meteorological or related tabular data from Google Drive.
- Separate predictors from the target temperature variable, `Temp(oC)`.
- Split the available training data into training, validation, and internal test subsets.
- Tune and train a Random Forest regression model.
- Tune and train a Keras-based ANN regression model.
- Evaluate model performance using \(R^2\), RMSE, and MAE.
- Save models, preprocessing objects, metrics, charts, and prediction outputs for later deployment or analysis. 

## Environment and setup

The notebook runs in Google Colab and mounts Google Drive at `/content/drive`. The project files are expected under:

```text
/content/drive/MyDrive/Colab Notebooks/ACT-Africa2026/
```

The input and output locations are configured as follows:

```python
PROJECT_DIR = "/content/drive/MyDrive/Colab Notebooks/ACT-Africa2026"

TRAIN_FILE = f"{PROJECT_DIR}/Train_Data.csv"
TEST_FILE = f"{PROJECT_DIR}/Test_Data.csv"
OUTPUT_DIR = f"{PROJECT_DIR}/TrainedNetworks_hack"
```

The notebook creates the output directory automatically if it does not already exist, and checks that both input CSV files are present before model training begins. 

## Dependencies

The notebook uses the following main packages:

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading, transformation, and numerical operations |
| `scikit-learn` | Data splitting, standardisation, Random Forest, grid search, and evaluation metrics |
| `tensorflow` | ANN construction, training, and model export |
| `matplotlib`, `seaborn` | Model-performance and prediction visualisations |
| `pickle`, `joblib` | Saving trained models and preprocessing artifacts |
| `shap` | Imported for model explainability, though no SHAP analysis is shown in the visible workflow |
| `torch`, `scipy` | Available in the environment for potential extensions |

A fixed seed (`SEED = 42`) is assigned to Python, NumPy, and TensorFlow to improve reproducibility across executions.

## Input data

The workflow expects two CSV datasets:

- `Train_Data.csv`: used for model development, including training, validation, and internal testing.
- `Test_Data.csv`: used as the separate evaluation dataset.

The recorded data dimensions are:

| Dataset | Rows | Columns |
|---|---:|---:|
| Training data | 1,048,575 | 7 |
| External test data | 519,310 | 7 |
| Input features | 6 | — |
| Target variable | `Temp(oC)` | — |

The notebook extracts six independent variables and uses `Temp(oC)` as the dependent variable. 

## Data preparation

The training dataset is partitioned into three subsets:

| Subset | Samples | Intended use |
|---|---:|---|
| Training | 363,517 | Fit model parameters |
| Validation | 77,896 | Hyperparameter selection and training monitoring |
| Internal test | 77,897 | Additional holdout evaluation |

Feature scaling is conducted with `StandardScaler`, and the fitted scaler is saved as `scaler.pkl`. The same scaler should be reused when predicting temperatures for any new data so that its features use the same scale as the model-training data. 
## Random Forest model

### Hyperparameter tuning

The notebook defines `gridsearch_random_forest()` to perform a three-fold cross-validated grid search over the following Random Forest settings:

```python
param_grid = {
    "n_estimators": [10, 50],
    "max_depth": [10, 20]
}
```

Model selection is based on validation \(R^2\), and `GridSearchCV` runs with `n_jobs=-1` to use available CPU resources. A horizontal bar chart of the tested configurations is saved as `gridsearch_results_bar_chart.png`. 
### Final training

The function `train_random_forest_optimized()` fits a `RandomForestRegressor` using the selected values of `n_estimators` and `max_depth`. The trained model is serialized to:

```text
TrainedNetworks_hack/optimized_rf_model.pkl
```

It also plots actual and predicted temperatures for the final 100 evaluation samples and constructs an approximate uncertainty band from the variability among individual trees. 

### Random Forest results

The displayed Random Forest performance is:

| Metric | Result |
|---|---:|
| \(R^2\) | 0.9983 |
| RMSE | 1.0642 °C |
| MAE | 0.7013 °C |

These results indicate that the Random Forest explains approximately 99.83% of the variation in the evaluated temperature values, with relatively low prediction error in temperature units. 

## ANN model

### Model approach

The ANN is implemented using TensorFlow/Keras through a function named `build_keras_ann()`. The model is configured for regression and supports alternative optimizers and learning rates, including Adam, RMSprop, and SGD. 

The visible hyperparameter search evaluates eight ANN configurations using a 100,000-sample subset for faster optimisation. The combinations vary the optimizer, learning rate (`0.01` or `0.001`), and number of epochs (`10` or `15`). 

### Search configurations

| Optimizer | Learning rate | Epochs |
|---|---:|---:|
| Adam | 0.01 | 10, 15 |
| RMSprop | 0.01 | 10, 15 |
| Adam | 0.001 | 10, 15 |
| RMSprop | 0.001 | 10, 15 |

The grid-search outputs are saved to `keras_ann_grid_search_results.csv`, and a bar chart of the results is saved as `keras_ann_grid_search_bar_chart.png`. 
### ANN evaluation

After hyperparameter selection, the best ANN is retrained and evaluated on the test data. The final displayed performance is:

| Metric | Result |
|---|---:|
| Test RMSE | 3.995259 °C |
| Test MAE | 3.034772 °C |
| Test \(R^2\) | 0.975683 |
| Epochs trained | 10 |

The ANN therefore explains about 97.57% of the variation in the evaluated target values, although its displayed test error is higher than that of the Random Forest model. 

## Evaluation metrics

The notebook uses the following regression metrics:

\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

- **\(R^2\)**: measures the proportion of variance in observed temperatures explained by the model; values closer to 1 are better.
- **RMSE**: measures the typical magnitude of prediction error while placing greater weight on larger errors.
- **MAE**: measures the average absolute difference between actual and predicted temperature.

The `evaluate_model()` function returns each model name together with rounded \(R^2\), RMSE, and MAE values.
## Generated outputs

All artifacts are saved in:

```text
/content/drive/MyDrive/Colab Notebooks/ACT-Africa2026/TrainedNetworks_hack/
```

| File | Description |
|---|---|
| `optimized_rf_model.pkl` | Serialized optimized Random Forest model |
| `best_keras_ann_model.keras` | Saved Keras ANN model |
| `scaler.pkl` | Fitted feature scaler |
| `gridsearch_results_bar_chart.png` | Random Forest hyperparameter-search chart |
| `predicted_vs_actual_plot.png` | Random Forest actual-versus-predicted plot with approximate uncertainty interval |
| `keras_ann_grid_search_bar_chart.png` | ANN hyperparameter-search chart |
| `keras_ann_grid_search_results.csv` | ANN configuration and validation-results table |
| `keras_ann_test_metrics.csv` | Final ANN test metrics |
| `keras_ann_last_100_actual_predicted_95CI.png` | ANN actual-versus-predicted chart for the last 100 samples |
| `keras_ann_last_100_predictions_95CI.csv` | ANN prediction values and uncertainty-related outputs for the last 100 samples |

These files enable reproducible comparison, model reuse, and reporting without rerunning the complete training workflow. 

## Using a saved model

To predict temperature for new observations, prepare a DataFrame with the same six predictor columns used during training, in exactly the same feature order. Load the saved scaler, transform the input data, then load the model and generate predictions.

```python
import pickle
import pandas as pd

# Load trained Random Forest and scaler
with open("optimized_rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# new_data must contain the same six feature columns used in training
new_data_scaled = scaler.transform(new_data)

# Generate temperature predictions
temperature_predictions = rf_model.predict(new_data_scaled)
```

For the ANN, load `best_keras_ann_model.keras` using `tf.keras.models.load_model()` and apply the same saved scaler before inference. The scaling step is essential because the ANN was trained using standardized feature inputs.

## Notes and recommendations

- The Random Forest has the stronger displayed predictive performance in this run, with higher \(R^2\) and lower RMSE/MAE than the ANN. 

- Keep feature names, feature order, missing-value handling, and measurement units consistent between training and inference.
- The displayed ANN tuning table appears to rank configurations in ascending numerical order of rank while the strongest shown validation \(R^2\) is 0.743668 for RMSprop, learning rate 0.01, and 15 epochs; verify the ranking logic before interpreting the reported “best” configuration. 

- The Random Forest interval is based on dispersion across decision trees, so it is an approximate model-ensemble uncertainty indication rather than a formally calibrated prediction interval. 
