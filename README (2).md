# Iris Flower Classification

A neural network built with Keras and TensorFlow to classify Iris flowers into three species, evaluated with 10-fold cross-validation using scikit-learn.

## Overview

The Iris dataset contains 150 samples, 50 for each of three Iris species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

The goal is to predict the species of an unseen Iris plant from four measurements. This is a multi-class classification problem.

## Dataset

The notebook reads `data/iris_dataset.csv` (no header row) with these columns:

| Column | Description |
|--------|-------------|
| 0 | Sepal length (cm) |
| 1 | Sepal width (cm) |
| 2 | Petal length (cm) |
| 3 | Petal width (cm) |
| 4 | Class (species) |

## Project Structure

```
.
├── Iris_flower_classification.ipynb
├── data/
│   └── iris_dataset.csv
└── README.md
```

## Requirements

- Python 3.5+
- pandas
- numpy
- keras (with TensorFlow backend)
- scikit-learn
- matplotlib
- jupyter

```bash
pip install pandas numpy keras tensorflow scikit-learn matplotlib jupyter
```

> **Note:** The notebook was written for an older Keras/TensorFlow stack. It uses `keras.utils.np_utils` and `keras.wrappers.scikit_learn.KerasClassifier`, which have been removed in recent versions. See [Compatibility](#compatibility) below.

## Methodology

1. **Load data**: read the CSV into a pandas DataFrame.
2. **Visualize**: scatter plots of sepal length vs. width and petal length vs. width, colored by species.
3. **Prepare data**: split into features `X` (4 columns) and labels `y`.
4. **Encode labels**: `LabelEncoder` followed by one-hot encoding (`to_categorical`).
5. **Build the model**: a baseline fully connected network:
   - Input: 4 features
   - Hidden layer: Dense, 8 neurons, ReLU
   - Output layer: Dense, 3 neurons, softmax
   - Loss: categorical cross-entropy; optimizer: Adam; metric: accuracy
6. **Wrap for scikit-learn**: `KerasClassifier` with 200 epochs and batch size 5.
7. **Evaluate**: 10-fold cross-validation (shuffled, seed 7) with `cross_val_score`.

## Results

| Metric | Value |
|--------|-------|
| Mean accuracy (10-fold CV) | **97.33%** |
| Standard deviation | 4.42% |

Per-fold accuracies ranged from 86.67% to 100%.

## Usage

```bash
jupyter notebook Iris_flower_classification.ipynb
```

Place `iris_dataset.csv` in a `data/` folder next to the notebook and run the cells in order.

The notebook reads the file with a Windows-style path (`"data\iris_dataset.csv"`). On Linux or macOS, change it to `"data/iris_dataset.csv"`.

## Compatibility

With modern TensorFlow 2.x, update these imports:

```python
# Old
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
dummy_y = np_utils.to_categorical(encoder_y)

# New
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier   # pip install scikeras
dummy_y = to_categorical(encoder_y)
```

With SciKeras, `KerasClassifier` takes `model=baseline_model` instead of `build_fn=baseline_model`.

## Possible Improvements

- Hold out a separate test set for final evaluation
- Tune the hidden layer size, epochs, and batch size
- Compare against classical models such as logistic regression, SVM, or random forest
