
# DigiSkills 3.0 - AIP301
# Hands-on Exercise No. 04
# Task 1: CNN and RNN Mushroom Classification

from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    Flatten, Dense, SimpleRNN
)

# ----------------------------------
# PROJECT PATHS
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(42)

# ----------------------------------
# STEP 1: LOAD DATASET
# ----------------------------------

print("=" * 60)
print("TASK 1: CNN AND RNN CLASSIFICATION")
print("=" * 60)

df = pd.read_csv(DATA_DIR / "mushrooms.csv")

print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 Rows:")
print(df.head().to_string(index=False))

# ----------------------------------
# STEP 2: SELECT FEATURES AND TARGET
# ----------------------------------

target_candidates = [
    col for col in df.columns
    if col.lower() in ("class", "target", "label")
]

if not target_candidates:
    raise ValueError(
        "Target column not found. Check CSV columns."
    )

target_column = target_candidates[0]

# Missing target values cannot be used for training.
df = df.dropna(subset=[target_column]).copy()

X = df.drop(columns=[target_column])
y = df[target_column]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)

print("\nTarget Column:", target_column)
print("Classes:", label_encoder.classes_)

if num_classes < 2:
    raise ValueError("At least two classes are required.")

# ----------------------------------
# STEP 3: TRAIN TEST SPLIT (70/30)
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ----------------------------------
# STEP 4: CATEGORICAL PREPROCESSING
# ----------------------------------

# Treat mushroom characteristics as categorical values.
X_train = X_train.astype("string")
X_test = X_test.astype("string")

imputer = SimpleImputer(
    strategy="most_frequent",
    missing_values=pd.NA
)

# Convert to object arrays with np.nan for compatibility.
train_raw = X_train.replace({pd.NA: np.nan}).to_numpy(dtype=object)
test_raw = X_test.replace({pd.NA: np.nan}).to_numpy(dtype=object)

train_imputed = imputer.fit_transform(train_raw)
test_imputed = imputer.transform(test_raw)

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=np.float32
)

X_train_encoded = encoder.fit_transform(train_imputed)
X_test_encoded = encoder.transform(test_imputed)

print("\nEncoded Feature Count:", X_train_encoded.shape[1])

# CNN and RNN require 3D input:
# (samples, timesteps, channels/features)
X_train_dl = X_train_encoded.reshape(
    X_train_encoded.shape[0],
    X_train_encoded.shape[1],
    1
)

X_test_dl = X_test_encoded.reshape(
    X_test_encoded.shape[0],
    X_test_encoded.shape[1],
    1
)

input_shape = (X_train_dl.shape[1], 1)

print("CNN/RNN Input Shape:", input_shape)

# ----------------------------------
# STEP 5: CNN MODEL
# ----------------------------------

cnn_model = Sequential([
    Input(shape=input_shape),
    Conv1D(32, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation="relu"),
    Dense(
        num_classes,
        activation="softmax"
    )
])

cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN MODEL SUMMARY")
cnn_model.summary()

print("\nTraining CNN: Batch=30, Epochs=10, Adam")

cnn_history = cnn_model.fit(
    X_train_dl,
    y_train,
    batch_size=30,
    epochs=10,
    validation_split=0.20,
    verbose=1
)

# ----------------------------------
# STEP 6: RNN MODEL
# ----------------------------------

rnn_model = Sequential([
    Input(shape=input_shape),
    SimpleRNN(32, activation="tanh"),
    Dense(32, activation="relu"),
    Dense(
        num_classes,
        activation="softmax"
    )
])

rnn_model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nRNN MODEL SUMMARY")
rnn_model.summary()

print("\nTraining RNN: Batch=40, Epochs=20, RMSprop")

rnn_history = rnn_model.fit(
    X_train_dl,
    y_train,
    batch_size=40,
    epochs=20,
    validation_split=0.20,
    verbose=1
)

# ----------------------------------
# STEP 7: PREDICTIONS AND EVALUATION
# ----------------------------------

cnn_predictions = np.argmax(
    cnn_model.predict(X_test_dl, verbose=0),
    axis=1
)

rnn_predictions = np.argmax(
    rnn_model.predict(X_test_dl, verbose=0),
    axis=1
)

cnn_accuracy = accuracy_score(
    y_test, cnn_predictions
)

rnn_accuracy = accuracy_score(
    y_test, rnn_predictions
)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"\nCNN Accuracy: {cnn_accuracy:.4f}")
print(f"RNN Accuracy: {rnn_accuracy:.4f}")

labels = np.arange(num_classes)
class_names = [str(c) for c in label_encoder.classes_]

print("\nCNN Confusion Matrix:")
print(confusion_matrix(y_test, cnn_predictions, labels=labels))

print("\nCNN Classification Report:")
print(classification_report(
    y_test,
    cnn_predictions,
    labels=labels,
    target_names=class_names,
    zero_division=0
))

print("\nRNN Confusion Matrix:")
print(confusion_matrix(y_test, rnn_predictions, labels=labels))

print("\nRNN Classification Report:")
print(classification_report(
    y_test,
    rnn_predictions,
    labels=labels,
    target_names=class_names,
    zero_division=0
))

# ----------------------------------
# STEP 8: VISUALIZATIONS
# ----------------------------------

# Graph 1: Model Accuracy Comparison
plt.figure(figsize=(8, 5))
plt.bar(
    ["CNN", "RNN"],
    [cnn_accuracy * 100, rnn_accuracy * 100]
)
plt.ylabel("Test Accuracy (%)")
plt.title("CNN vs RNN Test Accuracy")
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "task1_accuracy_comparison.png",
    dpi=180
)
plt.close()

# Graph 2: CNN Training History
plt.figure(figsize=(9, 5))
plt.plot(cnn_history.history["accuracy"], label="Train")
plt.plot(cnn_history.history["val_accuracy"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("CNN Training Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "task1_cnn_training.png",
    dpi=180
)
plt.close()

# Graph 3: RNN Training History
plt.figure(figsize=(9, 5))
plt.plot(rnn_history.history["accuracy"], label="Train")
plt.plot(rnn_history.history["val_accuracy"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("RNN Training Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "task1_rnn_training.png",
    dpi=180
)
plt.close()

# Graph 4 and 5: Confusion Matrices
for name, predictions in [
    ("cnn", cnn_predictions),
    ("rnn", rnn_predictions)
]:
    cm = confusion_matrix(
        y_test, predictions, labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    display.plot(cmap="Blues")
    plt.title(f"{name.upper()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"task1_{name}_confusion.png",
        dpi=180
    )
    plt.close()

# ----------------------------------
# STEP 9: SAVE MODELS AND RESULTS
# ----------------------------------

cnn_model.save(MODEL_DIR / "cnn_model.keras")
rnn_model.save(MODEL_DIR / "rnn_model.keras")

results = pd.DataFrame({
    "Model": ["CNN", "RNN"],
    "Batch_Size": [30, 40],
    "Epochs": [10, 20],
    "Optimizer": ["Adam", "RMSprop"],
    "Test_Accuracy": [cnn_accuracy, rnn_accuracy]
})

results.to_csv(
    OUTPUT_DIR / "task1_model_comparison.csv",
    index=False
)

predictions_df = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_test),
    "CNN_Prediction": label_encoder.inverse_transform(cnn_predictions),
    "RNN_Prediction": label_encoder.inverse_transform(rnn_predictions)
})

predictions_df.to_csv(
    OUTPUT_DIR / "task1_predictions.csv",
    index=False
)

# ----------------------------------
# FINAL RESULT
# ----------------------------------

if cnn_accuracy > rnn_accuracy:
    winner = "CNN"
elif rnn_accuracy > cnn_accuracy:
    winner = "RNN"
else:
    winner = "Both models achieved equal accuracy"

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results.to_string(index=False))
print("\nBest Performing Model:", winner)
print("\nAll models, predictions and graphs saved.")
print("Task 1 Completed Successfully!")
