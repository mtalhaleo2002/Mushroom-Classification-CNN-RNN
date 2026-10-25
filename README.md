
# Mushroom Classification Using CNN and RNN

### DigiSkills 3.0 — Artificial Intelligence Using Python (AIP301)

**Hands-on Exercise No. 04 | Batch 04**

A Python-based Deep Learning and Natural Language Processing project developed as part of the DigiSkills 3.0 Artificial Intelligence Using Python course.

The project focuses on mushroom classification using Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), followed by text tokenization using Hugging Face Transformers.

---

## Project Overview

This exercise consists of two main tasks:

1. Mushroom classification using CNN and RNN models.
2. Text tokenization using Hugging Face Transformers.

The project also includes automatic generation of code images, execution outputs, visualizations, and a Microsoft Word report.

## Technologies Used

- Python 3
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Hugging Face Transformers
- Pillow
- python-docx
- VS Code
- Linux Ubuntu

---

## Project Structure

The tree below shows the files intended to be committed to GitHub. Generated files,
local datasets, and the virtual environment are intentionally ignored by Git.

```text
ex4/
├── data/
│   └── .gitkeep
├── check_datasets.py
├── generate_report.py
├── task1_cnn.py
├── task1_comparison.py
├── task1_deep_learning.py
├── task1_rnn.py
├── task2_tokenization.py
├── requirements.txt
├── .gitignore
└── README.md
```

The following folders are created or used locally and are excluded from GitHub:

```text
data/       Downloaded datasets
models/     Trained .keras models
outputs/    Generated CSV files and plots
screenshots/ Code and output images generated for the report
reports/    Generated Word report
venv/       Local Python virtual environment
```

GitHub shows committed files, not every file present on your computer. With the
current ignore rules, the dataset files and generated folders above will not
appear in the repository; `data/.gitkeep` keeps the empty `data/` folder visible.
If you want to publish any generated results, screenshots, or the Word report,
remove the matching ignore rule from `.gitignore` before adding those files.

---

## Task 1 — CNN and RNN Classification

### Objective

Build, train, evaluate, and compare CNN and RNN models using the Mushroom Classification dataset.

### Dataset

The dataset contains categorical mushroom characteristics, including:

- Cap shape and surface
- Odor
- Gill characteristics
- Stalk characteristics
- Habitat
- Mushroom classification label

Dataset source:

https://lms.digiskills.pk/Courses/AIP301/Downloads/mushrooms.csv

### Data Preprocessing

The preprocessing workflow includes:

1. Load the Mushroom dataset with Pandas.
2. Identify independent features and the target column.
3. Handle missing categorical values.
4. Encode categorical features using One-Hot Encoding.
5. Encode target labels.
6. Split the dataset into training and testing sets.
7. Reshape encoded data for CNN and RNN input.

### Train-Test Split

| Dataset | Percentage |
|---------|------------|
| Training | 70% |
| Testing | 30% |

### Model Configurations

| Model | Batch Size | Epochs | Optimizer |
|-------|------------|--------|-----------|
| CNN | 30 | 10 | Adam |
| RNN | 40 | 20 | RMSprop |

### CNN Architecture

The CNN model uses:

- Input layer
- Conv1D layer
- MaxPooling1D layer
- Flatten layer
- Dense hidden layer
- Softmax output layer

### RNN Architecture

The RNN model uses:

- Input layer
- SimpleRNN layer
- Dense hidden layer
- Softmax output layer

The models are implemented using TensorFlow/Keras.

### Model Evaluation

Both models are evaluated using:

- Test accuracy
- Predictions
- Confusion matrix
- Precision
- Recall
- F1-score
- Classification report
- Training accuracy visualizations

The model with the higher observed test accuracy is identified in the generated results.

**Important:** The dataset contains tabular categorical features rather than a natural sequential signal. CNN and RNN are used as experimental architectures to meet the assignment requirements.

---

## Task 2 — Generative AI Basics

### Objective

Load text from a JSON file and perform text tokenization using the Hugging Face Transformers library.

### Dataset

https://lms.digiskills.pk/Courses/AIP301/Downloads/exercise_text.json

### Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
```

### Tokenization Workflow

1. Load JSON data.
2. Extract text values.
3. Initialize the Hugging Face tokenizer.
4. Convert text into tokens.
5. Generate token IDs.
6. Generate attention masks.
7. Save the tokenization results to CSV.

---

## Installation Guide

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd ex4
```

Replace `<your-repository-url>` with your actual GitHub repository URL.

### 2. Create Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

On Linux Ubuntu:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a Python version compatible with the TensorFlow version specified in `requirements.txt`.

### 5. Download Datasets

```bash
mkdir -p data

wget -O data/mushrooms.csv \
"https://lms.digiskills.pk/Courses/AIP301/Downloads/mushrooms.csv"

wget -O data/exercise_text.json \
"https://lms.digiskills.pk/Courses/AIP301/Downloads/exercise_text.json"
```

If LMS authentication prevents direct downloads, download the files manually through your LMS account and place them inside `data/`.

---

## Running the Project

### Verify Datasets

```bash
python check_datasets.py
```

### Train CNN and RNN Models

```bash
python task1_deep_learning.py
```

This script trains both models, evaluates their predictions, and generates the comparison results and visualizations.

### Run Hugging Face Tokenization

```bash
python task2_tokenization.py
```

### Generate Word Report

```bash
python generate_report.py
```

The generated report is saved as:

```text
reports/AIP301_Exercise_04_Solution.docx
```

### Open Word Report on Ubuntu

```bash
libreoffice --writer reports/AIP301_Exercise_04_Solution.docx
```

---

## Generated Visualizations

The project generates:

1. CNN vs RNN test accuracy comparison
2. CNN training accuracy graph
3. RNN training accuracy graph
4. CNN confusion matrix
5. RNN confusion matrix

All visualizations are generated using Matplotlib.

---

## Automated Report Generation

The `generate_report.py` script automatically creates a Word document containing:

- Assignment title and introduction
- Task 1 Python source code rendered as images
- Saved CNN and RNN performance results
- Model evaluation graphs
- Task 2 Python source code rendered as images
- Tokenization execution output
- Conclusion

Code and output images are generated programmatically using Pillow. They are not literal VS Code desktop screenshots.

The report should be reviewed before submission to ensure it meets the institution's screenshot and formatting requirements.

---

## Project Notes

- TensorFlow/Keras is used for Deep Learning.
- PyTorch is not used.
- The same 70/30 split is used to evaluate both models.
- Categorical features are converted into numerical representations.
- Generated outputs depend on dataset contents, package versions, and model training.
- The higher-accuracy model should be selected using actual execution results, not assumed in advance.
- The Hugging Face tokenizer may require an internet connection on its first run.

---

## Author

**Mohammad Talha Khursheed**

DigiSkills 3.0 — Artificial Intelligence Using Python

Hands-on Exercise No. 04

---

## License and Usage

Created for educational and practical learning purposes.

Dataset files are obtained separately from the DigiSkills LMS and are not distributed in this repository.
