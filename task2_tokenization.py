
# DigiSkills 3.0 - AIP301
# Hands-on Exercise No. 04
# Task 2: Generative AI Basics - Text Tokenization

from pathlib import Path
import json
import pandas as pd
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("TASK 2: GENERATIVE AI BASICS - TEXT TOKENIZATION")
print("=" * 65)

# Step 1: Load JSON dataset
json_path = DATA_DIR / "exercise_text.json"

with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("\n1. JSON DATASET LOADED SUCCESSFULLY")
print("Root Data Type:", type(data).__name__)

# Step 2: Extract text from JSON
def extract_text(obj):
    texts = []

    if isinstance(obj, str):
        if obj.strip():
            texts.append(obj)

    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text(item))

    elif isinstance(obj, dict):
        for key, value in obj.items():
            texts.extend(extract_text(value))

    return texts


texts = extract_text(data)

if not texts:
    raise ValueError("No text found in JSON dataset!")

print("\n2. TOTAL TEXT ITEMS:", len(texts))

# Step 3: Load Hugging Face tokenizer
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("\n3. TOKENIZER INFORMATION")
print("Tokenizer:", model_name)
print("Vocabulary Size:", tokenizer.vocab_size)

# Step 4: Tokenize text
results = []

for index, text in enumerate(texts, start=1):

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=128
    )

    tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"]
    )

    print("\n" + "-" * 65)
    print(f"TEXT ITEM {index}")
    print("-" * 65)

    print("Original Text:", text)
    print("Tokens:", tokens)
    print("Token IDs:", encoded["input_ids"])
    print("Attention Mask:", encoded["attention_mask"])
    print("Total Tokens:", len(tokens))

    results.append({
        "Text_ID": index,
        "Original_Text": text,
        "Tokens": json.dumps(tokens),
        "Token_IDs": json.dumps(encoded["input_ids"]),
        "Attention_Mask": json.dumps(
            encoded["attention_mask"]
        ),
        "Total_Tokens": len(tokens)
    })

# Step 5: Save results
result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_DIR / "task2_tokenization_results.csv",
    index=False
)

print("\n" + "=" * 65)
print("TASK 2 COMPLETED SUCCESSFULLY")
print("=" * 65)
print("Output: outputs/task2_tokenization_results.csv")
