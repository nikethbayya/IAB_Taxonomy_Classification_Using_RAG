import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# Step 1: Load the CSV file
file_path = "output_with_related_terms_llm_generated_cleaned_flattened_keywords_only.xlsx"
output_file = "keywords_with_embeddings_llm_generated.csv"
sheet_name = "Sheet1"
keyword_column = "keywords"

df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
df[keyword_column] = df[keyword_column].str.lower().str.strip()

# Step 2: Initialize models and tokenizers
models = {
    "roberta-base": {
        "name": "xlm-roberta-base",
        "tokenizer": None,
        "model": None,
    },
    "mbert": {
        "name": "bert-base-multilingual-cased",
        "tokenizer": None,
        "model": None,
    },
    "mdeberta-v3": {
        "name": "microsoft/mdeberta-v3-base",
        "tokenizer": None,
        "model": None,
    },
    "labse": {
        "name": "sentence-transformers/LaBSE",
        "tokenizer": None,
        "model": None,
    },
    "distilmBERT": {
        "name": "distilbert-base-multilingual-cased",
        "tokenizer": None,
        "model": None,
    },
    "mt5": {
        "name": "google/mt5-base",
        "tokenizer": None,
        "model": None,
    },
    "xlm": {
        "name": "xlm-mlm-tlm-xnli15-1024",
        "tokenizer": None,
        "model": None,
    },
}

# Load all models and tokenizers
for key, model_info in models.items():
    try:
        print(f"Loading model: {model_info['name']}")
        model_info["tokenizer"] = AutoTokenizer.from_pretrained(model_info["name"])
        model_info["model"] = AutoModel.from_pretrained(model_info["name"])
    except Exception as e:
        print(f"Error loading {key}: {e}")

# Step 3: Function to generate embeddings
def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def generate_embedding_mt5(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Step 4: Generate embeddings
for key, model_info in models.items():
    if model_info["model"] and model_info["tokenizer"]:
        try:
            column_name = f"{key}_embedding"
            if key == "mt5":
                df[column_name] = df[keyword_column].apply(
                    lambda x: generate_embedding_mt5(str(x), model_info["tokenizer"], model_info["model"])
                )
            else:
                df[column_name] = df[keyword_column].apply(
                    lambda x: generate_embedding(str(x), model_info["tokenizer"], model_info["model"])
                )
            print(f"Embeddings generated for {key}.")
        except Exception as e:
            print(f"Error generating embeddings for {key}: {e}")

# Step 5: Save results to CSV
print(df)
df.to_csv(output_file, index=False)
print(f"Embeddings saved to {output_file}")

