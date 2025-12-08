from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline
)

BASE_PATH = Path(__file__).resolve().parents[2] / "models"


def load_tokenizer(model_name: str):
    model_path = BASE_PATH / model_name
    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)


def load_sequence_model(model_name: str):
    model_path = BASE_PATH / model_name
    return AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)


def load_embedding_model(model_name: str):
    model_path = BASE_PATH / model_name
    return AutoModel.from_pretrained(str(model_path), local_files_only=True)


def build_classification_pipeline(model_name: str):
    model_path = BASE_PATH / model_name
    tokenizer = load_tokenizer(model_name)
    model = load_sequence_model(model_name)
    return pipeline(
        "text-classification",
        tokenizer=tokenizer,
        model=model,
        top_k=None,
        truncation=True
    )
