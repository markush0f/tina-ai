from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pathlib import Path

MODEL_PATH = Path("/home/markus/tina/tina-ai/text-service/models/emotion_model").resolve()




def analyze(text, classifier):
    result = classifier(text)[0]
    result = sorted(result, key=lambda x: x["score"], reverse=True)
    return result


if __name__ == "__main__":
    classifier = build_pipeline()
    texts = [
        "I feel exhausted and nothing makes sense anymore.",
        "Today was amazing, I feel grateful and full of joy.",
        "I am angry about what happened."
    ]
    for t in texts:
        scores = analyze(t, classifier)
        print(t)
        print(scores)
        print()
