from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent.parent

MICROSERVICES = {
    "text-service": {
        "emotion_model": "j-hartmann/emotion-english-distilroberta-base"
    },
#    "translation-service": {
#        "translation_model": "Qwen/Qwen2.5-1.5B-Instruct"
#    },
#    "embedding-service": {
 #       "embedding_model": "BAAI/bge-large-en-v1.5"
#    }
}


def download_for_microservice(micro_name: str, models: dict):
    micro_path = BASE_DIR / micro_name
    models_dir = micro_path / "models"
    models_dir.mkdir(exist_ok=True)

    for local_name, repo_id in models.items():
        target = models_dir / local_name

        if target.exists() and any(target.iterdir()):
            print(f"[SKIP] {micro_name} -> {local_name} already exists.")
            continue

        print(f"[DOWNLOAD] {micro_name} -> {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False
        )
        print(f"[OK] {micro_name}: saved {local_name}")


def main():
    for micro_name, models in MICROSERVICES.items():
        print(f"\n=== Processing {micro_name} ===")
        download_for_microservice(micro_name, models)


if __name__ == "__main__":
    main()
