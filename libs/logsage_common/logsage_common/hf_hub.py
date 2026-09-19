"""
Hugging Face Hub integration.

- Reads HF_TOKEN from the environment and passes it explicitly, so gated or
  rate-limited model downloads authenticate instead of failing anonymously.
- Loads the sentence-transformers embedding model from the Hub.
- Publishes/fetches trained detector artifacts (the TensorFlow autoencoder)
  as a Hub model repo, so services can pull a versioned model instead of
  retraining on startup.
"""
from __future__ import annotations

import os


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, token=hf_token())


def push_detector(local_dir: str, repo_id: str, private: bool = True) -> str:
    """Uploads a saved detector directory to the Hub; returns the repo URL."""
    from huggingface_hub import HfApi

    token = hf_token()

    if token is None:
        raise RuntimeError("HF_TOKEN is not set; cannot push to the Hugging Face Hub")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")

    return f"https://huggingface.co/{repo_id}"


def pull_detector(repo_id: str, revision: str | None = None) -> str:
    """Downloads a detector repo snapshot; returns the local directory."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, repo_type="model", revision=revision, token=hf_token())