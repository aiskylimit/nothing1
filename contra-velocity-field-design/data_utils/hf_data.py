import os
import posixpath

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


HF_DATA_PREFIX = "hf://"
DEFAULT_DATA_REVISION = "main"


def _parse_hf_path(path: str):
    if not path.startswith(HF_DATA_PREFIX):
        return None

    parts = path[len(HF_DATA_PREFIX):].strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid Hugging Face dataset path: {path}")

    repo_id = "/".join(parts[:2])
    repo_path = "/".join(parts[2:])
    return repo_id, repo_path


def resolve_data_file(data_path: str, filename: str) -> str:
    parsed = _parse_hf_path(data_path)
    if parsed is None:
        return os.path.join(data_path, filename)

    repo_id, repo_path = parsed
    remote_filename = posixpath.join(repo_path, filename) if repo_path else filename
    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=remote_filename,
        revision=os.environ.get("CONTRA_DATA_REVISION", DEFAULT_DATA_REVISION),
    )


def resolve_data_uri(path: str) -> str:
    parsed = _parse_hf_path(path)
    if parsed is None:
        return path

    repo_id, remote_filename = parsed
    if not remote_filename:
        raise ValueError(f"Hugging Face path must identify a file: {path}")
    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=remote_filename,
        revision=os.environ.get("CONTRA_DATA_REVISION", DEFAULT_DATA_REVISION),
        token=False,
    )


def find_data_file(data_path: str, filename: str):
    try:
        path = resolve_data_file(data_path, filename)
    except EntryNotFoundError:
        return None
    return path if os.path.isfile(path) else None
