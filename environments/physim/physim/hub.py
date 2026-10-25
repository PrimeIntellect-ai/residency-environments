"""Immutable Hugging Face resolution into a verified, offline-capable cache.

No remote code is imported. Every request is pinned to a full commit. Downloads
are streamed with byte caps, then atomically promoted only after verification.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .bundles import (
    MAX_JSON,
    Bundle,
    BundleError,
    canonical,
    digest,
    read_json,
    read_manifest,
    relative_path,
    safe_path,
)


def _arguments(repo, revision, cache):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo):
        raise BundleError("expected an owner/repository Hugging Face dataset ID")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise BundleError("revision must be a full lowercase 40-character commit; branches and tags are mutable")
    return Path(cache or os.environ.get("PHYSIM_CACHE", Path.home() / ".cache/physim")).expanduser().resolve()


def _download(repo, revision, name, destination, maximum, expected_bytes=None):
    # Use Hub's URL and HTTP client (redirects/auth are handled by that library),
    # but stream directly so even an incorrect remote manifest cannot cause an
    # unbounded download. Public reference datasets do not require a token.
    try:
        from huggingface_hub import get_session, hf_hub_url
    except ImportError as exc:
        raise BundleError("install physim[hub] to fetch Hugging Face data") from exc
    url = hf_hub_url(repo, name, repo_type="dataset", revision=revision)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with get_session().stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        length = response.headers.get("content-length")
        if length and int(length) > maximum:
            raise BundleError("remote response exceeds declared byte cap")
        count = 0
        with destination.open("xb") as stream:
            for block in response.iter_bytes(chunk_size=64 * 1024):
                count += len(block)
                if count > maximum:
                    raise BundleError("remote response exceeds declared byte cap")
                stream.write(block)
        if expected_bytes is not None and count != expected_bytes:
            raise BundleError("download size differs from manifest")


def fetch_bundle(*, repo, revision, path, profile="evaluation", cache=None, offline=False):
    root = _arguments(repo, revision, cache)
    relative_path(path)
    if profile not in ("simulation", "evaluation"):
        raise BundleError("unknown download profile")
    key = hashlib.sha256(canonical(dict(repo=repo, revision=revision, path=path))).hexdigest()
    target = root / "bundles" / key / profile
    if target.exists():
        Bundle(target, profile=profile)
        return target
    if offline:
        raise BundleError("requested revision and profile are not in the verified offline cache")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".fetch-", dir=target.parent))
    try:
        _download(repo, revision, path + "/manifest.json", temp / "manifest.json", MAX_JSON)
        manifest = read_manifest(temp / "manifest.json")
        for record in manifest["files"]:
            if profile in record["profiles"]:
                destination = safe_path(temp, record["path"])
                _download(repo, revision, path + "/" + record["path"], destination, record["bytes"], record["bytes"])
                if digest(destination) != record["sha256"]:
                    raise BundleError("download hash differs from manifest")
        Bundle(temp, profile=profile)
        (temp / "download.json").write_text(
            json.dumps(dict(repo=repo, revision=revision, path=path, profile=profile)) + "\n"
        )
        try:
            temp.rename(target)
        except OSError:
            if not target.exists():
                raise
            Bundle(target, profile=profile)  # Another concurrent verified download won.
        return target
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def fetch_catalog(*, repo, revision, cache=None, offline=False):
    root = _arguments(repo, revision, cache)
    key = hashlib.sha256(canonical(dict(repo=repo, revision=revision))).hexdigest()
    directory = root / "catalogs" / key
    target = directory / "catalog.jsonl"
    checksum = directory / "sha256.json"
    if not target.exists():
        if offline:
            raise BundleError("catalog revision is not available offline")
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".catalog-", dir=directory) as tmp:
            staged = Path(tmp) / "catalog.jsonl"
            _download(repo, revision, "catalog.jsonl", staged, MAX_JSON)
            _catalog_rows(staged)
            staged.replace(target)
            checksum.write_text(json.dumps(dict(sha256=digest(target))) + "\n")
    if target.is_symlink() or not checksum.is_file() or digest(target) != read_json(checksum)["sha256"]:
        raise BundleError("cached catalog hash mismatch")
    return dict(repo=repo, revision=revision, worlds=_catalog_rows(target))


def _catalog_rows(path):
    if path.stat().st_size > MAX_JSON:
        raise BundleError("catalog exceeds byte cap")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) > 1000 or any(type(row) is not dict or "bundle_path" not in row for row in rows):
        raise BundleError("invalid catalog")
    for row in rows:
        relative_path(row["bundle_path"])
    return rows
