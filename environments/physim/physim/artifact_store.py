"""Bounded workspace archives that preserve Linux filenames on every host.

Submitted code is data here: only regular files/directories enter a canonical
tar archive. Extraction happens in Linux, never on the host filesystem.
"""

import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

FORMAT = "physim-workspace-tar-v1"
MARKER = ".physim-artifact.json"
ARCHIVE = "workspace.tar"


class SandboxError(RuntimeError):
    pass


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _name(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise SandboxError("unsafe artifact member path")
    return path.as_posix()


class _HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size):
        value = self.stream.read(size)
        self.hash.update(value)
        return value


def archive_workspace(stream, target, limits):
    """Validate a tar stream and atomically publish a canonical archive/manifest."""
    target = Path(target)
    if target.exists():
        raise SandboxError("artifact snapshot directory already exists")
    target.mkdir(parents=True)
    pending = target / (ARCHIVE + ".partial")
    files, seen, total = [], {}, 0
    try:
        with tarfile.open(fileobj=stream, mode="r|*") as source, tarfile.open(pending, "w") as output:
            for member in source:
                name = _name(member.name)
                if not (member.isdir() or member.isfile()) or member.issym() or member.islnk():
                    raise SandboxError("artifact may contain only regular files and directories")
                if name == "." and member.isdir():
                    continue
                if name == "." or name in seen:
                    raise SandboxError("duplicate or invalid artifact member path")
                path = PurePosixPath(name)
                if any(seen.get(p.as_posix()) == "file" for p in path.parents) or (
                    member.isfile() and any(p.startswith(name + "/") for p in seen)
                ):
                    raise SandboxError("conflicting artifact file and directory paths")
                if len(seen) >= 10000:
                    raise SandboxError("artifact member count exceeds cap")
                seen[name] = "directory" if member.isdir() else "file"
                clean = tarfile.TarInfo(name)
                clean.mode = 0o755 if member.isdir() else 0o644
                if member.isdir():
                    clean.type = tarfile.DIRTYPE
                    output.addfile(clean)
                    continue
                total += member.size
                if (
                    member.size < 0
                    or member.size > limits.file_mib * 1024 * 1024
                    or total > limits.artifact_mib * 1024 * 1024
                    or len(files) >= 2000
                ):
                    raise SandboxError("artifact exceeds file or total-size limit")
                clean.size = member.size
                with source.extractfile(member) as handle:
                    hashed = _HashedReader(handle)
                    output.addfile(clean, hashed)
                files.append(dict(path=name, bytes=member.size, sha256=hashed.hash.hexdigest()))
        if "predictor.py" not in {f["path"] for f in files}:
            raise SandboxError("write /workspace/predictor.py before submitting")
        archive = target / ARCHIVE
        pending.replace(archive)
        manifest = dict(format=FORMAT, archive=ARCHIVE, archive_sha256=digest(archive), files=files, bytes=total)
        # Written last: a failed/partial export is never a valid artifact.
        (target / MARKER).write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    except (tarfile.TarError, EOFError) as exc:
        raise SandboxError("invalid or incomplete artifact archive") from exc
    finally:
        pending.unlink(missing_ok=True)


def archive_info(root):
    """Return a verified archive manifest, or None for a legacy directory."""
    root = Path(root)
    marker = root / MARKER
    if not marker.exists():
        return None
    if root.is_symlink() or marker.is_symlink() or not marker.is_file():
        raise SandboxError("artifact symlinks are forbidden")
    info = json.loads(marker.read_text())
    if info.get("format") != FORMAT or info.get("archive") != ARCHIVE:
        raise SandboxError("unsupported workspace archive format")
    archive = root / ARCHIVE
    if archive.is_symlink() or not archive.is_file() or digest(archive) != info["archive_sha256"]:
        raise SandboxError("artifact archive hash mismatch")
    return info


def snapshot_artifact(source, target, limits):
    """Freeze either an archived workspace or an existing directory artifact."""
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise SandboxError("public input directory is missing or a symlink")
    info = archive_info(source)
    if info is not None:
        with (source / ARCHIVE).open("rb") as stream:
            result = archive_workspace(stream, target, limits)
        if result["files"] != info["files"] or result["bytes"] != info["bytes"]:
            (Path(target) / MARKER).unlink()
            raise SandboxError("artifact file manifest mismatch")
        return result
    # Legacy files are packed without ever recreating their names on the host.
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode="w") as archive:
            total, count = 0, 0
            for path in sorted(source.rglob("*")):
                if path.is_symlink():
                    raise SandboxError("artifact symlinks are forbidden")
                entry = tarfile.TarInfo(path.relative_to(source).as_posix())
                if path.is_dir():
                    entry.type = tarfile.DIRTYPE
                    archive.addfile(entry)
                elif path.is_file():
                    entry.size = path.stat().st_size
                    total += entry.size
                    count += 1
                    if (
                        entry.size > limits.file_mib * 1024 * 1024
                        or total > limits.artifact_mib * 1024 * 1024
                        or count > 2000
                    ):
                        raise SandboxError("artifact exceeds file or total-size limit")
                    with path.open("rb") as handle:
                        archive.addfile(entry, handle)
                else:
                    raise SandboxError("artifact may contain only regular files and directories")
        stream.seek(0)
        return archive_workspace(stream, target, limits)


def read_artifact_files(root, expected):
    """Recover only manifest-listed bytes, preserving exact case for checkpoints."""
    root = Path(root)
    info = archive_info(root)
    if info is None:
        raise SandboxError("checkpoint is not an archived artifact")
    rows = {row["path"]: row for row in info["files"]}
    with tarfile.open(root / ARCHIVE, "r:") as archive:
        members = {member.name: member for member in archive}
        result = []
        for row in expected:
            name = row["path"]
            if name != _name(name) or any(p.startswith(".vf-") for p in PurePosixPath(name).parts):
                raise SandboxError("invalid checkpoint file path")
            if rows.get(name, {}).get("sha256") != row["sha256"]:
                raise SandboxError("checkpoint file hash mismatch")
            member = members.get(name)
            if member is None or not member.isfile() or member.size != rows[name]["bytes"]:
                raise SandboxError("checkpoint file is not a regular file")
            with archive.extractfile(member) as handle:
                data = handle.read()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise SandboxError("checkpoint file hash mismatch")
            result.append((name, data))
    return result
