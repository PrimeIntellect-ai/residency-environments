"""Centralized data-tree paths and sandbox-relative I/O helpers.

``DataTree`` resolves paths under the packaged data root. ``Workspace`` validates
relative paths before passing sandbox I/O to the runtime.
"""

import pathlib

from pmpp_hard.errors import PathDisciplineError

# Runtime mount point used by the environment.
SANDBOX_ROOT = "/app"

REF_NAMES = (
    "reference_solution.cu",
    "reference_solution.py",
    "reference.py",
    "reference_triton.py",
)


class DataTree:
    """Provide path access for the packaged data tree or an alternate root."""

    def __init__(self, root: pathlib.Path | None = None) -> None:
        self.root = root if root is not None else pathlib.Path(__file__).parent / "data"

    def manifest(self, name: str) -> pathlib.Path:
        return self.root / name

    def sanity_manifest(self, name: str) -> pathlib.Path:
        return self.root / name

    def bundle(self, task_id: str) -> pathlib.Path:
        return self.root / "bundles" / task_id

    def sanity_dir(self, task_id: str) -> pathlib.Path:
        return self.root / "sanity" / task_id

    def reference_dir(self, task_id: str) -> pathlib.Path:
        return self.root / "reference" / task_id

    def find_reference(
        self, task_id: str, cuda_only: bool = False
    ) -> pathlib.Path | None:
        """Find the release reference source for a task."""
        pat = "reference_solution.cu" if cuda_only else "reference_solution.*"
        return next(self.reference_dir(task_id).glob(pat), None)

    def bundle_ids(self) -> list[str]:
        b = self.root / "bundles"
        if not b.is_dir():
            return []
        return sorted(
            p.name for p in b.iterdir() if p.is_dir() and p.name != "__pycache__"
        )

    def reference_ids(self) -> list[str]:
        r = self.root / "reference"
        if not r.is_dir():
            return []
        return sorted(p.name for p in r.iterdir() if p.is_dir())

    def sanity_ids(self) -> list[str]:
        s = self.root / "sanity"
        if not s.is_dir():
            return []
        return sorted(p.name for p in s.iterdir() if p.is_dir())


class Workspace:
    """Adapt runtime I/O to relative paths beneath a configured sandbox root."""

    def __init__(self, runtime, root: str = SANDBOX_ROOT) -> None:
        self.runtime = runtime
        self.root = root

    def abs(self, rel: str) -> str:
        """Resolve a non-empty relative path without parent-directory traversal."""
        if not rel or rel.startswith("/") or any(seg == ".." for seg in rel.split("/")):
            raise PathDisciplineError(f"sandbox path escapes workspace root: {rel!r}")
        return f"{self.root}/{rel}"

    async def write(self, rel: str, data: bytes) -> None:
        await self.runtime.write(self.abs(rel), data)

    async def read(self, rel: str) -> bytes:
        return await self.runtime.read(self.abs(rel))

    async def run(self, argv: list[str], env: dict[str, str] | None = None):
        return await self.runtime.run(argv, env or {})
