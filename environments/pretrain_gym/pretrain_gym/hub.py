"""Hugging Face Hub tools limited to data available on or before a cutoff date."""

import asyncio
import fnmatch
import glob
import json
import os
import shutil
import uuid
from collections.abc import Callable, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import verifiers.v1 as vf
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
from prime_sandboxes import AsyncSandboxClient
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import HubToolsetConfig
from .state import PretrainState
from .workspace import DockerBox, PrimeBox, open_box

CARD_CHARS = 20_000
LISTED_COMMITS = 100
LISTED_DOWNLOADS = 200


class HubError(Exception):
    """A request the Hub or the cutoff cannot satisfy; its message is returned to the agent."""


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return (
        isinstance(exc, HfHubHTTPError)
        and exc.response is not None
        and exc.response.status_code in {429, 500, 502, 503, 504}
    )


_retrying = retry(
    retry=retry_if_exception(_transient), stop=stop_after_attempt(6), wait=wait_exponential(min=2, max=60), reraise=True
)


async def _call(fn, *args, **kwargs):
    try:
        return await asyncio.to_thread(_retrying(fn), *args, **kwargs)
    except RepositoryNotFoundError as exc:
        raise HubError(f"repository not found or not accessible: {exc}") from exc
    except EntryNotFoundError as exc:
        raise HubError(f"file not found: {exc}") from exc
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code < 500:
            raise HubError(str(exc)) from exc
        raise


def _day(moment: datetime | None) -> date | None:
    return moment.date() if moment is not None else None


def _page(items: Iterable, keep: Callable[[Any], bool], offset: int, limit: int) -> tuple[list, bool]:
    """Items `offset` to `offset + limit` among those kept, and whether more follow."""
    page, seen = [], 0
    for item in items:
        if not keep(item):
            continue
        if seen >= offset + limit:
            return page, True
        if seen >= offset:
            page.append(item)
        seen += 1
    return page, False


RepoType = Literal["dataset", "model"]
DESTINATIONS = {"dataset": "data", "model": "models"}
# Tags and descriptions are left out: they show a repository as it is now, not as of the cutoff.
EXPAND = {
    "dataset": ["createdAt", "downloads", "likes", "gated"],
    "model": ["createdAt", "downloads", "likes", "gated", "pipeline_tag", "library_name"],
}


class HubToolset(vf.Toolset[HubToolsetConfig, PretrainState]):
    TOOL_PREFIX = "hf"

    def __init__(self, config: HubToolsetConfig) -> None:
        super().__init__(config)
        self.token = os.environ.get(config.hub.token_env)
        self.api = HfApi(token=self.token)
        self.cutoff = config.hub.cutoff_date
        self.host_dir = config.host_dir.expanduser()
        self._checked: set[tuple[str, str]] = set()
        self._commits: dict[tuple[str, str, str | None], list] = {}
        self._trees: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        self._downloaded = 0
        self._client: AsyncSandboxClient | None = None

    async def setup(self) -> None:
        self._exit_stack.push_async_callback(self._close_client)

    async def _close_client(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _workspace(self) -> DockerBox | PrimeBox:
        if self.state.workspace_kind == "prime" and self._client is None:
            self._client = AsyncSandboxClient()
        return open_box(self.state.workspace_kind, self.state.workspace, self._client)

    def _before_cutoff(self, moment: datetime | None) -> bool:
        return moment is not None and _day(moment) <= self.cutoff

    async def _history(self, repo_type: RepoType, repo_id: str, branch: str | None) -> list:
        """Commits of `branch` (the default branch when None) made on or before the cutoff, newest first."""
        if (repo_type, repo_id) not in self._checked:
            info = await _call(self.api.repo_info, repo_id, repo_type=repo_type, expand=["createdAt"])
            if info.created_at is not None and _day(info.created_at) > self.cutoff:
                raise HubError(f"{repo_id} was created after {self.cutoff}")
            self._checked.add((repo_type, repo_id))
        if (repo_type, repo_id, branch) not in self._commits:
            commits = await _call(self.api.list_repo_commits, repo_id, repo_type=repo_type, revision=branch)
            self._commits[repo_type, repo_id, branch] = [c for c in commits if self._before_cutoff(c.created_at)]
        return self._commits[repo_type, repo_id, branch]

    async def _revision(
        self, repo_type: RepoType, repo_id: str, revision: str | None, branch: str | None
    ) -> tuple[str, date]:
        """The newest commit of `branch` on or before the cutoff, or the one whose hash starts with `revision`."""
        commits = await self._history(repo_type, repo_id, branch)
        if revision is not None:
            commits = [commit for commit in commits if commit.commit_id.startswith(revision)]
            if len(commits) != 1:
                raise HubError(f"{revision!r} names no single commit of {repo_id} made on or before {self.cutoff}")
        if not commits:
            raise HubError(f"{repo_id} has no commit on or before {self.cutoff}")
        return commits[0].commit_id, _day(commits[0].created_at)

    async def _tree(self, repo_type: RepoType, repo_id: str, sha: str) -> list[dict[str, Any]]:
        if (repo_type, repo_id, sha) not in self._trees:
            entries = await _call(
                lambda: list(self.api.list_repo_tree(repo_id, revision=sha, repo_type=repo_type, recursive=True))
            )
            self._trees[repo_type, repo_id, sha] = [
                dict(path=entry.path, size=entry.size) for entry in entries if getattr(entry, "size", None) is not None
            ]
        return self._trees[repo_type, repo_id, sha]

    @vf.tool
    async def search(
        self,
        repo_type: RepoType = "dataset",
        query: str = "",
        author: str | None = None,
        tags: list[str] | None = None,
        sort: Literal["downloads", "likes", "created_at", "last_modified", "trending_score"] = "downloads",
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """Search Hugging Face Hub datasets or models created on or before the data cutoff.

        `query` matches repository ids, `author` a user or organization, and every one of `tags`
        must be on the repository (Hub tags such as `language:en`, `task_categories:text-generation`,
        `size_categories:1B<n<10B`, or `text-classification`). Results are sorted by `sort`,
        highest or newest first. Returns each repository's id, creation date, download and like
        counts, and gating, plus a model's task and library.
        """
        lister = self.api.list_datasets if repo_type == "dataset" else self.api.list_models
        found = lister(search=query or None, author=author, filter=tags, sort=sort, expand=EXPAND[repo_type])
        try:
            page, more = await _call(_page, found, lambda info: self._before_cutoff(info.created_at), offset, limit)
        except HubError as exc:
            return dict(error=str(exc))
        results = []
        for info in page:
            result = dict(
                id=info.id,
                created=str(_day(info.created_at)),
                downloads=info.downloads,
                likes=info.likes,
                gated=info.gated,
            )
            if repo_type == "model":
                result.update(task=info.pipeline_tag, library=info.library_name)
            results.append(result)
        return dict(results=results, more=more)

    @vf.tool
    async def repo_info(
        self,
        repo_id: str,
        repo_type: RepoType = "dataset",
        branch: str | None = None,
        revision: str | None = None,
        path_prefix: str = "",
        offset: int = 0,
        limit: int = 100,
        card_offset: int = 0,
    ) -> dict:
        """Describe a Hugging Face Hub dataset or model at a commit made on or before the data cutoff.

        The commit is the newest one on `branch` (default: the default branch; any ref name works,
        e.g. `refs/convert/parquet`), or the one on it whose hash starts with `revision`. Returns
        the commit and its date; the repository's branches, tags, and converted refs; the newest
        commits on the branch up to the cutoff; the card (README) from `card_offset`; and `limit`
        files from `offset` among those whose path starts with `path_prefix`, each with its size
        in bytes, plus their count and total size.
        """
        try:
            sha, day = await self._revision(repo_type, repo_id, revision, branch)
            commits = await self._history(repo_type, repo_id, branch)
            tree = await self._tree(repo_type, repo_id, sha)
            refs = await _call(self.api.list_repo_refs, repo_id, repo_type=repo_type)
            try:
                card_path = await _call(
                    hf_hub_download, repo_id, "README.md", repo_type=repo_type, revision=sha, token=self.token
                )
                card = Path(card_path).read_text(encoding="utf-8", errors="replace")
            except HubError:
                card = ""
        except HubError as exc:
            return dict(error=str(exc))
        files = [entry for entry in tree if entry["path"].startswith(path_prefix)]
        return dict(
            repo_id=repo_id,
            revision=sha,
            revision_date=str(day),
            refs=[ref.name for ref in (*refs.branches, *refs.tags)] + [ref.ref for ref in refs.converts],
            commits=[
                dict(id=commit.commit_id, date=str(_day(commit.created_at)), title=commit.title)
                for commit in commits[:LISTED_COMMITS]
            ],
            commit_count=len(commits),
            card=card[card_offset : card_offset + CARD_CHARS],
            card_chars=len(card),
            files=files[offset : offset + limit],
            file_count=len(files),
            total_bytes=sum(entry["size"] for entry in files),
        )

    @vf.tool
    async def download(
        self,
        repo_id: str,
        files: list[str],
        repo_type: RepoType = "dataset",
        branch: str | None = None,
        revision: str | None = None,
    ) -> dict:
        """Download files of a Hugging Face Hub dataset into `data/<repo_id>/`, or of a model into
        `models/<repo_id>/`, in the workspace.

        `files` are paths or glob patterns matched against the repository's file paths at the
        commit `repo_info` would describe for the same `branch` and `revision`. Files keep their
        repository paths. Returns the commit, the number of files and bytes written, and their paths.
        """
        try:
            sha, _ = await self._revision(repo_type, repo_id, revision, branch)
            tree = await self._tree(repo_type, repo_id, sha)
        except HubError as exc:
            return dict(error=str(exc))
        matched = [entry for entry in tree if any(fnmatch.fnmatchcase(entry["path"], pattern) for pattern in files)]
        if not matched:
            return dict(error=f"no file of {repo_id} at {sha} matches {files}")
        size = sum(entry["size"] for entry in matched)
        if self.config.hub.max_download_gb is not None:
            left = int(self.config.hub.max_download_gb * 1e9) - self._downloaded
            if size > left:
                return dict(error=f"{size:,} bytes would exceed this episode's download limit ({left:,} bytes left)")
        directory = f"{DESTINATIONS[repo_type]}/{repo_id}"
        staging = self.host_dir / "downloads" / uuid.uuid4().hex
        try:
            await _call(
                snapshot_download,
                repo_id,
                repo_type=repo_type,
                revision=sha,
                allow_patterns=[glob.escape(entry["path"]) for entry in matched],
                local_dir=staging,
                token=self.token,
            )
            shutil.rmtree(staging / ".cache", ignore_errors=True)
            await self._workspace().put_tree(staging, directory)
        except HubError as exc:
            return dict(error=str(exc))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        self._downloaded += size
        record = dict(repo_type=repo_type, repo_id=repo_id, revision=sha, files=len(matched), bytes=size)
        log = self.host_dir / "rollouts" / self.state.rollout / "downloads.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as file:
            file.write(json.dumps(record) + "\n")
        return dict(
            revision=sha,
            directory=directory,
            file_count=len(matched),
            bytes=size,
            files=[f"{directory}/{entry['path']}" for entry in matched[:LISTED_DOWNLOADS]],
        )

    @vf.tool
    async def search_papers(
        self,
        query: str | None = None,
        date: str | None = None,
        week: str | None = None,
        month: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """Find Hugging Face Papers published on or before the data cutoff.

        Searches by `query`, or lists the daily papers of a `date` (`YYYY-MM-DD`), `week`
        (`YYYY-Www`), or `month` (`YYYY-MM`). Returns each paper's id, title, publication date,
        and abstract.
        """
        if query:
            papers = self.api.list_papers(query=query, limit=4 * (offset + limit))
        elif date or week or month:
            papers = self.api.list_daily_papers(date=date, week=week, month=month)
        else:
            return dict(error="give a query, date, week, or month")
        try:
            page, more = await _call(
                _page, papers, lambda paper: self._before_cutoff(paper.published_at), offset, limit
            )
        except HubError as exc:
            return dict(error=str(exc))
        return dict(
            papers=[
                dict(id=paper.id, title=paper.title, published=str(_day(paper.published_at)), abstract=paper.summary)
                for paper in page
            ],
            more=more,
        )

    @vf.tool
    async def read_paper(self, paper_id: str, offset: int = 0, length: int = 40_000) -> dict:
        """Read a paper published on or before the data cutoff as Markdown.

        Returns `length` characters starting at `offset` and the paper's total length.
        """
        try:
            info = await _call(self.api.paper_info, paper_id)
            if not self._before_cutoff(info.published_at):
                return dict(error=f"paper {paper_id} was not published on or before {self.cutoff}")
            text = await _call(self.api.read_paper, paper_id)
        except HubError as exc:
            return dict(error=str(exc))
        return dict(id=paper_id, title=info.title, text=text[offset : offset + length], total_chars=len(text))


if __name__ == "__main__":
    HubToolset.run()
