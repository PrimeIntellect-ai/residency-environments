"""Supported local and immutable-revision bundle workflows."""

import argparse
import json
import sys
from pathlib import Path

from .bundles import Bundle, BundleError, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="physim")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "demo", "experiment", "grade"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", required=True, type=Path)
        if name != "inspect":
            p.add_argument("--output", required=True, type=Path)
        if name == "experiment":
            p.add_argument("--request", required=True, type=Path)
        if name == "grade":
            p.add_argument("--artifact", required=True, type=Path)
            p.add_argument("--observations", required=True, type=Path)
    p = sub.add_parser("validate")
    p.add_argument("--artifact", required=True, type=Path)
    p.add_argument("--observations", required=True, type=Path)
    p.add_argument("--bundle", type=Path, help="Use this bundle's public port count; defaults to the 12-port reference")
    for name in ("fetch", "catalog"):
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True)
        p.add_argument("--revision", required=True, help="Full immutable 40-character HF commit")
        p.add_argument("--offline", action="store_true")
        p.add_argument("--cache", type=Path)
        if name == "fetch":
            p.add_argument("--path", required=True, help="Bundle directory within the dataset repository")
            p.add_argument("--profile", choices=["simulation", "evaluation"], default="evaluation")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            b = Bundle(args.bundle, profile="simulation")
            result = dict(
                references=b.references(),
                world=b.manifest["objects"]["world"],
                profiles={
                    p: sum(r["bytes"] for r in b.files.values() if p in r["profiles"])
                    for p in ("simulation", "evaluation")
                },
                status=b.manifest.get("status"),
                licenses=b.manifest.get("licenses"),
            )
        elif args.command == "demo":
            from .evaluation import reference_demo

            result = reference_demo(args.bundle, args.output)
        elif args.command == "experiment":
            import numpy as np

            from .blobround6_explore import ExperimentService
            from .evaluation import dump

            b = Bundle(args.bundle, profile="simulation")
            request = read_json(args.request)
            if set(request) != {"actions", "queries"}:
                raise BundleError("experiment request requires actions and queries only")
            # Reserve output before beginning a new native experiment.
            args.output.mkdir(parents=True, exist_ok=False)
            service = ExperimentService(
                b.make_oracle(), roster=b.roster, limits=b.limits, max_experiments=1, max_total_tu=50
            )
            prediction = service.experiment(request["actions"], request["queries"])
            np.savez_compressed(
                args.output / "observation.npz",
                request=json.dumps(request),
                **{f"query{i}": a for i, a in enumerate(prediction["samples"])},
            )
            result = dict(
                references=b.references(),
                usage=service.usage(),
                observation=str((args.output / "observation.npz").resolve()),
            )
            dump(args.output / "experiment.json", result)
        elif args.command == "validate":
            from .evaluation import validate_predictor

            kwargs = {"roster": Bundle(args.bundle, profile="simulation").roster} if args.bundle else {}
            result = validate_predictor(args.artifact, args.observations, **kwargs)
            if not result["ok"]:
                print(json.dumps(result, indent=2))
                return 1
        elif args.command == "grade":
            from .evaluation import grade

            result = grade(args.artifact, args.observations, args.output, bundle=args.bundle)
            if result["status"] != "COMPLETE":
                print(json.dumps(result, indent=2))
                return 1
        else:
            from .hub import fetch_bundle, fetch_catalog

            common = dict(repo=args.repo, revision=args.revision, cache=args.cache, offline=args.offline)
            result = (
                fetch_catalog(**common)
                if args.command == "catalog"
                else dict(directory=str(fetch_bundle(path=args.path, profile=args.profile, **common)))
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"physim: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
