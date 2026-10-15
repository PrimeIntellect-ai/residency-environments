"""Compatibility alias; Eleusis now supplies rate-limit recovery in every native eval."""

from verifiers.v1.cli.eval.main import main

if __name__ == "__main__":
    main()
