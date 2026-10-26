"""midog-mitosis: a hard, binary-verifiable mitotic-figure benchmark and RL environment.

A Verifiers v1 taskset over MIDOG++ (CC0): 503 H&E tumor slides from humans and dogs with consensus mitotic-figure
annotations. One judge-free task in two tiers: find every mitotic figure in an image holding at least three (the benchmark)
or in a field holding exactly one (warm-up), and answer with the coordinates of each. The answer is free response and the
reward is binary (the points match the annotations one-to-one, or not). The benchmark is 100 test tasks on a custom
slide-disjoint split; every train and validation config excludes all test slides.

The taskset carries no tools: a coding harness plays it in a network-free sandbox that holds the image as a file, with
Pillow and NumPy installed. ``eval midog-mitosis`` uses the built-in ``bash`` harness unless told otherwise.
"""

from .taskset import MidogConfig, MidogData, MidogTask, MidogTaskset

__all__ = ["MidogConfig", "MidogData", "MidogTask", "MidogTaskset"]
