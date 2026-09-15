"""Trusted local exploration host using the same R6 experiment semantics.

This adapter is intended to sit outside an eventual isolated agent runtime. It
is not a security boundary or a replacement for R5's production transport.
"""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import secrets
import math
from threading import Lock

from . import blobround6 as R6
from .blobround6_eval import (DEFAULT_LIMITS, DEFAULT_ROSTER, EvaluationError,
                             validate_case)


class ExperimentService:
    """One prepared physical origin; independent forcing on every experiment.

    Each request gives its complete action history from the same opaque start.
    A serialized host-owned budget also prevents simultaneous requests from
    bypassing the experiment cap. Outputs contain native sensor values only.
    """

    def __init__(self, oracle, *, roster=DEFAULT_ROSTER, max_experiments=100,
                 max_total_tu=5000, limits=None):
        if type(max_experiments) is not int or max_experiments < 1:
            raise ValueError('max_experiments must be a positive integer')
        if type(max_total_tu) not in (int, float) or not 0 < max_total_tu <= 1e6:
            raise ValueError('max_total_tu must be finite and in (0,1000000]')
        self._oracle = oracle
        self._roster = roster
        self._limits = limits or replace(DEFAULT_LIMITS, max_horizon_tu=50.)
        self._max_experiments = max_experiments
        self._max_ticks = math.floor(max_total_tu / R6.SIM_DT)
        self._calls = self._ticks = 0
        self._lock = Lock()
        self._seeds = set()

    def experiment(self, actions, queries):
        """Return {'samples':[arrays]} with one independently noisy member.

        Invalid requests do not consume budget. Once execution is admitted its
        full requested horizon is charged, including if native execution fails.
        Caller-supplied seeds, fields, poses and hidden metadata are unsupported.
        """
        case = dict(id='experiment', actions=deepcopy(actions), queries=deepcopy(queries))
        shapes = validate_case(case, roster=self._roster, limits=self._limits, allow_empty=True)
        if sum(t * p * s for t, p, s in shapes) > self._limits.max_output_values:
            raise EvaluationError('experimental output exceeds the scalar-value cap')
        ticks = max((round(t / R6.SIM_DT) for q in case['queries'] for t in q['t']), default=0)
        with self._lock:
            if self._calls >= self._max_experiments or self._ticks + ticks > self._max_ticks:
                raise EvaluationError('experimental budget exhausted')
            self._calls += 1
            self._ticks += ticks
            seed = secrets.randbits(64)
            while seed in self._seeds:
                seed = secrets.randbits(64)
            self._seeds.add(seed)
            return self._oracle.sample_truth(case['actions'], case['queries'],
                                             n_samples=1, truth_seed=seed)

    def usage(self):
        with self._lock:
            return dict(experiments=self._calls, max_experiments=self._max_experiments,
                        charged_tu=self._ticks * R6.SIM_DT,
                        max_total_tu=self._max_ticks * R6.SIM_DT)
