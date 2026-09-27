"""Trusted local exploration host using the same R6 experiment semantics.

This adapter is intended to sit outside an eventual isolated agent runtime. It
is not a security boundary or a replacement for R5's production transport.
"""
from __future__ import annotations

import math
import secrets
from copy import deepcopy
from dataclasses import replace
from threading import Lock

from . import blobround6 as R6
from .blobround6_eval import DEFAULT_LIMITS, DEFAULT_ROSTER, EvaluationError, validate_case


class RequestError(EvaluationError):
    """A public grammar/budget rejection, safe to return to the agent."""


class ExperimentService:
    """One prepared physical origin; independent forcing on every experiment.

    Each request gives its complete action history from the same opaque start.
    A serialized host-owned budget also prevents simultaneous requests from
    bypassing the experiment cap. Outputs contain native sensor values only.
    """

    def __init__(self, oracle, *, roster=DEFAULT_ROSTER, max_experiments=100,
                 max_total_tu=5000, limits=None):
        if max_experiments is not None and (type(max_experiments) is not int or max_experiments < 1):
            raise ValueError('max_experiments must be a positive integer or None')
        if max_total_tu is not None and (type(max_total_tu) not in (int, float) or not 0 < max_total_tu <= 1e6):
            raise ValueError('max_total_tu must be finite and in (0,1000000], or None')
        self._oracle = oracle
        self._roster = roster
        self._limits = limits or replace(DEFAULT_LIMITS, max_horizon_tu=50.)
        self._max_experiments = max_experiments
        self._max_ticks = None if max_total_tu is None else math.floor(max_total_tu / R6.SIM_DT)
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
        try:
            shapes = validate_case(case, roster=self._roster, limits=self._limits, allow_empty=True)
        except EvaluationError as exc:
            # Describe admissible timestamps, not the numerical implementation.
            # Keep the frozen simulator/parser bytes unchanged for existing bundles.
            message = str(exc).replace(f"on the dt={R6.SIM_DT:g} grid", f"a multiple of {R6.SIM_DT:g} time units")
            message = message.replace("at least one dt tick", f"at least {R6.SIM_DT:g} time units")
            message = message.replace("exact tick representation range", "supported timestamp range")
            message = message.replace("exact tick range", "supported timestamp range")
            raise RequestError(message) from None
        if sum(t * p * s for t, p, s in shapes) > self._limits.max_output_values:
            raise RequestError('experimental output exceeds the scalar-value cap')
        ticks = max((round(t / R6.SIM_DT) for q in case['queries'] for t in q['t']), default=0)
        with self._lock:
            if ((self._max_experiments is not None and self._calls >= self._max_experiments)
                    or (self._max_ticks is not None and self._ticks + ticks > self._max_ticks)):
                raise RequestError('experimental budget exhausted')
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
                        max_total_tu=None if self._max_ticks is None else self._max_ticks * R6.SIM_DT)
