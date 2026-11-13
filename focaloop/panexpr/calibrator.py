"""Compact online empirical error model (16 Q-by-mate cells) fit per sequencing run.

Bins mean-Phred bucket[8] x mate[2] x outcome[3: match/sub/indel] (48 sufficient
statistics -> 16 substitution rates and 16 indel rates) on the first 5,000 confident
alignments for ``MappyKernel``; external SAM/cDBG kernels score candidates from aligner
``AS:i:`` tags or native equivalence-class weights.
"""

from __future__ import annotations

import math

import numpy as np


class OnlineErrorCalibrator:
    def __init__(self) -> None:
        self.counts = np.zeros((8, 2, 3), dtype=np.float64)
        nominal_q = np.array([12, 18, 23, 28, 32, 35, 38, 41], dtype=np.float64)
        p_err = np.power(10.0, -nominal_q / 10.0)
        for qb in range(8):
            pe = float(p_err[qb])
            self.counts[qb, :, 0] = 160.0 * (1.0 - pe)
            self.counts[qb, :, 1] = 160.0 * (pe * 0.85)
            self.counts[qb, :, 2] = 160.0 * (pe * 0.15)
        self.n_calibrated_reads = 0
        self._dirty = True
        self._sub_rate_by_q = np.zeros((8, 2))
        self._indel_rate_by_q = np.zeros((8, 2))
        self._refresh()

    @staticmethod
    def mean_q(qual: str) -> float:
        if not qual:
            return 30.0
        b = qual.encode("ascii", "replace")
        return (sum(b) / len(b)) - 33.0

    @staticmethod
    def _q_bucket(mean_q: float) -> int:
        for i, thr in enumerate((15, 20, 25, 30, 34, 37, 40)):
            if mean_q < thr:
                return i
        return 7

    def update_from_hit(self, qual: str | float, mate: int, mlen: int, blen: int, nm: int) -> None:
        if self.n_calibrated_reads >= 5000 or blen < 50 or nm > 4:
            return
        mq = float(qual) if isinstance(qual, (int, float)) else self.mean_q(qual)
        qb = self._q_bucket(mq)
        indels = max(0, blen - mlen)
        subs = max(0, nm - indels)
        matches = max(0, mlen - subs)
        self.counts[qb, mate, 0] += matches
        self.counts[qb, mate, 1] += subs
        self.counts[qb, mate, 2] += indels
        self.n_calibrated_reads += 1
        self._dirty = True

    def _refresh(self) -> None:
        c = self.counts.sum(axis=1) if self.counts.ndim == 4 else self.counts
        tot = c.sum(axis=2)
        subs = c[:, :, 1]
        inds = c[:, :, 2]
        self._sub_rate_by_q = np.clip(subs / tot, 1e-4, 0.15)
        self._indel_rate_by_q = np.clip(inds / tot, 5e-5, 0.05)
        self._dirty = False

    def score_alignment(self, seq_len: int, mean_q: float, mate: int, mlen: int, blen: int, nm: int) -> float:
        if self._dirty:
            self._refresh()
        qb = self._q_bucket(mean_q)
        p_sub = float(self._sub_rate_by_q[qb, mate])
        p_ind = float(self._indel_rate_by_q[qb, mate])
        p_ok = max(0.5, 1.0 - p_sub - p_ind)
        indels = max(0, blen - mlen)
        subs = max(0, nm - indels)
        matches = max(0, mlen - subs)
        unaligned = max(0, seq_len - blen)
        return matches * math.log(p_ok) + subs * math.log(p_sub / 3.0) + indels * math.log(p_ind) + unaligned * math.log(0.04)

    @property
    def summary_error_rate(self) -> float:
        if self._dirty:
            self._refresh()
        c = self.counts.sum(axis=1) if self.counts.ndim == 4 else self.counts
        tot = float(c.sum())
        return float(c[:, :, 1:].sum()) / tot if tot > 0 else 0.002

    def clone(self) -> "OnlineErrorCalibrator":
        c = OnlineErrorCalibrator()
        c.counts = self.counts.copy()
        c.n_calibrated_reads = self.n_calibrated_reads
        c._dirty = True
        return c
