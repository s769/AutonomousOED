"""Synthetic tests for regular D-optimality a-posteriori certificates.

CPU-only, small Nt/Nd. Run from the AutonomousOED repo root:

  python -m unittest tests.test_certify_d_optimality -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from itertools import combinations

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from d_opt_certificate import (  # noqa: E402
    DenseBlockMatrix,
    adjust_gain,
    certify_selection,
    cholesky_from_selection,
    exhaustive_optimal_f,
    extend_cholesky_with_sensor,
    load_selection_file,
    marginal_gain_schur,
    normalized_f_dense,
    parse_checkpoints,
    top_k_sum,
)


def _make_psd_signal(Nd: int, Nt: int, r_sq: float, seed: int = 0) -> np.ndarray:
    """Build a PSD signal covariance C with multi-observation sensor blocks."""
    rng = np.random.RandomState(seed)
    n = Nd * Nt
    # Low-rank-ish factor + diagonal jitter => SPD signal covariance.
    rank = max(2, n // 2)
    A = rng.randn(n, rank)
    C = A @ A.T / rank
    # Mild within-sensor temporal correlation structure.
    for s in range(Nd):
        sl = slice(s * Nt, (s + 1) * Nt)
        C[sl, sl] += 0.05 * np.eye(Nt)
    C = 0.5 * (C + C.T)
    # Scale so r_sq*C + I is well-conditioned for the chosen r_sq.
    scale = float(np.trace(C) / n)
    C = C / scale
    return C


class TestHelpers(unittest.TestCase):
    def test_parse_checkpoints_default(self):
        self.assertEqual(parse_checkpoints(5, 5), [0, 5])

    def test_parse_checkpoints_stride_and_explicit(self):
        cps = parse_checkpoints(10, 10, checkpoints=[3], checkpoint_stride=4)
        self.assertEqual(cps, [0, 3, 4, 8, 10])

    def test_top_k_sum(self):
        s, mx, n = top_k_sum([1.0, 5.0, 3.0, 4.0], k=2)
        self.assertEqual(n, 4)
        self.assertEqual(mx, 5.0)
        self.assertAlmostEqual(s, 9.0)

    def test_adjust_gain_roundoff(self):
        g, clamped, err = adjust_gain(-1e-12, 1e-10)
        self.assertEqual(g, 0.0)
        self.assertTrue(clamped)
        self.assertIsNone(err)

    def test_adjust_gain_material_negative(self):
        g, clamped, err = adjust_gain(-1e-3, 1e-10)
        self.assertLess(g, 0.0)
        self.assertFalse(clamped)
        self.assertIsNotNone(err)

    def test_load_selection_formats(self):
        with tempfile.TemporaryDirectory() as td:
            two_col = os.path.join(td, "ckpt.txt")
            np.savetxt(two_col, np.array([[2, 1.5], [0, 2.7]]), fmt="%d %.18e")
            ids, logs = load_selection_file(two_col)
            self.assertEqual(ids, [2, 0])
            self.assertEqual(len(logs), 2)

            one_col = os.path.join(td, "ids.txt")
            np.savetxt(one_col, np.array([4, 1, 7]), fmt="%d")
            ids2, logs2 = load_selection_file(one_col)
            self.assertEqual(ids2, [4, 1, 7])
            self.assertIsNone(logs2)


class TestSchurVsDirect(unittest.TestCase):
    def test_schur_matches_direct_logdet(self):
        Nd, Nt, r_sq = 5, 3, 2.5
        C = _make_psd_signal(Nd, Nt, r_sq, seed=1)
        mat = DenseBlockMatrix(C, Nt)
        device = torch.device("cpu")
        dtype = torch.float64

        S = [1, 3]
        L, f_S = cholesky_from_selection(mat, Nt, S, device, r_sq, dtype)
        self.assertAlmostEqual(f_S, normalized_f_dense(C, Nt, S, r_sq), places=10)

        for cand in [0, 2, 4]:
            d_schur = marginal_gain_schur(mat, Nt, L, S, cand, device, r_sq, dtype)
            d_direct = normalized_f_dense(C, Nt, S + [cand], r_sq) - f_S
            self.assertAlmostEqual(d_schur, d_direct, places=9)

        # Empty set: M_i = K_ii.
        L0, f0 = cholesky_from_selection(mat, Nt, [], device, r_sq, dtype)
        self.assertEqual(f0, 0.0)
        for cand in range(Nd):
            d_schur = marginal_gain_schur(mat, Nt, L0, [], cand, device, r_sq, dtype)
            d_direct = normalized_f_dense(C, Nt, [cand], r_sq)
            self.assertAlmostEqual(d_schur, d_direct, places=10)


class TestCertificateVsExhaustive(unittest.TestCase):
    def _run_case(self, selected, k, r_sq=1.7, seed=2):
        Nd, Nt = 6, 2
        C = _make_psd_signal(Nd, Nt, r_sq, seed=seed)
        mat = DenseBlockMatrix(C, Nt)
        device = torch.device("cpu")
        f_opt, S_opt = exhaustive_optimal_f(C, Nt, Nd, k, r_sq)

        results, summary = certify_selection(
            h5_dset=mat,
            Nt=Nt,
            Nd=Nd,
            selected=selected,
            budget_k=k,
            checkpoints=list(range(0, k + 1)),
            r_sq=r_sq,
            device=device,
            dtype=torch.float64,
            h5_store="synthetic",
            dataset_name="dense",
        )
        f_final = summary.achieved_information
        self.assertAlmostEqual(f_final, normalized_f_dense(C, Nt, selected[:k], r_sq), places=10)

        for row in results:
            self.assertGreaterEqual(row.upper_bound + 1e-8, f_opt)

        # Ratio must not exceed the true achieved/optimal ratio.
        if f_opt > 1e-14 and summary.certified_ratio is not None:
            true_ratio = f_final / f_opt
            self.assertLessEqual(summary.certified_ratio, true_ratio + 1e-8)

        return results, summary, f_opt, S_opt

    def test_greedy_like_trajectory(self):
        # Near-greedy ordering by sequential max gain on a tiny instance.
        Nd, Nt, k, r_sq = 6, 2, 3, 1.7
        C = _make_psd_signal(Nd, Nt, r_sq, seed=2)
        mat = DenseBlockMatrix(C, Nt)
        device = torch.device("cpu")
        selected = []
        remaining = list(range(Nd))
        L, _ = cholesky_from_selection(mat, Nt, [], device, r_sq, torch.float64)
        for _ in range(k):
            best_i, best_g = None, -np.inf
            for cand in remaining:
                g = marginal_gain_schur(
                    mat, Nt, L, selected, cand, device, r_sq, torch.float64
                )
                if g > best_g:
                    best_g, best_i = g, cand
            L, _ = extend_cholesky_with_sensor(
                mat, Nt, L, selected, best_i, device, r_sq, torch.float64
            )
            selected.append(best_i)
            remaining.remove(best_i)

        results, summary, f_opt, _ = self._run_case(selected, k)
        self.assertGreaterEqual(summary.optimum_upper_bound + 1e-8, f_opt)
        self.assertGreaterEqual(summary.certified_ratio, 0.0)
        self.assertLessEqual(summary.certified_ratio, 1.0 + 1e-8)
        # Smoke: checkpoint CSV fields present.
        self.assertEqual(results[0].prefix_length, 0)
        self.assertEqual(results[-1].prefix_length, k)

    def test_non_greedy_trajectory(self):
        # Deliberately poor order: reverse of a decent ordering.
        selected = [5, 4, 3]
        results, summary, f_opt, _ = self._run_case(selected, k=3, seed=3)
        self.assertGreaterEqual(summary.optimum_upper_bound + 1e-8, f_opt)
        # Bound still valid even though trajectory is not greedy.
        for row in results:
            self.assertGreaterEqual(row.upper_bound + 1e-8, f_opt)

    def test_non_unit_noise_scaling(self):
        selected = [0, 2, 4]
        _, summary, f_opt, _ = self._run_case(selected, k=3, r_sq=0.35, seed=4)
        self.assertGreaterEqual(summary.optimum_upper_bound + 1e-8, f_opt)
        # Changing r_sq changes the absolute information scale.
        _, summary2, _, _ = self._run_case(selected, k=3, r_sq=4.0, seed=4)
        self.assertNotAlmostEqual(
            summary.achieved_information, summary2.achieved_information, places=6
        )


class TestEdgeCases(unittest.TestCase):
    def test_fewer_than_k_candidates_remaining(self):
        Nd, Nt, r_sq = 4, 2, 1.0
        C = _make_psd_signal(Nd, Nt, r_sq, seed=5)
        mat = DenseBlockMatrix(C, Nt)
        # k=3 but at prefix |S|=2 only 2 candidates remain; top-k sum uses min(k, remaining).
        selected = [0, 1, 2]
        results, summary = certify_selection(
            h5_dset=mat,
            Nt=Nt,
            Nd=Nd,
            selected=selected,
            budget_k=3,
            checkpoints=[2, 3],
            r_sq=r_sq,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        row = results[0]
        self.assertEqual(row.n_remaining, 2)
        self.assertLessEqual(row.top_k_marginal_sum, 1e6)  # finite
        # With only 2 remaining, top_k sum has at most 2 terms even though k=3.
        # Recompute expected max possible terms.
        self.assertEqual(summary.budget_k, 3)

    def test_incorrect_kmS_regression(self):
        """Expose why the sum must use k, not k-|S|.

        Construct gains at a nonempty checkpoint where summing only the largest
        (k-|S|) remaining marginals under-bounds the exhaustive optimum, while
        summing the largest k remaining marginals remains valid.
        """
        Nd, Nt, k, r_sq = 5, 2, 2, 1.0
        # Hand-built nearly block-diagonal signal: sensors are almost independent
        # so marginals stay large even after selecting one.
        n = Nd * Nt
        C = np.zeros((n, n), dtype=np.float64)
        for s in range(Nd):
            sl = slice(s * Nt, (s + 1) * Nt)
            # Distinct strengths so ordering is controlled.
            strength = 10.0 * (s + 1)
            C[sl, sl] = strength * np.eye(Nt)
        mat = DenseBlockMatrix(C, Nt)
        device = torch.device("cpu")
        dtype = torch.float64

        # Checkpoint S={0}, budget k=2. Remaining gains are large and comparable.
        S = [0]
        L, f_S = cholesky_from_selection(mat, Nt, S, device, r_sq, dtype)
        remaining = [1, 2, 3, 4]
        gains = [
            marginal_gain_schur(mat, Nt, L, S, c, device, r_sq, dtype) for c in remaining
        ]
        gains_sorted = sorted(gains, reverse=True)

        u_correct = f_S + sum(gains_sorted[:k])  # largest k
        u_wrong = f_S + sum(gains_sorted[: max(k - len(S), 0)])  # largest k-|S|

        f_opt, _ = exhaustive_optimal_f(C, Nt, Nd, k, r_sq)
        self.assertGreaterEqual(u_correct + 1e-8, f_opt)
        # The incorrect k-|S| formula fails to upper-bound the optimum here.
        self.assertLess(u_wrong, f_opt - 1e-6)

        # Full certificate path uses the correct formula.
        _, summary = certify_selection(
            h5_dset=mat,
            Nt=Nt,
            Nd=Nd,
            selected=[0, 4],
            budget_k=k,
            checkpoints=[1],
            r_sq=r_sq,
            device=device,
            dtype=dtype,
        )
        self.assertGreaterEqual(summary.optimum_upper_bound + 1e-8, f_opt)

    def test_zero_information(self):
        Nd, Nt, r_sq = 3, 2, 1.0
        C = np.zeros((Nd * Nt, Nd * Nt), dtype=np.float64)
        mat = DenseBlockMatrix(C, Nt)
        _, summary = certify_selection(
            h5_dset=mat,
            Nt=Nt,
            Nd=Nd,
            selected=[0, 1],
            budget_k=2,
            checkpoints=[0, 2],
            r_sq=r_sq,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        # K = I => logdet = 0 => f = 0.
        self.assertAlmostEqual(summary.achieved_information, 0.0, places=12)
        self.assertAlmostEqual(summary.optimum_upper_bound, 0.0, places=10)
        self.assertIsNone(summary.certified_ratio)
        self.assertTrue(any("Zero-information" in w for w in summary.warnings))

    def test_invalid_duplicate_sensor(self):
        Nd, Nt, r_sq = 3, 2, 1.0
        C = _make_psd_signal(Nd, Nt, r_sq, seed=6)
        mat = DenseBlockMatrix(C, Nt)
        with self.assertRaises(ValueError):
            certify_selection(
                h5_dset=mat,
                Nt=Nt,
                Nd=Nd,
                selected=[0, 0],
                budget_k=2,
                checkpoints=[0, 2],
                r_sq=r_sq,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )

    def test_invalid_sensor_id(self):
        Nd, Nt, r_sq = 3, 2, 1.0
        C = _make_psd_signal(Nd, Nt, r_sq, seed=6)
        mat = DenseBlockMatrix(C, Nt)
        with self.assertRaises(ValueError):
            certify_selection(
                h5_dset=mat,
                Nt=Nt,
                Nd=Nd,
                selected=[0, 99],
                budget_k=2,
                checkpoints=[2],
                r_sq=r_sq,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )

    def test_invalid_r_sq(self):
        Nd, Nt = 3, 2
        C = _make_psd_signal(Nd, Nt, 1.0, seed=6)
        mat = DenseBlockMatrix(C, Nt)
        with self.assertRaises(ValueError):
            certify_selection(
                h5_dset=mat,
                Nt=Nt,
                Nd=Nd,
                selected=[0],
                budget_k=1,
                checkpoints=[0, 1],
                r_sq=0.0,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )


class TestExhaustiveSanity(unittest.TestCase):
    def test_exhaustive_beats_all_listed(self):
        Nd, Nt, k, r_sq = 5, 2, 2, 1.2
        C = _make_psd_signal(Nd, Nt, r_sq, seed=7)
        f_opt, S_opt = exhaustive_optimal_f(C, Nt, Nd, k, r_sq)
        for S in combinations(range(Nd), k):
            self.assertGreaterEqual(f_opt + 1e-10, normalized_f_dense(C, Nt, S, r_sq))
        self.assertEqual(len(S_opt), k)


if __name__ == "__main__":
    unittest.main()
