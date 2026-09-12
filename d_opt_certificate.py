"""A posteriori optimality certificates for regular D-optimal sensor designs.

Matches AutonomousOED's isotropic noise convention:

    K = r_sq * C + I

where ``C`` is the stored HDF5 data-space signal covariance (``K_matrix``)
and ``R = I`` after that scaling. The normalized expected information gain is

    f(S) = 0.5 * [logdet(K_S) - logdet(R_S)] = 0.5 * logdet(K_S),

since ``logdet(R_S) = 0``. Production ``select_sensors.py`` logs the unnormalized
``logdet(K_S)``; this module reports the normalized ``f``.

The bound uses budget ``k`` (not ``k - |S|``) when summing the largest remaining
marginal gains at a checkpoint ``S``. Floating-point evaluation is not an
interval-certified enclosure.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from h5_batch_io import build_si_read_plan, load_ii_block, load_si_column_blocks


# Default absolute tolerance for treating tiny negative Schur gains as roundoff.
DEFAULT_NEG_GAIN_TOL = 1e-10
# Tolerance for U(S) >= f(S_final) consistency checks.
DEFAULT_BOUND_TOL = 1e-8


@dataclass
class CheckpointResult:
    prefix_length: int
    prefix_information: float
    max_remaining_marginal: Optional[float]
    top_k_marginal_sum: Optional[float]
    upper_bound: Optional[float]
    running_min_upper_bound: Optional[float]
    n_remaining: int
    n_negative_gains_clamped: int
    weak_upper_bound: Optional[float] = None
    n_gains_evaluated: int = 0
    n_fallback_singleton: int = 0
    n_unavailable: int = 0
    bound_valid: bool = True


@dataclass
class CertificateSummary:
    h5_store: str
    dataset_name: str
    matrix_convention: str
    noise_convention: str
    r_sq: float
    Nt: int
    Nd: int
    candidate_universe: list[int]
    selected_sensor_ids: list[int]
    budget_k: int
    checkpoints: list[int]
    achieved_information: float
    optimum_upper_bound: Optional[float]
    best_bound_checkpoint: Optional[int]
    certified_ratio: Optional[float]
    absolute_gap_bound: Optional[float]
    weak_optimum_upper_bound: Optional[float] = None
    weak_certified_ratio: Optional[float] = None
    precision: str = "float64"
    device: str = "cpu"
    objective: str = (
        "normalized expected information gain "
        "f(S)=0.5*logdet(K_S) with K=I+r_sq*A (noise-normalized)"
    )
    empty_checkpoint_omissions: int = 0
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checkpoint_rows: list[dict] = field(default_factory=list)


def open_k_matrix(h5_store: str, dataset_name: Optional[str] = None):
    """Open the stored signal covariance and infer Nt, Nd from chunking."""
    import h5py

    h5_file = h5py.File(h5_store, mode="r", rdcc_nbytes=0)
    if dataset_name is None:
        dataset_name = "K_matrix" if "K_matrix" in h5_file else list(h5_file.keys())[0]
    if dataset_name not in h5_file:
        h5_file.close()
        raise KeyError(f"Dataset {dataset_name!r} not found in {h5_store}")

    h5_dset = h5_file[dataset_name]
    if h5_dset.ndim != 2 or h5_dset.shape[0] != h5_dset.shape[1]:
        h5_file.close()
        raise ValueError(f"Expected square 2D dataset; got shape {h5_dset.shape}")
    if h5_dset.chunks is None:
        h5_file.close()
        raise ValueError(
            "HDF5 dataset has no chunk metadata; cannot infer sensor block size Nt"
        )

    Nt = int(h5_dset.chunks[0])
    if h5_dset.chunks[1] != Nt:
        h5_file.close()
        raise ValueError(
            f"Expected square chunks (Nt, Nt); got {h5_dset.chunks}"
        )
    N = int(h5_dset.shape[0])
    if N % Nt != 0:
        h5_file.close()
        raise ValueError(f"Matrix size {N} is not divisible by Nt={Nt}")
    Nd = N // Nt
    return h5_file, h5_dset, dataset_name, Nt, Nd


def load_selection_file(path: str) -> tuple[list[int], Optional[list[float]]]:
    """Load ordered sensor IDs from a select_sensors checkpoint or ID list.

    Supported formats:
      - two columns: ``sensor_id cumulative_logdet`` (production checkpoint)
      - one column: sensor IDs only
    """
    data = np.loadtxt(path)
    if data.size == 0:
        raise ValueError(f"Selection file {path} is empty")
    if data.ndim == 1:
        # Ambiguous: either a single row of two values, or a 1-D ID list.
        if data.shape[0] == 2 and not np.isclose(data[1], int(data[1])):
            sensor_ids = [int(data[0])]
            cum_logdets = [float(data[1])]
            return sensor_ids, cum_logdets
        sensor_ids = [int(x) for x in data.tolist()]
        return sensor_ids, None
    if data.shape[1] == 1:
        return [int(x) for x in data[:, 0].tolist()], None
    if data.shape[1] >= 2:
        sensor_ids = [int(x) for x in data[:, 0].tolist()]
        cum_logdets = [float(x) for x in data[:, 1].tolist()]
        return sensor_ids, cum_logdets
    raise ValueError(f"Unrecognized selection file shape {data.shape} in {path}")


def validate_selection(
    sensor_ids: Sequence[int],
    Nd: int,
    *,
    allow_empty: bool = False,
) -> list[str]:
    """Return warning strings; raise on hard errors."""
    warnings: list[str] = []
    if len(sensor_ids) == 0 and not allow_empty:
        raise ValueError("Selection is empty")
    seen: set[int] = set()
    for s in sensor_ids:
        if s < 0 or s >= Nd:
            raise ValueError(f"Sensor id {s} outside candidate universe [0, {Nd})")
        if s in seen:
            raise ValueError(f"Duplicate sensor id {s} in selection")
        seen.add(s)
    return warnings


def parse_checkpoints(
    selection_len: int,
    budget_k: int,
    checkpoints: Optional[Sequence[int]] = None,
    checkpoint_stride: Optional[int] = None,
    all_prefixes: bool = False,
) -> list[int]:
    """Build sorted unique checkpoint prefix lengths in ``[0, budget_k]``."""
    if budget_k < 0:
        raise ValueError("budget_k must be non-negative")
    if budget_k > selection_len:
        raise ValueError(
            f"budget_k={budget_k} exceeds selection length {selection_len}"
        )

    values: set[int] = set()
    if all_prefixes:
        values.update(range(0, budget_k + 1))
    if checkpoint_stride is not None:
        if checkpoint_stride <= 0:
            raise ValueError("checkpoint_stride must be positive")
        values.update(range(0, budget_k + 1, checkpoint_stride))
        values.add(budget_k)
    if checkpoints is not None:
        for c in checkpoints:
            if c < 0 or c > budget_k:
                raise ValueError(
                    f"Checkpoint {c} outside [0, budget_k={budget_k}]"
                )
            values.add(int(c))

    if not values:
        # Default: empty set and final design.
        values.update({0, budget_k})

    return sorted(values)


def _logdet_from_chol(L: torch.Tensor) -> float:
    return float(2.0 * torch.sum(torch.log(torch.diag(L))).item())


def _normalized_info_from_logdet(logdet_k: float) -> float:
    """f(S) for isotropic R = I."""
    return 0.5 * logdet_k


def solve_triangular_leading_dim(
    L_work: torch.Tensor,
    n: int,
    B: torch.Tensor,
    *,
    block: Optional[int] = None,
) -> torch.Tensor:
    """In-place solve ``L_work[:n, :n] Y = B`` when ``L_work`` has lda > n.

    PyTorch ``solve_triangular`` on a non-contiguous leading principal submatrix
    materializes a full ``n x n`` copy (tens of GB here). This blocked forward
    substitution only contiguizes small diagonal panels and thin ``L21`` panels.
    """
    if n == 0:
        return B
    if L_work.shape[0] < n or L_work.shape[1] < n:
        raise ValueError("L_work is smaller than n")
    if B.shape[0] != n:
        raise ValueError(f"B has {B.shape[0]} rows but n={n}")

    # Contiguous exact-sized factor: defer to the highly tuned kernel.
    L_view = L_work[:n, :n]
    if L_view.is_contiguous():
        torch.linalg.solve_triangular(L_view, B, upper=False, out=B)
        return B

    if block is None:
        block = min(n, max(64, B.shape[1] if B.ndim == 2 else 64))

    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        L11 = L_work[i0:i1, i0:i1].contiguous()
        rhs = B[i0:i1]
        torch.linalg.solve_triangular(L11, rhs, upper=False, out=rhs)
        if i1 < n:
            # Thin panel copy is O(n * block), not O(n^2).
            L21 = L_work[i1:n, i0:i1].contiguous()
            B[i1:].addmm_(L21, rhs, alpha=-1.0)
    return B


def build_k_submatrix_from_h5(
    h5_dset,
    Nt: int,
    S_indices: Sequence[int],
    device: torch.device,
    r_sq: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build ``K_S = r_sq * C_S + I`` for selected sensor blocks."""
    current_size = len(S_indices) * Nt
    K_S = torch.zeros((current_size, current_size), dtype=dtype, device=device)

    for row_idx, s_row in enumerate(S_indices):
        for col_idx, s_col in enumerate(S_indices):
            if col_idx < row_idx:
                continue
            block_start_row, block_start_col = s_row * Nt, s_col * Nt
            block_numpy = np.asarray(
                h5_dset[
                    block_start_row : block_start_row + Nt,
                    block_start_col : block_start_col + Nt,
                ]
            )
            if np.isnan(block_numpy).any():
                block_numpy = np.nan_to_num(block_numpy, nan=0.0)
            block_gpu = torch.from_numpy(np.ascontiguousarray(block_numpy)).to(
                device=device, dtype=dtype
            )
            if row_idx == col_idx:
                block_gpu = 0.5 * (block_gpu + block_gpu.T)
            block_gpu.mul_(r_sq)
            K_S[
                row_idx * Nt : (row_idx + 1) * Nt,
                col_idx * Nt : (col_idx + 1) * Nt,
            ] = block_gpu
            if col_idx > row_idx:
                K_S[
                    col_idx * Nt : (col_idx + 1) * Nt,
                    row_idx * Nt : (row_idx + 1) * Nt,
                ] = block_gpu.T

    if current_size > 0:
        K_S.diagonal().add_(1.0)
    return K_S


def cholesky_from_selection(
    h5_dset,
    Nt: int,
    S_indices: Sequence[int],
    device: torch.device,
    r_sq: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    """Return lower Cholesky factor of ``K_S`` and normalized information f(S)."""
    if len(S_indices) == 0:
        empty = torch.zeros((0, 0), dtype=dtype, device=device)
        return empty, 0.0

    K_S = build_k_submatrix_from_h5(h5_dset, Nt, S_indices, device, r_sq, dtype)
    # Soft symmetry check
    sym_err = float(torch.max(torch.abs(K_S - K_S.T)).item())
    if sym_err > 1e-6 * max(1.0, float(torch.max(torch.abs(K_S)).item())):
        K_S = 0.5 * (K_S + K_S.T)

    try:
        L = torch.linalg.cholesky(K_S)
    except torch.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"Cholesky failed for selection of length {len(S_indices)}: {exc}"
        ) from exc

    f_S = _normalized_info_from_logdet(_logdet_from_chol(L))
    return L, f_S


def extend_cholesky_with_sensor(
    h5_dset,
    Nt: int,
    L_S: torch.Tensor,
    S_indices_before: Sequence[int],
    new_sensor: int,
    device: torch.device,
    r_sq: float,
    dtype: torch.dtype,
    L_workspace: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, float]:
    """Append one sensor to an existing Cholesky factor; return (L_new, gain Δ)."""
    k = len(S_indices_before)
    col_start = new_sensor * Nt
    col_end = col_start + Nt

    h5_dtype = h5_dset.dtype

    K_ii = torch.empty((Nt, Nt), dtype=dtype, device=device)
    pinned_ii = np.empty((Nt, Nt), dtype=h5_dtype)
    load_ii_block(h5_dset, col_start, col_end, pinned_ii)
    K_ii.copy_(torch.from_numpy(np.ascontiguousarray(pinned_ii)))
    if torch.is_floating_point(K_ii):
        K_ii = 0.5 * (K_ii + K_ii.T)
    K_ii.mul_(r_sq).diagonal().add_(1.0)

    if k == 0:
        L_ii = torch.linalg.cholesky(K_ii)
        gain = _normalized_info_from_logdet(_logdet_from_chol(L_ii))
        if L_workspace is not None:
            L_workspace[:Nt, :Nt] = L_ii
            return L_workspace[:Nt, :Nt], gain
        return L_ii, gain

    plan = build_si_read_plan(np.asarray(S_indices_before), Nt)
    max_rows = max(
        (entry["h5_row_end"] - entry["h5_row_start"] for entry in plan), default=Nt
    )
    dest = np.empty((k * Nt, Nt), dtype=h5_dtype)
    slab = np.empty((max_rows, Nt), dtype=h5_dtype)
    load_si_column_blocks(
        h5_dset, col_start, col_end, Nt, dest, plan, slab_view=slab
    )
    K_Si = torch.from_numpy(np.ascontiguousarray(dest)).to(device=device, dtype=dtype)
    K_Si.mul_(r_sq)

    # In-place TRSM; supports non-contiguous leading views of L_workspace.
    if L_workspace is not None and L_S.data_ptr() == L_workspace.data_ptr():
        solve_triangular_leading_dim(L_workspace, k * Nt, K_Si)
        Y = K_Si
    else:
        torch.linalg.solve_triangular(L_S, K_Si, upper=False, out=K_Si)
        Y = K_Si
    K_ii.addmm_(Y.T, Y, beta=1.0, alpha=-1.0)
    L_ii = torch.linalg.cholesky(K_ii)
    gain = _normalized_info_from_logdet(_logdet_from_chol(L_ii))

    new_size = (k + 1) * Nt
    if L_workspace is not None:
        # Leading block already holds L_S when L_S is a view into L_workspace.
        if L_S.data_ptr() != L_workspace.data_ptr():
            L_workspace[: k * Nt, : k * Nt] = L_S
        L_workspace[k * Nt : new_size, : k * Nt] = Y.T
        L_workspace[k * Nt : new_size, k * Nt : new_size] = L_ii
        return L_workspace[:new_size, :new_size], gain

    L_new = torch.zeros((new_size, new_size), dtype=dtype, device=device)
    L_new[: k * Nt, : k * Nt] = L_S
    L_new[k * Nt : new_size, : k * Nt] = Y.T
    L_new[k * Nt : new_size, k * Nt : new_size] = L_ii
    return L_new, gain


def marginal_gain_schur(
    h5_dset,
    Nt: int,
    L_S: torch.Tensor,
    S_indices: Sequence[int],
    candidate: int,
    device: torch.device,
    r_sq: float,
    dtype: torch.dtype,
    *,
    si_plan=None,
    si_slab=None,
    pinned_Si=None,
    pinned_ii=None,
    L_workspace: Optional[torch.Tensor] = None,
) -> float:
    """Normalized marginal gain Δ_i(S) via Schur complement."""
    k = len(S_indices)
    col_start = candidate * Nt
    col_end = col_start + Nt
    h5_dtype = h5_dset.dtype

    if pinned_ii is None:
        pinned_ii = np.empty((Nt, Nt), dtype=h5_dtype)
    load_ii_block(h5_dset, col_start, col_end, pinned_ii)

    K_ii = torch.from_numpy(np.ascontiguousarray(pinned_ii)).to(
        device=device, dtype=dtype
    )
    K_ii = 0.5 * (K_ii + K_ii.T)
    K_ii.mul_(r_sq).diagonal().add_(1.0)

    if k == 0:
        L_ii = torch.linalg.cholesky(K_ii)
        return _normalized_info_from_logdet(_logdet_from_chol(L_ii))

    if si_plan is None:
        si_plan = build_si_read_plan(np.asarray(S_indices), Nt)
    if pinned_Si is None:
        pinned_Si = np.empty((k * Nt, Nt), dtype=h5_dtype)
    load_si_column_blocks(
        h5_dset,
        col_start,
        col_end,
        Nt,
        pinned_Si[: k * Nt, :],
        si_plan,
        slab_view=si_slab,
    )
    K_Si = torch.from_numpy(np.ascontiguousarray(pinned_Si[: k * Nt, :])).to(
        device=device, dtype=dtype
    )
    K_Si.mul_(r_sq)
    n = k * Nt
    if L_workspace is not None and L_S.data_ptr() == L_workspace.data_ptr():
        solve_triangular_leading_dim(L_workspace, n, K_Si)
    else:
        torch.linalg.solve_triangular(L_S, K_Si, upper=False, out=K_Si)
    K_ii.addmm_(K_Si.T, K_Si, beta=1.0, alpha=-1.0)
    L_ii = torch.linalg.cholesky(K_ii)
    return _normalized_info_from_logdet(_logdet_from_chol(L_ii))


def top_k_sum(values: Iterable[float], k: int) -> tuple[float, Optional[float], int]:
    """Return (sum of largest min(k,n) values, max value or None, n)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    heap: list[float] = []
    n = 0
    max_v: Optional[float] = None
    for v in values:
        n += 1
        max_v = v if max_v is None else max(max_v, v)
        if k == 0:
            continue
        if len(heap) < k:
            heapq.heappush(heap, v)
        elif v > heap[0]:
            heapq.heapreplace(heap, v)
    return (float(sum(heap)) if heap else 0.0), max_v, n


def adjust_gain(
    gain: float,
    neg_tol: float,
) -> tuple[float, bool, Optional[str]]:
    """Clamp tiny negative gains; flag larger negatives as inconsistent."""
    if gain >= 0.0:
        return gain, False, None
    if gain >= -neg_tol:
        return 0.0, True, None
    return gain, False, (
        f"Materially negative marginal gain {gain!r} exceeds tolerance {neg_tol}"
    )


def compute_singleton_gains(
    h5_dset,
    Nt: int,
    candidates: Sequence[int],
    device: torch.device,
    r_sq: float,
    dtype: torch.dtype,
    neg_tol: float = DEFAULT_NEG_GAIN_TOL,
    comm=None,
    show_progress: bool = False,
) -> tuple[dict[int, float], list[int], list[str]]:
    """Compute f({{i}}) for each candidate; return (gains, failed_ids, warnings)."""
    rank = 0
    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        size = 1

    local = list(candidates)[rank::size]
    L0 = torch.zeros((0, 0), dtype=dtype, device=device)
    local_gains: dict[int, float] = {}
    local_failed: list[int] = []
    warnings: list[str] = []

    iterator = local
    if show_progress and rank == 0:
        try:
            from tqdm import tqdm

            iterator = tqdm(local, desc="singleton gains", leave=False)
        except ImportError:
            pass

    for cand in iterator:
        try:
            raw = marginal_gain_schur(
                h5_dset, Nt, L0, [], cand, device, r_sq, dtype
            )
        except torch.linalg.LinAlgError as exc:
            local_failed.append(int(cand))
            warnings.append(f"Singleton Cholesky failed for candidate {cand}: {exc}")
            continue
        adj, clamped, err = adjust_gain(raw, neg_tol)
        if err is not None:
            local_failed.append(int(cand))
            warnings.append(
                f"Singleton gain for candidate {cand} is materially negative "
                f"({raw}); treating as unavailable fallback."
            )
            continue
        if clamped:
            warnings.append(
                f"Singleton gain for candidate {cand} clamped from {raw} to 0 "
                f"(roundoff tol={neg_tol})."
            )
        local_gains[int(cand)] = adj

    if comm is not None:
        parts = comm.gather((local_gains, local_failed, warnings), root=0)
        if rank == 0:
            gains: dict[int, float] = {}
            failed: list[int] = []
            all_warn: list[str] = []
            for g, f, w in parts:
                gains.update(g)
                failed.extend(f)
                all_warn.extend(w)
            payload = (gains, failed, all_warn)
        else:
            payload = None
        gains, failed, warnings = comm.bcast(payload, root=0)
        return gains, failed, warnings

    return local_gains, local_failed, warnings


def scan_remaining_marginals(
    h5_dset,
    Nt: int,
    L_S: torch.Tensor,
    S_indices: Sequence[int],
    remaining: Sequence[int],
    device: torch.device,
    r_sq: float,
    dtype: torch.dtype,
    budget_k: int,
    neg_tol: float = DEFAULT_NEG_GAIN_TOL,
    comm=None,
    show_progress: bool = False,
    L_workspace: Optional[torch.Tensor] = None,
    singleton_gains: Optional[dict[int, float]] = None,
) -> tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    int,
    list[str],
    int,
    int,
    int,
    bool,
]:
    """Scan remaining candidates; return top-k sum and diagnostics.

    Failed Schur evaluations are never silently omitted. Their contribution is
    replaced by the monotone upper bound ``Δ_i(S) <= Δ_i(∅) = f({{i}})`` when
    available. If that fallback is also unavailable, the checkpoint bound is
    marked invalid.

    Returns
    -------
    top_sum, max_gain, weak_extra, n_clamped, warnings,
    n_evaluated, n_fallback, n_unavailable, bound_valid
    """
    rank = 0
    size = 1
    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()

    local_remaining = list(remaining)[rank::size]
    warnings: list[str] = []
    k_prefix = len(S_indices)
    h5_dtype = h5_dset.dtype
    pinned_ii = np.empty((Nt, Nt), dtype=h5_dtype)
    pinned_Si = None
    si_plan = None
    si_slab = None
    if k_prefix > 0:
        si_plan = build_si_read_plan(np.asarray(S_indices), Nt)
        max_rows = max(
            (e["h5_row_end"] - e["h5_row_start"] for e in si_plan), default=Nt
        )
        pinned_Si = np.empty((k_prefix * Nt, Nt), dtype=h5_dtype)
        si_slab = np.empty((max_rows, Nt), dtype=h5_dtype)

    gains: list[float] = []
    n_clamped = 0
    n_fallback = 0
    n_unavailable = 0
    iterator = local_remaining
    if show_progress and rank == 0:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                local_remaining,
                desc=f"scan |S|={k_prefix} (rank0/{size})",
                leave=False,
            )
        except ImportError:
            pass

    for cand in iterator:
        used_fallback = False
        try:
            raw = marginal_gain_schur(
                h5_dset,
                Nt,
                L_S,
                S_indices,
                cand,
                device,
                r_sq,
                dtype,
                si_plan=si_plan,
                si_slab=si_slab,
                pinned_Si=pinned_Si,
                pinned_ii=pinned_ii,
                L_workspace=L_workspace,
            )
        except torch.linalg.LinAlgError as exc:
            if singleton_gains is not None and cand in singleton_gains:
                raw = singleton_gains[cand]
                used_fallback = True
                warnings.append(
                    f"Cholesky failed for candidate {cand} at |S|={k_prefix}; "
                    f"using singleton upper bound Delta_i(empty)={raw:.6g}. ({exc})"
                )
            else:
                n_unavailable += 1
                warnings.append(
                    f"Cholesky failed for candidate {cand} at |S|={k_prefix} and "
                    f"no singleton fallback is available; checkpoint bound invalid. ({exc})"
                )
                continue

        adj, clamped, err = adjust_gain(raw, neg_tol)
        if err is not None and not used_fallback:
            if singleton_gains is not None and cand in singleton_gains:
                adj = singleton_gains[cand]
                used_fallback = True
                clamped = False
                warnings.append(
                    f"Materially negative marginal {raw} for candidate {cand} at "
                    f"|S|={k_prefix}; using singleton upper bound {adj:.6g}."
                )
            else:
                n_unavailable += 1
                warnings.append(
                    f"{err}; no singleton fallback for candidate {cand}; "
                    "checkpoint bound invalid."
                )
                continue
        if clamped:
            n_clamped += 1
        if used_fallback:
            n_fallback += 1
        gains.append(adj)

    if comm is not None:
        all_gains = comm.gather(gains, root=0)
        all_warnings = comm.gather(warnings, root=0)
        stats = comm.gather(
            (n_clamped, n_fallback, n_unavailable, len(local_remaining)), root=0
        )
        if rank == 0:
            gains = [g for part in all_gains for g in part]
            warnings = [w for part in all_warnings for w in part]
            n_clamped = int(sum(s[0] for s in stats))
            n_fallback = int(sum(s[1] for s in stats))
            n_unavailable = int(sum(s[2] for s in stats))
            n_remaining_total = int(sum(s[3] for s in stats))
        else:
            gains = []
            warnings = []
            n_clamped = 0
            n_fallback = 0
            n_unavailable = 0
            n_remaining_total = 0
    else:
        n_remaining_total = len(remaining)

    # IMPORTANT: sum the largest `budget_k` remaining gains, not (budget_k - |S|).
    if comm is None or rank == 0:
        bound_valid = n_unavailable == 0 and len(gains) == n_remaining_total
        if n_unavailable > 0:
            top_sum = None
            max_gain = None
            weak_extra = None
        else:
            n_take = min(budget_k, len(gains))
            top_sum, max_gain, _ = top_k_sum(gains, n_take)
            weak_extra = budget_k * max_gain if max_gain is not None else 0.0
        payload = (
            top_sum,
            max_gain,
            weak_extra,
            n_clamped,
            warnings,
            len(gains),
            n_fallback,
            n_unavailable,
            bound_valid,
        )
    else:
        payload = None

    if comm is not None:
        payload = comm.bcast(payload, root=0)
    return payload


def certify_selection(
    h5_dset,
    Nt: int,
    Nd: int,
    selected: Sequence[int],
    budget_k: int,
    checkpoints: Sequence[int],
    r_sq: float,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    candidate_universe: Optional[Sequence[int]] = None,
    neg_tol: float = DEFAULT_NEG_GAIN_TOL,
    bound_tol: float = DEFAULT_BOUND_TOL,
    dataset_name: str = "K_matrix",
    h5_store: str = "",
    comm=None,
    show_progress: bool = False,
) -> tuple[list[CheckpointResult], CertificateSummary]:
    """Compute submodular a-posteriori certificates along a fixed selection."""
    if r_sq <= 0.0:
        raise ValueError(f"r_sq must be positive; got {r_sq}")

    rank = 0 if comm is None else comm.Get_rank()

    notes = [
        "Stored HDF5 matrix is treated as signal covariance A; "
        "working data covariance is K = I + r_sq * A (noise-normalized).",
        "Reported values use f(S)=0.5*logdet(K_S); ratios match g=logdet(K_S).",
        "Upper bounds are rigorous in exact arithmetic; FP64 evaluation is not "
        "an interval-certified enclosure.",
        "This certificate applies only to regular D-optimality with independent "
        "isotropic sensor-noise blocks; it does not apply to goal-oriented "
        "D-optimality or cross-sensor-correlated noise.",
        "The top-k marginal sum uses budget k, not k-|S|, because the unknown "
        "optimum need not contain the checkpoint set.",
        "Failed Schur evaluations use the monotone fallback "
        "Δ_i(S) <= Δ_i(∅)=f({i}); candidates are never silently omitted.",
        "The classical (1-1/e) greedy guarantee is not applied automatically.",
        "Cholesky factors are built incrementally along the supplied trajectory "
        "(no full dense K_S materialization).",
    ]
    warnings: list[str] = []

    universe = list(range(Nd) if candidate_universe is None else candidate_universe)
    if sorted(universe) != list(range(Nd)) and set(universe) != set(range(Nd)):
        missing = set(range(Nd)) - set(universe)
        if missing:
            notes.append(
                f"Candidate universe is a subset of all matrix blocks "
                f"(excluded {len(missing)} sensors)."
            )
    validate_selection(selected[:budget_k], Nd)
    for s in selected[:budget_k]:
        if s not in set(universe):
            raise ValueError(f"Selected sensor {s} is not in the candidate universe")

    selected = list(selected[:budget_k])
    cps = parse_checkpoints(len(selected), budget_k, checkpoints=checkpoints)

    if rank == 0:
        print("Precomputing singleton gains for fallback upper bounds...", flush=True)
    singleton_gains, singleton_failed, singleton_warnings = compute_singleton_gains(
        h5_dset,
        Nt,
        universe,
        device,
        r_sq,
        dtype,
        neg_tol=neg_tol,
        comm=comm,
        show_progress=show_progress,
    )
    warnings.extend(singleton_warnings)
    empty_omissions = len(singleton_failed)
    if empty_omissions:
        warnings.append(
            f"{empty_omissions} candidate(s) lack a usable singleton gain; "
            "any checkpoint that needs those fallbacks will be marked invalid."
        )
    elif rank == 0:
        print(
            f"Singleton gains OK for all {len(universe)} candidates "
            f"(empty-checkpoint omissions = 0).",
            flush=True,
        )

    # Pre-allocate Cholesky workspace for the longest needed prefix.
    max_prefix = max(max(cps), budget_k) if cps else budget_k
    max_size = max_prefix * Nt
    if rank == 0:
        bytes_l = max_size * max_size * (8 if dtype == torch.float64 else 4)
        print(
            f"Allocating Cholesky workspace {max_size}x{max_size} on {device} "
            f"(~{bytes_l / 1e9:.1f} GB)",
            flush=True,
        )
    L_workspace = torch.zeros((max_size, max_size), dtype=dtype, device=device)
    L_S = L_workspace[0:0, 0:0]
    f_S = 0.0
    prefix: list[int] = []

    results: list[CheckpointResult] = []
    u_best = math.inf
    best_cp: Optional[int] = None
    next_needed = 0

    for cp in cps:
        if cp < next_needed:
            raise RuntimeError("checkpoints must be nondecreasing")
        # Extend Cholesky along the supplied trajectory (not a new greedy step).
        extend_range = range(len(prefix), cp)
        if show_progress and rank == 0 and cp > len(prefix):
            try:
                from tqdm import tqdm

                extend_range = tqdm(
                    extend_range, desc=f"build L -> |S|={cp}", leave=False
                )
            except ImportError:
                pass
        for _ in extend_range:
            new_s = selected[len(prefix)]
            L_S, _delta = extend_cholesky_with_sensor(
                h5_dset,
                Nt,
                L_S,
                prefix,
                new_s,
                device,
                r_sq,
                dtype,
                L_workspace=L_workspace,
            )
            prefix.append(new_s)
            f_S = _normalized_info_from_logdet(_logdet_from_chol(L_S))
        next_needed = cp

        # Recompute f from current L for numerical consistency.
        f_S = 0.0 if cp == 0 else _normalized_info_from_logdet(_logdet_from_chol(L_S))
        if rank == 0:
            print(
                f"Checkpoint |S|={cp}: f(S)={f_S:.6g}; scanning remaining candidates...",
                flush=True,
            )

        remaining = [i for i in universe if i not in set(prefix)]
        (
            top_sum,
            max_gain,
            weak_extra,
            n_clamped,
            scan_warnings,
            n_eval,
            n_fallback,
            n_unavail,
            bound_valid,
        ) = scan_remaining_marginals(
            h5_dset,
            Nt,
            L_S,
            prefix,
            remaining,
            device,
            r_sq,
            dtype,
            budget_k,
            neg_tol=neg_tol,
            comm=comm,
            show_progress=show_progress,
            L_workspace=L_workspace,
            singleton_gains=singleton_gains,
        )
        warnings.extend(scan_warnings)
        if n_clamped:
            warnings.append(
                f"Checkpoint |S|={cp}: clamped {n_clamped} tiny negative "
                f"marginal gain(s) to 0 within tol={neg_tol}"
            )
        if n_fallback:
            warnings.append(
                f"Checkpoint |S|={cp}: used singleton fallback for {n_fallback} "
                "failed/unstable marginal(s) (bound remains valid but looser)."
            )

        if bound_valid and top_sum is not None:
            u_s: Optional[float] = f_S + top_sum
            weak_u = None if max_gain is None else f_S + budget_k * max_gain
            if u_s < u_best:
                u_best = u_s
                best_cp = cp
        else:
            u_s = None
            weak_u = None
            warnings.append(
                f"Checkpoint |S|={cp}: upper bound unavailable "
                f"({n_unavail} candidate(s) without fallback)."
            )

        if rank == 0:
            if u_s is None:
                print(
                    f"  bound UNAVAILABLE (evaluated={n_eval}, fallback={n_fallback}, "
                    f"unavailable={n_unavail})",
                    flush=True,
                )
            else:
                print(
                    f"  maxΔ={max_gain}; top-{budget_k} sum={top_sum:.6g}; "
                    f"U={u_s:.6g}; U_best={u_best:.6g}; "
                    f"fallback={n_fallback}",
                    flush=True,
                )

        results.append(
            CheckpointResult(
                prefix_length=cp,
                prefix_information=f_S,
                max_remaining_marginal=max_gain,
                top_k_marginal_sum=top_sum,
                upper_bound=u_s,
                running_min_upper_bound=(
                    None if not math.isfinite(u_best) else u_best
                ),
                n_remaining=len(remaining),
                n_negative_gains_clamped=n_clamped,
                weak_upper_bound=weak_u,
                n_gains_evaluated=n_eval,
                n_fallback_singleton=n_fallback,
                n_unavailable=n_unavail,
                bound_valid=bound_valid,
            )
        )

    # Ensure f(S_final) even if budget_k was not itself a checkpoint.
    while len(prefix) < budget_k:
        new_s = selected[len(prefix)]
        L_S, _delta = extend_cholesky_with_sensor(
            h5_dset,
            Nt,
            L_S,
            prefix,
            new_s,
            device,
            r_sq,
            dtype,
            L_workspace=L_workspace,
        )
        prefix.append(new_s)
    f_final = (
        0.0
        if budget_k == 0
        else _normalized_info_from_logdet(_logdet_from_chol(L_S))
    )

    if not math.isfinite(u_best):
        warnings.append(
            "No valid checkpoint upper bound was obtained; "
            "certified_ratio is undefined."
        )
        u_best_out: Optional[float] = None
        gap: Optional[float] = None
        ratio: Optional[float] = None
    else:
        u_best_out = u_best
        # Consistency: every valid U(S) should be >= f(S_final) within tolerance.
        for row in results:
            if row.upper_bound is None:
                continue
            if row.upper_bound + bound_tol < f_final:
                warnings.append(
                    f"Inconsistent bound at prefix {row.prefix_length}: "
                    f"U={row.upper_bound} < f(S_final)={f_final} "
                    f"(diff={f_final - row.upper_bound})"
                )

        if abs(f_final) <= bound_tol and abs(u_best) <= bound_tol:
            ratio = None
            warnings.append(
                "Zero-information case: f(S_final) and U_best are both ~0; "
                "certified_ratio is left undefined."
            )
            gap = u_best - f_final
        elif abs(u_best) <= bound_tol:
            ratio = None
            warnings.append(
                "U_best is ~0 while f(S_final) is not; certified_ratio left undefined."
            )
            gap = u_best - f_final
        else:
            ratio = f_final / u_best
            gap = u_best - f_final
            if ratio > 1.0 + bound_tol:
                warnings.append(
                    f"certified_ratio={ratio} exceeds 1 beyond tolerance; "
                    "bound/design inconsistency (not clamped)."
                )

    weak_best = None
    weak_ratio = None
    finite_weak = [r.weak_upper_bound for r in results if r.weak_upper_bound is not None]
    if finite_weak:
        weak_best = min(finite_weak)
        if abs(weak_best) > bound_tol:
            weak_ratio = f_final / weak_best

    summary = CertificateSummary(
        h5_store=h5_store,
        dataset_name=dataset_name,
        matrix_convention="stored A = signal covariance (F Γ_0 F*); not full K",
        noise_convention="noise-normalized: K = I + r_sq * A",
        r_sq=float(r_sq),
        Nt=Nt,
        Nd=Nd,
        candidate_universe=list(universe),
        selected_sensor_ids=selected,
        budget_k=budget_k,
        checkpoints=list(cps),
        achieved_information=f_final,
        optimum_upper_bound=u_best_out,
        best_bound_checkpoint=best_cp,
        certified_ratio=ratio,
        absolute_gap_bound=gap,
        weak_optimum_upper_bound=weak_best,
        weak_certified_ratio=weak_ratio,
        precision="float64" if dtype == torch.float64 else "float32",
        device=str(device),
        empty_checkpoint_omissions=empty_omissions,
        notes=notes,
        warnings=warnings,
        checkpoint_rows=[asdict(r) for r in results],
    )
    return results, summary


def write_checkpoint_csv(path: str, results: Sequence[CheckpointResult]) -> None:
    fieldnames = [
        "prefix_length",
        "prefix_information",
        "max_remaining_marginal",
        "top_k_marginal_sum",
        "upper_bound",
        "running_min_upper_bound",
        "n_remaining",
        "n_negative_gains_clamped",
        "weak_upper_bound",
        "n_gains_evaluated",
        "n_fallback_singleton",
        "n_unavailable",
        "bound_valid",
    ]
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def write_summary_json(path: str, summary: CertificateSummary) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = asdict(summary)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def format_terminal_summary(summary: CertificateSummary) -> str:
    ratio = summary.certified_ratio
    if summary.optimum_upper_bound is None:
        ratio_str = "undefined (no valid U_best)"
        ub_str = "unavailable"
        gap_str = "unavailable"
        best_cp_str = "n/a"
    else:
        ratio_str = "undefined (zero-information)" if ratio is None else f"{ratio:.6f}"
        ub_str = f"{summary.optimum_upper_bound:.12g}"
        gap_str = (
            "undefined"
            if summary.absolute_gap_bound is None
            else f"{summary.absolute_gap_bound:.12g}"
        )
        best_cp_str = str(summary.best_bound_checkpoint)
    lines = [
        "=== D-optimality a-posteriori certificate ===",
        f"Matrix: {summary.h5_store} [{summary.dataset_name}]",
        f"Noise:  {summary.noise_convention} (r_sq={summary.r_sq})",
        f"Nt={summary.Nt}, Nd={summary.Nd}, k={summary.budget_k}",
        f"Checkpoints: {summary.checkpoints}",
        f"Empty-checkpoint singleton omissions: {summary.empty_checkpoint_omissions}",
        f"Selected: {summary.selected_sensor_ids}",
        "",
        f"achieved_information f(S_final) = {summary.achieved_information:.12g}",
        f"optimum_upper_bound U_best      = {ub_str}"
        f"  (from checkpoint |S|={best_cp_str})",
        f"certified_ratio f/U_best       = {ratio_str}",
        f"absolute_gap_bound U - f       = {gap_str}",
        "",
        "Note: ratio is for noise-normalized information "
        "f=0.5*logdet(K_S) with K=I+r_sq*A (equivalently g=logdet; factor 1/2 cancels).",
        "Floating-point bounds are not interval-certified enclosures.",
    ]
    if summary.weak_optimum_upper_bound is not None:
        wr = summary.weak_certified_ratio
        wr_str = "undefined" if wr is None else f"{wr:.6f}"
        lines.extend(
            [
                "",
                "Weaker bound using k * max remaining marginal "
                "(does not need the full top-k list):",
                f"  weak_U_best = {summary.weak_optimum_upper_bound:.12g}",
                f"  weak_ratio  = {wr_str}",
            ]
        )
    if summary.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in summary.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# In-memory helpers for synthetic tests (no HDF5).
# ---------------------------------------------------------------------------


class DenseBlockMatrix:
    """Minimal block-matrix view over a dense Nd*Nt covariance (signal C)."""

    def __init__(self, C: np.ndarray, Nt: int):
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("C must be square")
        if C.shape[0] % Nt != 0:
            raise ValueError("size not divisible by Nt")
        self.C = np.asarray(C, dtype=np.float64)
        self.Nt = Nt
        self.Nd = C.shape[0] // Nt
        self.shape = C.shape
        self.dtype = np.float64
        self.chunks = (Nt, Nt)

    def __getitem__(self, selection):
        return self.C[selection]

    def read_direct(self, dest, selection):
        dest[...] = self.C[selection]


def normalized_f_dense(C: np.ndarray, Nt: int, S: Sequence[int], r_sq: float) -> float:
    """Direct f(S) = 0.5 * logdet(K_S) for isotropic R=I."""
    if len(S) == 0:
        return 0.0
    idx = []
    for s in S:
        idx.extend(range(s * Nt, (s + 1) * Nt))
    C_S = C[np.ix_(idx, idx)]
    K_S = r_sq * C_S + np.eye(len(idx))
    sign, logdet = np.linalg.slogdet(K_S)
    if sign <= 0:
        raise RuntimeError("K_S not SPD in dense f()")
    return 0.5 * float(logdet)


def exhaustive_optimal_f(
    C: np.ndarray, Nt: int, Nd: int, k: int, r_sq: float
) -> tuple[float, tuple[int, ...]]:
    """Brute-force maximum f(S) over all designs with |S|<=k (actually |S|=k)."""
    from itertools import combinations

    best_f = 0.0
    best_S: tuple[int, ...] = ()
    # Empty set
    # |S| from 1..k (monotonicity => optimum at size k for positive gains, but
    # keep <=k for completeness).
    for m in range(0, k + 1):
        for S in combinations(range(Nd), m):
            val = normalized_f_dense(C, Nt, S, r_sq)
            if val > best_f:
                best_f = val
                best_S = S
    return best_f, best_S
