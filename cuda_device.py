"""Resolve the CUDA device for the current MPI/Slurm task."""

import os

import torch


def _probe_device(device_id: int) -> bool:
    try:
        torch.cuda.set_device(device_id)
        x = torch.zeros(1, device=torch.device(f"cuda:{device_id}"))
        torch.cuda.synchronize(device_id)
        del x
        return True
    except RuntimeError:
        return False


def _slurm_gpu_ids() -> list[int] | None:
    for key in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        raw = os.environ.get(key)
        if not raw:
            continue
        ids = [int(part) for part in raw.replace(" ", "").split(",") if part]
        if ids:
            return ids
    return None


def _candidate_device_ids(local_rank: int, device_count: int) -> list[int]:
    candidates: list[int] = []
    seen: set[int] = set()

    def add(device_id: int) -> None:
        if 0 <= device_id < device_count and device_id not in seen:
            seen.add(device_id)
            candidates.append(device_id)

    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_list = (
        [int(part) for part in visible_raw.split(",") if part.strip() != ""]
        if visible_raw
        else None
    )

    if visible_list is not None and len(visible_list) == 1:
        add(0)
        return candidates

    slurm_gpus = _slurm_gpu_ids()
    if slurm_gpus:
        physical_id = slurm_gpus[local_rank % len(slurm_gpus)]
        if visible_list is not None:
            if physical_id in visible_list:
                add(visible_list.index(physical_id))
        else:
            add(physical_id)

    add(local_rank % device_count)
    for device_id in range(device_count):
        add(device_id)

    return candidates


def resolve_local_rank(mpi_rank: int = 0) -> int:
    return int(
        os.environ.get(
            "SLURM_LOCALID",
            os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", mpi_rank),
        )
    )


def resolve_torch_device(
    local_rank: int | None = None,
    mpi_rank: int | None = None,
) -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return torch.device("cpu")

    if local_rank is None:
        local_rank = resolve_local_rank(0 if mpi_rank is None else mpi_rank)

    for device_id in _candidate_device_ids(local_rank, torch.cuda.device_count()):
        if _probe_device(device_id):
            return torch.device(f"cuda:{device_id}")

    return torch.device("cpu")
