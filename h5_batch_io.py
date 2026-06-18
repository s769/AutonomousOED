import numpy as np


def build_si_read_plan(s_indices, nt):
    """Group sensor row blocks into contiguous HDF5 hyperslab reads."""
    indexed = sorted(enumerate(s_indices), key=lambda pair: int(pair[1]))
    plan = []
    i = 0
    n = len(indexed)
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[j][1] + 1:
            j += 1
        dest_blocks = [indexed[t][0] for t in range(i, j + 1)]
        row_start = int(indexed[i][1])
        row_end = int(indexed[j][1])
        plan.append(
            {
                "h5_row_start": row_start * nt,
                "h5_row_end": (row_end + 1) * nt,
                "dest_blocks": dest_blocks,
            }
        )
        i = j + 1
    return plan


def _dest_blocks_contiguous(dest_blocks):
    return dest_blocks[-1] - dest_blocks[0] + 1 == len(dest_blocks)


def _read_hyperslab(h5_dset, dest, selection):
    try:
        h5_dset.read_direct(dest, selection)
    except (TypeError, ValueError, OSError):
        dest[...] = h5_dset[selection]


def load_si_column_blocks(h5_dset, col_start, col_end, nt, dest_view, read_plan, slab_view=None):
    """Load cross-term blocks for one candidate column into a pinned (k*Nt, Nt) array."""
    for entry in read_plan:
        h5_rs = entry["h5_row_start"]
        h5_re = entry["h5_row_end"]
        dest_blocks = entry["dest_blocks"]
        selection = np.s_[h5_rs:h5_re, col_start:col_end]
        n_rows = h5_re - h5_rs

        if _dest_blocks_contiguous(dest_blocks):
            d0 = dest_blocks[0] * nt
            d1 = (dest_blocks[-1] + 1) * nt
            _read_hyperslab(h5_dset, dest_view[d0:d1, :], selection)
            continue

        if slab_view is None or slab_view.shape[0] < n_rows:
            slab = h5_dset[h5_rs:h5_re, col_start:col_end]
        else:
            slab = slab_view[:n_rows, :]
            _read_hyperslab(h5_dset, slab, selection)

        for b, block in enumerate(dest_blocks):
            row0 = block * nt
            dest_view[row0 : row0 + nt, :] = slab[b * nt : (b + 1) * nt, :]


def load_ii_block(h5_dset, col_start, col_end, dest_view):
    """Load the candidate diagonal Nt x Nt block."""
    selection = np.s_[col_start:col_end, col_start:col_end]
    _read_hyperslab(h5_dset, dest_view, selection)
