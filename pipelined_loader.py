import threading

import torch


class PipelinedLoader:
    """Loads data into pinned buffers on a background thread."""

    def __init__(self, load_fn, cuda_device=None):
        self.load_fn = load_fn
        self.cuda_device = cuda_device
        self.io_time = 0.0
        self._thread = None
        self._last_elapsed = 0.0

    def _run_load(self, c_idx, buf_idx, before_load=None):
        if self.cuda_device is not None:
            torch.cuda.set_device(self.cuda_device)
        if before_load is not None:
            before_load()
        return self.load_fn(c_idx, buf_idx)

    def load_sync(self, c_idx, buf_idx, before_load=None):
        elapsed = self._run_load(c_idx, buf_idx, before_load)
        if elapsed is not None:
            self.io_time += elapsed

    def start_async(self, c_idx, buf_idx, before_load=None):
        if self._thread is not None:
            raise RuntimeError("async load already in flight")

        def worker():
            self._last_elapsed = self._run_load(c_idx, buf_idx, before_load)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def wait(self):
        if self._thread is None:
            return
        self._thread.join()
        if self._last_elapsed is not None:
            self.io_time += self._last_elapsed
        self._thread = None
        self._last_elapsed = 0.0
