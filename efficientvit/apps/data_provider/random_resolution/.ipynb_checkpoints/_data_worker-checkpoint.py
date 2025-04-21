"""This file is based on jittor/utils/data/_utils/worker.py

Contains definitions of the methods used by the _BaseDataLoaderIter workers.
These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import os
import queue
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import jittor as jt
from jittor._utils import ExceptionWrapper
from jittor.utils.data._utils import HAS_NUMPY, IS_WINDOWS, MP_STATUS_CHECK_INTERVAL, signal_handling

if TYPE_CHECKING:
    from jittor.utils.data import Dataset

from .controller import RRSController

if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import BOOL, DWORD, HANDLE

    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog:
        def __init__(self):
            self.manager_pid = os.getppid()

            # mypy cannot detect this code is windows only
            self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(SYNCHRONIZE, 0, self.manager_pid)

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]

            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                # Value obtained from https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
            return not self.manager_dead

else:

    class ManagerWatchdog:  # type: ignore[no-redef]
        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead


_worker_info = None


class WorkerInfo:
    id: int
    num_workers: int
    seed: int
    dataset: "Dataset"
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__keys = tuple(kwargs.keys())
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError("Cannot assign attributes to {} objects".format(self.__class__.__name__))
        return super().__setattr__(key, val)

    def __repr__(self):
        items = []
        for k in self.__keys:
            items.append("{}={}".format(k, getattr(self, k)))
        return "{}({})".format(self.__class__.__name__, ", ".join(items))


def get_worker_info() -> Optional[WorkerInfo]:
    """Returns the information about the current
    :class:`~jittor.utils.data.DataLoader` iterator worker process.

    When called in a worker, this returns an object guaranteed to have the
    following attributes:

    * :attr:`id`: the current worker id.
    * :attr:`num_workers`: the total number of workers.
    * :attr:`seed`: the random seed set for the current worker. This value is
      determined by main process RNG and the worker id. See
      :class:`~jittor.utils.data.DataLoader`'s documentation for more details.
    * :attr:`dataset`: the copy of the dataset object in **this** process. Note
      that this will be a different object in a different process than the one
      in the main process.

    When called in the main process, this returns ``None``.

    .. note::
       When used in a :attr:`worker_init_fn` passed over to
       :class:`~jittor.utils.data.DataLoader`, this method can be useful to
       set up each worker process differently, for instance, using ``worker_id``
       to configure the ``dataset`` object to only read a specific fraction of a
       sharded dataset, or use ``seed`` to seed other libraries used in dataset
       code.
    """
    return _worker_info


"""Dummy class used to signal the end of an IterableDataset"""


@dataclass(frozen=True)
class _IterableDatasetStopIteration:
    worker_id: int


"""Dummy class used to resume the fetching when worker reuse is enabled"""


@dataclass(frozen=True)
class _ResumeIteration:
    seed: Optional[int] = None


def _generate_state(base_seed, worker_id):
    INIT_A = 0x43B0D7E5
    MULT_A = 0x931E8875
    INIT_B = 0x8B51F9DD
    MULT_B = 0x58F38DED
    MIX_MULT_L = 0xCA01F9DD
    MIX_MULT_R = 0x4973F715
    XSHIFT = 4 * 8 // 2
    MASK32 = 0xFFFFFFFF

    entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [0] * 4

    hash_const_A = INIT_A

    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value

    def mix(x, y):
        result_x = (MIX_MULT_L * x) & MASK32
        result_y = (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    # Add in the entropy to the pool.
    for i in range(len(pool)):
        pool[i] = hash(entropy[i])

    # Mix all bits together so late bits can affect earlier bits.
    for i_src in range(len(pool)):
        for i_dst in range(len(pool)):
            if i_src != i_dst:
                pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

    hash_const_B = INIT_B
    state = []
    for i_dst in range(4):
        data_val = pool[i_dst]
        data_val = (data_val ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        data_val = (data_val * hash_const_B) & MASK32
        data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
        state.append(data_val)
    return state


def _worker_loop(
    dataset_kind,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    base_seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
    shared_seed,
):
    try:
        signal_handling._set_worker_signal_handlers()

        jt.set_global_seed(base_seed + worker_id)
        random.seed(base_seed + worker_id)
        if HAS_NUMPY:
            np_seed = _generate_state(base_seed, worker_id)
            import numpy as np
            np.random.seed(np_seed)

        from jittor.utils.data import IterDataPipe
        from jittor.utils.data.graph_settings import apply_random_seed

        shared_rng = jt.random(seed=shared_seed) if shared_seed is not None else None
        if isinstance(dataset, IterDataPipe) and shared_rng is not None:
            dataset = apply_random_seed(dataset, shared_rng)

        global _worker_info
        _worker_info = WorkerInfo(id=worker_id, num_workers=num_workers, seed=base_seed + worker_id, dataset=dataset)

        from jittor.utils.data import _DatasetKind

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)
        except Exception:
            init_exception = ExceptionWrapper(where="in DataLoader worker process {}".format(worker_id))

        iteration_end = False
        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if isinstance(r, _ResumeIteration):
                data_queue.put((r, None))
                iteration_end = False

                if isinstance(dataset, IterDataPipe):
                    assert r.seed is not None
                    shared_rng = jt.random(seed=r.seed)
                    dataset = apply_random_seed(dataset, shared_rng)

                fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)
                continue
            elif r is None:
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                continue
            idx, index = r
            """ Added """
            RRSController.sample_resolution(batch_id=idx)
            """ Added """
            data: Union[_IterableDatasetStopIteration, ExceptionWrapper]
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(index)
                except Exception as e:
                    if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
                        data = _IterableDatasetStopIteration(worker_id)
                        iteration_end = True
                    else:
                        data = ExceptionWrapper(where="in DataLoader worker process {}".format(worker_id))
            data_queue.put((idx, data))
            del data, idx, index, r
    except KeyboardInterrupt:
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()