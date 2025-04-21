import jittor as jt
import warnings
from typing import Any, Callable, Generic, Iterable, List, Optional, Sequence, TypeVar, Union
from jittor.dataset import Dataset
import math  

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]

class RRSDataLoader:
    def __init__(
        self,
        dataset,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler=None,
        batch_sampler=None,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = ""
    ):
        """
        Jittor implementation of RRSDataLoader with random resolution sampling support
        
        Args:
            dataset: Dataset to load data from (must be jittor Dataset)
            batch_size: How many samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of subprocesses for data loading
            drop_last: Drop last incomplete batch
            worker_init_fn: Worker initialization function
            persistent_workers: Keep workers alive between epochs
        """
        # Jittor specific initialization
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle if shuffle is not None else False
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.collate_fn = collate_fn
        
        # Jittor doesn't use these parameters but keep for compatibility
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.prefetch_factor = prefetch_factor
        self.pin_memory_device = pin_memory_device
        
        # Initialize dataset with Jittor settings
        self._init_dataset()
        
        # For random resolution sampling
        self._enable_rrs = hasattr(dataset, 'enable_rrs')
        if self._enable_rrs:
            self.dataset.enable_rrs(True)
            
        # For distributed training
        self._world_size = jt.world_size if hasattr(jt, 'world_size') else 1
        self._rank = jt.rank if hasattr(jt, 'rank') else 0

    def _init_dataset(self):
        """Initialize dataset with Jittor settings"""
        if not isinstance(self.dataset, jt.dataset.Dataset):
            raise TypeError("Dataset must be a jittor Dataset")
            
        # Set dataset attributes
        self.dataset.set_attrs(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )
        
    def __iter__(self):
        """Create iterator for the dataset"""
        if self.persistent_workers and self.num_workers > 0:
            if not hasattr(self, '_iterator'):
                self._iterator = iter(self.dataset)
            return self._iterator
        return iter(self.dataset)
        
    def __len__(self):
        """Get number of batches"""
        if self.batch_size is None:
            return len(self.dataset)
            
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
            
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for random sampling"""
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)
        elif self._enable_rrs:
            # For RRS, we can set random seed based on epoch
            jt.set_global_seed(epoch)
            
    def update_batch_size(self, new_batch_size: int):
        """Update batch size dynamically"""
        self.batch_size = new_batch_size
        self.dataset.set_attrs(batch_size=new_batch_size)
        
    def enable_rrs(self, enable: bool = True):
        """Enable/disable random resolution sampling"""
        if self._enable_rrs:
            self.dataset.enable_rrs(enable)
        else:
            warnings.warn("Dataset does not support random resolution sampling")

# Helper classes for compatibility
class _InfiniteConstantSampler:
    """Jittor version of infinite sampler"""
    def __iter__(self):
        while True:
            yield None

def _get_distributed_settings():
    """Get distributed settings from Jittor"""
    return (jt.world_size if hasattr(jt, 'world_size') else 1, 
            jt.rank if hasattr(jt, 'rank') else 0)

def _share_dist_seed(generator=None):
    """Share random seed in distributed training"""
    seed = jt.randint(0, 2**32-1)
    if hasattr(jt, 'distributed') and jt.distributed.is_initialized():
        jt.distributed.broadcast(seed, src=0)
    return seed

class _DatasetKind:
    Map = 0
    Iterable = 1

class _InfiniteConstantSampler:
    def __iter__(self):
        while True:
            yield None

class RRSDataLoader(Generic[T_co]):
    def __init__(
        self,
        dataset,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler=None,
        batch_sampler=None,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn=None,
        generator=None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = ""
    ):
        """
        Jittor implementation of RRSDataLoader with full PyTorch compatibility
        
        Args:
            dataset: Input dataset (Jittor Dataset or compatible)
            batch_size: How many samples per batch
            shuffle: Whether to shuffle the data
            sampler: Custom sampler (limited support in Jittor)
            batch_sampler: Custom batch sampler (limited support)
            num_workers: Number of parallel workers
            collate_fn: Custom collate function
            pin_memory: Automatic memory pinning (Jittor handles automatically)
            drop_last: Drop last incomplete batch
            timeout: Worker timeout (not fully supported in Jittor)
            worker_init_fn: Worker initialization function
            generator: Random generator (uses Jittor's RNG)
            prefetch_factor: Prefetch batches per worker
            persistent_workers: Maintain workers between epochs
            pin_memory_device: Device for pinned memory
        """
        # Basic validation
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError("prefetch_factor must be non-negative")
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")

        # Core attributes
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self._iterator = None

        # Determine dataset kind
        if hasattr(dataset, '__iter__') and not hasattr(dataset, '__getitem__'):
            self._dataset_kind = _DatasetKind.Iterable
        else:
            self._dataset_kind = _DatasetKind.Map
            shuffle = bool(shuffle) if shuffle is not None else False

        # Handle samplers
        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError("batch_sampler is mutually exclusive with other args")
            self.batch_size = None
            self.drop_last = False
        elif batch_size is None and drop_last:
            raise ValueError("batch_size=None is incompatible with drop_last")

        # Initialize default samplers
        if sampler is None:
            if self._dataset_kind == _DatasetKind.Iterable:
                sampler = _InfiniteConstantSampler()
            else:
                sampler = jt.randperm(len(dataset)) if shuffle else range(len(dataset))

        if batch_size is not None and batch_sampler is None:
            batch_sampler = self._create_batch_sampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

        # Initialize Jittor dataset settings
        self._init_jittor_dataset()

        # RRS support
        self._enable_rrs = hasattr(dataset, 'enable_rrs')
        if self._enable_rrs:
            dataset.enable_rrs(True)

        # Distributed settings
        self._world_size = jt.in_mpi and jt.world_size or 1
        self._rank = jt.in_mpi and jt.rank or 0

    def _init_jittor_dataset(self):
        """Configure underlying Jittor dataset"""
        if not isinstance(self.dataset, jt.dataset.Dataset):
            # Wrap non-Jittor datasets
            self.dataset = self._wrap_dataset(self.dataset)
            
        # Set core attributes
        self.dataset.set_attrs(
            batch_size=self.batch_size or 1,
            shuffle=self.shuffle if hasattr(self, 'shuffle') else False,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )

    def _wrap_dataset(self, dataset):
        """Create Jittor compatible dataset wrapper"""
        class DatasetWrapper(jt.dataset.Dataset):
            def __init__(self, dataset):
                super().__init__()
                self.dataset = dataset
                self.total_len = len(dataset) if hasattr(dataset, '__len__') else 0
                
            def __getitem__(self, idx):
                return self.dataset[idx]
                
            def __len__(self):
                return self.total_len
                
        return DatasetWrapper(dataset)

    def _create_batch_sampler(self, sampler, batch_size, drop_last):
        """Create batch sampler for map-style datasets"""
        if hasattr(sampler, '__len__'):
            sampler_len = len(sampler)
            num_batches = sampler_len // batch_size
            if drop_last and sampler_len % batch_size != 0:
                num_batches = math.floor(sampler_len / batch_size)
            else:
                num_batches = math.ceil(sampler_len / batch_size)
                
            def batch_sampler():
                for i in range(num_batches):
                    start = i * batch_size
                    end = min(start + batch_size, sampler_len)
                    yield sampler[start:end]
                    
            return batch_sampler()
        else:
            # For iterable samplers
            def batch_sampler():
                batch = []
                for idx in sampler:
                    batch.append(idx)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
                if batch and not drop_last:
                    yield batch
                    
            return batch_sampler()

    def __iter__(self):
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = iter(self.dataset)
            return self._iterator
        return iter(self.dataset)

    def __len__(self):
        if self._dataset_kind == _DatasetKind.Iterable:
            if not hasattr(self.dataset, '__len__'):
                raise TypeError("IterableDataset has no length")
            length = len(self.dataset)
            if self.batch_size is not None:
                length = length // self.batch_size if self.drop_last else math.ceil(length / self.batch_size)
            return length
        else:
            if self.batch_sampler is not None and hasattr(self.batch_sampler, '__len__'):
                return len(self.batch_sampler)
            elif hasattr(self.sampler, '__len__'):
                if self.batch_size is not None:
                    return len(self.sampler) // self.batch_size if self.drop_last else math.ceil(len(self.sampler) / self.batch_size)
                return len(self.sampler)
            else:
                raise TypeError("Sampler has no length")

    def set_epoch(self, epoch):
        """Set epoch for random sampling (for distributed training)"""
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)
        jt.set_global_seed(epoch)

    def enable_rrs(self, enable=True):
        """Enable/disable random resolution sampling"""
        if self._enable_rrs:
            self.dataset.enable_rrs(enable)
        else:
            warnings.warn("Dataset does not support random resolution sampling")

    def check_worker_number_rationality(self):
        """Check if worker count exceeds system resources"""
        if self.num_workers <= 0:
            return
            
        try:
            cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
            if cpu_count and self.num_workers > cpu_count:
                warnings.warn(
                    f"Using {self.num_workers} workers which exceeds system CPU count ({cpu_count}). "
                    "This may cause performance degradation."
                )
        except:
            pass

class _BaseDataLoaderIter:
    def __init__(self, loader: 'RRSDataLoader') -> None:
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        
        # Jittor distributed settings
        self._world_size = jt.world_size if hasattr(jt, 'world_size') else 1
        self._rank = jt.rank if hasattr(jt, 'rank') else 0
        
        # Memory pinning (Jittor handles automatically)
        self._pin_memory = loader.pin_memory
        self._pin_memory_device = loader.pin_memory_device
        
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = jt.randint(0, 2**32-1).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = f"enumerate(DataLoader)#{self.__class__.__name__}.__next__"

    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        if hasattr(loader, '_IterableDataset_len_called'):
            self._IterableDataset_len_called = loader._IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        # Jittor doesn't have profiler, so we skip profiling
        if self._sampler_iter is None:
            self._reset()
            
        data = self._next_data()
        self._num_yielded += 1
        
        # Warn if yielding more items than dataset length
        if (self._dataset_kind == _DatasetKind.Iterable and 
            hasattr(self, '_IterableDataset_len_called') and
            self._num_yielded > self._IterableDataset_len_called):
            warnings.warn(
                f"Length of IterableDataset {self._dataset} was reported to be {self._IterableDataset_len_called}, "
                f"but {self._num_yielded} samples have been fetched."
            )
        return data

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        raise NotImplementedError(f"{self.__class__.__name__} cannot be pickled")


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        # Initialize dataset fetcher
        self._dataset_fetcher = self._create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last
        )

    def _create_fetcher(self, kind, dataset, auto_collation, collate_fn, drop_last):
        """Jittor version of dataset fetcher"""
        if kind == _DatasetKind.Map:
            def map_fetcher(index):
                if isinstance(index, (list, tuple)):  # batch
                    return [dataset[i] for i in index]
                return dataset[index]
            return map_fetcher
        else:
            def iterable_fetcher(index):
                return next(dataset)
            return iterable_fetcher

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher(index)
        
        # Jittor handles pin_memory automatically
        if self._pin_memory and jt.has_cuda:
            data = jt.array(data).pin_memory()
        return data

class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        
        self._prefetch_factor = loader.prefetch_factor or 2
        self._worker_init_fn = loader.worker_init_fn
        self._shutdown = False
        
        # Jittor uses thread-based parallelism instead of multiprocessing
        self._workers = []
        self._data_queue = queue.Queue(maxsize=self._prefetch_factor * self._num_workers)
        
        # Start worker threads
        for i in range(self._num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        # Initialize task tracking
        self._reset(loader, first_iter=True)

    def _worker_loop(self, worker_id):
        """Jittor version of worker loop using threads"""
        try:
            if self._worker_init_fn is not None:
                self._worker_init_fn(worker_id)
                
            while True:
                # Get next batch index
                try:
                    idx = self._next_index()
                except StopIteration:
                    break
                    
                # Fetch and process data
                try:
                    if self._dataset_kind == _DatasetKind.Map:
                        data = self._dataset[idx]
                    else:  # IterableDataset
                        data = next(self._dataset)
                        
                    if self._collate_fn is not None:
                        data = self._collate_fn(data)
                        
                    # Put result in queue
                    self._data_queue.put((idx, data))
                except Exception as e:
                    self._data_queue.put((idx, ExceptionWrapper(e)))
                    
        except Exception as e:
            warnings.warn(f"DataLoader worker {worker_id} failed: {str(e)}")
            self._data_queue.put((None, ExceptionWrapper(e)))

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._send_idx = 0
        self._rcvd_idx = 0
        self._task_info = {}
        self._tasks_outstanding = 0
        
        # Pre-fetch batches
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _try_get_data(self, timeout=0.1):
        """Non-blocking attempt to get data from queue"""
        try:
            return True, self._data_queue.get(timeout=timeout)
        except queue.Empty:
            # Check if any worker threads died
            for i, worker in enumerate(self._workers):
                if not worker.is_alive():
                    warnings.warn(f"DataLoader worker thread {i} died unexpectedly")
                    self._shutdown = True
                    raise RuntimeError("DataLoader worker thread died")
            return False, None

    def _get_data(self):
        """Get next batch from queue with error handling"""
        while True:
            success, data = self._try_get_data()
            if success:
                return data
            if self._shutdown:
                raise RuntimeError("DataLoader workers shutdown unexpectedly")

    def _next_data(self):
        """Get next batch with out-of-order handling"""
        while True:
            # Check for available data
            if self._rcvd_idx in self._task_info:
                data = self._task_info.pop(self._rcvd_idx)
                self._rcvd_idx += 1
                self._try_put_index()
                return self._process_data(data)
                
            # Get new data
            idx, data = self._get_data()
            if idx == self._rcvd_idx:
                self._rcvd_idx += 1
                self._try_put_index()
                return self._process_data(data)
            else:
                self._task_info[idx] = data

    def _process_data(self, data):
        """Process batch data with error handling"""
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data

    def _try_put_index(self):
        """Schedule next batch to be processed"""
        if self._tasks_outstanding >= self._prefetch_factor * self._num_workers:
            return
            
        try:
            index = self._next_index()
            self._tasks_outstanding += 1
            # In Jittor, workers automatically pick up next batch
            return True
        except StopIteration:
            return False

    def _shutdown_workers(self):
        """Cleanup worker threads"""
        if not self._shutdown:
            self._shutdown = True
            for worker in self._workers:
                if worker.is_alive():
                    worker.join(timeout=0.1)

    def __del__(self):
        self._shutdown_workers()

class ExceptionWrapper:
    """Wrapper for exceptions in worker threads"""
    def __init__(self, exc):
        self.exc = exc
        
    def reraise(self):
        raise self.exc

    