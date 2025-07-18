import torch
import torch.utils.data as torchdata
from torch.utils.data.distributed import DistributedSampler
import logging
from functools import partial

from .transform.collate import collate
#from ..config import configurable
from ..utils.comm import get_world_size
from ..utils.env import seed_all_rng
from ..evaluation.vil_utils import CustomBatchSampler,CustomSamper

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)

def build_batch_data_loader(
    dataset,
    total_batch_size,
    *,
    num_workers=0,
    collate_fn=None,
    drop_last: bool = True,
    single_gpu_batch_size=None,
    prefetch_factor=2,
    persistent_workers=False,
    pin_memory=False,
    seed=None,
    shuffle = True,
    **kwargs,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.
        single_gpu_batch_size: You can specify either `single_gpu_batch_size` or `total_batch_size`.
            `single_gpu_batch_size` specifies the batch size that will be used for each gpu/process.
            `total_batch_size` allows you to specify the total aggregate batch size across gpus.
            It is an error to supply a value for both.
        drop_last (bool): if ``True``, the dataloader will drop incomplete batches.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    if world_size == 1:
        # if total_batch_size:
        #     raise ValueError(
        #         """total_batch_size and single_gpu_batch_size are mutually incompatible.
        #         Please specify only one. """
        #     )
        batch_size = total_batch_size
        sampler = None
    else:
        world_size = get_world_size()
        rank = get_rank()
        assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
        ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
            total_batch_size, world_size
        )
        batch_size = total_batch_size // world_size
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False  
    logger = logging.getLogger(__name__)
    logger.info("Making batched data loader with batch_size=%d", batch_size)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=partial(collate,samples_per_gpu=batch_size),
        # collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        worker_init_fn=worker_init_reset_seed,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        generator=generator,
        shuffle=shuffle,
        **kwargs,
    )

def build_batchvil_test_data_loader(
    dataset,
    total_batch_size,
    *,
    num_workers=0,
    collate_fn=None,
    drop_last: bool = True,
    single_gpu_batch_size=None,
    prefetch_factor=2,
    persistent_workers=False,
    pin_memory=False,
    seed=None,
    shuffle = True,
    **kwargs,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.
        single_gpu_batch_size: You can specify either `single_gpu_batch_size` or `total_batch_size`.
            `single_gpu_batch_size` specifies the batch size that will be used for each gpu/process.
            `total_batch_size` allows you to specify the total aggregate batch size across gpus.
            It is an error to supply a value for both.
        drop_last (bool): if ``True``, the dataloader will drop incomplete batches.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    if single_gpu_batch_size:
        if total_batch_size:
            raise ValueError(
                """total_batch_size and single_gpu_batch_size are mutually incompatible.
                Please specify only one. """
            )
        batch_size = single_gpu_batch_size
    else:
        world_size = get_world_size()
        assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
        ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
            total_batch_size, world_size
        )
        batch_size = total_batch_size // world_size
    logger = logging.getLogger(__name__)
    logger.info("Making batched data loader with batch_size=%d", batch_size)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    sampler = CustomSamper(dataset)
    bs = CustomBatchSampler(sampler,batch_size,False)

    return torchdata.DataLoader(
        dataset,
        batch_sampler=bs,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=partial(collate,samples_per_gpu=batch_size),
        # collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        worker_init_fn=worker_init_reset_seed,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        generator=generator,
        shuffle=shuffle,
        **kwargs,
    )
