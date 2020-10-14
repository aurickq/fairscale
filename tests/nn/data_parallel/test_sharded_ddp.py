# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing OssDdp class.
"""

import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential

from fairscale.nn.data_parallel import ShardedDataParallel

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
skip_if_single_gpu = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multiple GPUs required")


def test_on_cpu():
    run_test(backend=dist.Backend.GLOO, device=torch.device("cpu"), world_size=4)


@skip_if_no_cuda
@skip_if_single_gpu
def test_on_gpu():
    run_test(backend=dist.Backend.NCCL, device=torch.device("cuda"))


def run_one_step(rank, world_size, backend, device, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)

    # Any model works. Add one different buffer per rank
    model = Sequential(Linear(2, 3), Linear(3, 4)).to(device)
    model.register_buffer("test_buffer", torch.ones((1)) * rank)
    model.to(device)

    ddp = ShardedDataParallel(
        module=model,
        optimizer=torch.optim.SGD,
        optimizer_params={"lr": 0.01, "momentum": 0.99},
        world_size=world_size,
        broadcast_buffers=True,
    )
    optimizer = ddp.optimizer
    model = ddp.module

    def check_same_model_params():
        # Check that all the params are the same on all ranks
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                receptacle = [p.clone() for _ in range(world_size)] if rank == 0 else []
                dist.gather(p, receptacle, dst=0)
                if rank == 0:
                    for sync_p in receptacle[1:]:
                        assert torch.all(torch.eq(receptacle[0], sync_p)), "Models differ in between ranks"

        # Check that all the buffers are in sync (authoritative rank is 0, its buffer is 0)
        for b in model.buffers():
            assert b.cpu().item() == 0.0

    # The model should be synchronized in between the ranks at ShardedDataParallel construction time, check that
    check_same_model_params()

    # Optim loop
    def closure():
        optimizer.zero_grad()
        # Different input per rank, allows for checking that the gradients have been properly reduced
        input_tensor = torch.zeros((64, 2)).to(device) if rank == 0 else torch.ones((64, 2)).to(device)

        loss = ddp(input_tensor).abs().sum()
        loss.backward()

        if rank == 0:
            assert optimizer.optim.param_groups[0]["params"][0].grad.sum().item() == 0.0

        ddp.reduce()

        return loss

    # The raw gradients are different in between the ranks, check that they are properly reduced
    # the models should stay the same in between the ranks
    for i in range(10):
        _ = optimizer.step(closure=closure)
        check_same_model_params()

    dist.destroy_process_group()


def run_test(backend, device, world_size=2):
    temp_file_name = tempfile.mkstemp()[1]
    mp.spawn(run_one_step, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)


def run_eval_mode(_unused):
    """ Testing eval mode make sure this is no asserts. """
    dist.init_process_group(
        init_method=f"file://{tempfile.mkstemp()[1]}", backend=dist.Backend.GLOO, rank=0, world_size=1
    )
    model = Sequential(Linear(2, 3), Linear(3, 4))
    optimizer_params = {"lr": 0.1, "momentum": 0.99}
    ddp = ShardedDataParallel(model, torch.optim.SGD, optimizer_params, 1, broadcast_buffers=False)
    optimizer = ddp.optimizer

    ddp.eval()
    for _ in range(5):
        input_tensor = torch.rand((64, 2))
        output = ddp(input_tensor)

    ddp.train()
    try:
        for _ in range(5):
            input_tensor = torch.rand((64, 2))
            output = ddp(input_tensor)
    except RuntimeError:
        pass
    else:
        assert False, "Multiple forward passes on training mode should not pass"

    dist.destroy_process_group()


def test_eval_mode():
    mp.spawn(run_eval_mode, args=(), join=True)
