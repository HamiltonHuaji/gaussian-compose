import os
import torch
import slangtorch
import torch.nn.functional as F

from typing import *
from .utils import quaternion_raw_multiply, quaternion_invert

transform_module = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "transform.slang"))

class _transform_slang_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trans, rotors, means, quats, indices):
        trans, rotors, means, quats, indices = trans.contiguous(), rotors.contiguous(), means.contiguous(), quats.contiguous(), indices.contiguous()

        ctx.save_for_backward(trans, rotors, means, quats, indices)
        result_means, result_quats = transform_module.indexed_transform_wxyz_fwd(trans, rotors, means, quats, indices)

        return result_means, result_quats

    @staticmethod
    def backward(ctx, dresult_means, dresult_quats):
        trans, rotors, means, quats, indices = ctx.saved_variables
        return transform_module.indexed_transform_wxyz_bwd(trans, rotors, means, quats, indices, dresult_means.clone(), dresult_quats.clone()) + (None,)

def _indexed_transform_torch(trans, rotors, means, quats, indices):
    rotors = torch.nn.functional.normalize(rotors, p=2, dim=-1)
    quats = torch.nn.functional.normalize(quats, p=2, dim=-1)

    q = rotors[indices.squeeze(-1), ...]
    q_inv = quaternion_invert(q)
    qv = torch.cat([torch.zeros_like(means[:, :1]), means], dim=-1)

    result_means = quaternion_raw_multiply(quaternion_raw_multiply(q, qv), q_inv)[..., 1:4] + trans[indices.squeeze(-1), ...]
    result_quats = quaternion_raw_multiply(rotors[indices.squeeze(-1), ...], quats)
    return result_means, result_quats

_indexed_transform_torch_opt = torch.compile(_indexed_transform_torch)

def indexed_transform(trans: torch.Tensor, rotors: torch.Tensor, means: torch.Tensor, quats: torch.Tensor, indices: torch.Tensor, implementation="torch") -> Tuple[torch.Tensor, torch.Tensor]:
    if implementation=="slang":
        return _transform_slang_func.apply(trans, rotors, means, quats, indices)
    elif implementation=="torch":
        return _indexed_transform_torch_opt(trans, rotors, means, quats, indices)
    elif implementation=="torch_raw":
        return _indexed_transform_torch(trans, rotors, means, quats, indices)
    else:
        raise NotImplementedError(f"Unknown implementation {implementation}")

def _prepare_inputs(seed=42, n=1, m=16384, return_dict=False):
    torch.manual_seed(seed)

    means = torch.randn(m, 3).cuda().requires_grad_(True)
    quats = torch.randn(m, 4).cuda().requires_grad_(True)
    trans = torch.randn(n, 3).cuda().requires_grad_(True)
    rotors = torch.randn(n, 4).cuda().requires_grad_(True)
    indices = torch.randint(0, n, (m, 1)).cuda().to(torch.int32)

    if return_dict:
        return {
            "means": means,
            "quats": quats,
            "trans": trans,
            "rotors": rotors,
            "indices": indices
        }
    else:
        return trans, rotors, means, quats, indices

def _expand_outputs(trans, rotors, means, quats, indices, implementation="slang"):
    result_means, result_quats = indexed_transform(trans, rotors, means, quats, indices, implementation=implementation)
    loss = result_means.mean() + result_quats.mean()
    loss.backward()
    return {
        "result_means": result_means,
        "result_quats": result_quats,
        "trans_grad": trans.grad,
        "rotors_grad": rotors.grad,
        "means_grad": means.grad,
        "quats_grad": quats.grad,
    }

def _test_once(seed=42, n=10, m=5):
    results = {
        "slang": _expand_outputs(*_prepare_inputs(seed, n, m, return_dict=False), implementation="slang"),
        "torch": _expand_outputs(*_prepare_inputs(seed, n, m, return_dict=False), implementation="torch"),
    }

    if not torch.allclose(results["slang"]["result_means"], results["torch"]["result_means"], atol=1e-6, rtol=1e-4):
        print(results["slang"]["result_means"])
        print(results["torch"]["result_means"])
        print(torch.abs(results["slang"]["result_means"] - results["torch"]["result_means"]).max())
        raise AssertionError("result_means")
    
    if not torch.allclose(results["slang"]["result_quats"], results["torch"]["result_quats"], atol=1e-6, rtol=1e-4):
        print(results["slang"]["result_quats"])
        print(results["torch"]["result_quats"])
        print(torch.abs(results["slang"]["result_quats"] - results["torch"]["result_quats"]).max())
        raise AssertionError("result_quats")

    if not torch.allclose(results["slang"]["trans_grad"], results["torch"]["trans_grad"], atol=1e-6, rtol=1e-4):
        print(results["slang"]["trans_grad"])
        print(results["torch"]["trans_grad"])
        print(torch.abs(results["slang"]["trans_grad"] - results["torch"]["trans_grad"]).max())
        raise AssertionError("trans_grad")
    
    if not torch.allclose(results["slang"]["rotors_grad"], results["torch"]["rotors_grad"], atol=1e-6, rtol=1e-4):
        print(results["slang"]["rotors_grad"])
        print(results["torch"]["rotors_grad"])
        print(torch.abs(results["slang"]["rotors_grad"] - results["torch"]["rotors_grad"]).max())
        raise AssertionError("rotors_grad")

    if not torch.allclose(results["slang"]["means_grad"], results["torch"]["means_grad"], atol=1e-6, rtol=1e-4):
        print(results["slang"]["means_grad"])
        print(results["torch"]["means_grad"])
        print(torch.abs(results["slang"]["means_grad"] - results["torch"]["means_grad"]).max())
        raise AssertionError("means_grad")
    
    if not torch.allclose(results["slang"]["quats_grad"], results["torch"]["quats_grad"], atol=1e-6, rtol=1e-4):
        print(results["slang"]["quats_grad"])
        print(results["torch"]["quats_grad"])
        print(torch.abs(results["slang"]["quats_grad"] - results["torch"]["quats_grad"]).max())
        raise AssertionError("quats_grad")

def test():
    import random
    from tqdm.auto import trange, tqdm
    pbar = trange(16384)

    for seed in pbar:
        n, m = random.randint(1, 32), random.randint(1, 16384 * 32)
        pbar.set_description(f"seed={seed}, n={n}, m={m}")
        _test_once(seed, n, m)

def profile(n=10, m=16384*32, implementation="slang"):
    import random
    from tqdm.auto import trange, tqdm

    trans, rotors, means, quats, indices = _prepare_inputs(n=n, m=m, return_dict=False)

    _expand_outputs(trans, rotors, means, quats, indices, implementation=implementation)
    torch.cuda.synchronize()

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    pbar = trange(8192)
    for seed in pbar:
        iter_start.record()

        _expand_outputs(trans, rotors, means, quats, indices, implementation=implementation)

        iter_end.record()

    torch.cuda.synchronize()

    # print results
    print(f"Time: {iter_start.elapsed_time(iter_end) / 16384 * 1000 * 1000:.2f}ns")