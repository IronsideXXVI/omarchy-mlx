#!/usr/bin/env python3
"""Parity tests for the topk/multinomial lowerings in mlx_vulkan.

Runs under plain python3 (no pytest needed) or pytest.
Requires torch + mlx on the path; run from this directory or set
PYTHONPATH to it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _np(a):
    return np.array(a)


def test_topk():
    rng = np.random.RandomState(0)
    x = rng.randn(4, 7).astype(np.float32)
    for k, dim in ((1, 0), (3, 0), (1, 1), (3, 1), (7, 1), (2, -1), (3, None)):
        vals, idx = mlx_vulkan._topk(mx.array(x), k, dim)
        tv, ti = torch.topk(torch.tensor(x), k, -1 if dim is None else dim)
        assert np.allclose(_np(vals), tv.numpy()), (k, dim)
        assert np.array_equal(_np(idx), ti.numpy()), (k, dim)
    # largest=False
    vals, idx = mlx_vulkan._topk(mx.array(x), 3, 1, largest=False)
    tv, ti = torch.topk(torch.tensor(x), 3, 1, largest=False)
    assert np.allclose(_np(vals), tv.numpy())
    assert np.array_equal(_np(idx), ti.numpy())
    # 3-D, middle dim
    x3 = rng.randn(2, 5, 3).astype(np.float32)
    vals, idx = mlx_vulkan._topk(mx.array(x3), 2, 1)
    tv, ti = torch.topk(torch.tensor(x3), 2, 1)
    assert np.allclose(_np(vals), tv.numpy())
    assert np.array_equal(_np(idx), ti.numpy())
    print("test_topk PASS")


def test_multinomial():
    rng = np.random.RandomState(0)
    # peaked dist: sample must be the argmax row-wise (deterministic check)
    probs = np.full((8, 16), 1e-4, np.float32)
    peak = np.array([3, 9, 0, 15, 7, 1, 12, 5])
    probs[np.arange(8), peak] = 0.99
    out = mlx_vulkan._multinomial(mx.array(probs), 1)
    assert np.array_equal(_np(out).reshape(-1), peak), _np(out)
    assert _np(out).dtype == np.int64
    # shape/dtype: n>1 without replacement gives n distinct in-range draws
    probs2 = rng.rand(4, 32).astype(np.float32)
    out2 = np.array(mlx_vulkan._multinomial(mx.array(probs2), 5))
    assert out2.shape == (4, 5) and out2.dtype == np.int64
    assert out2.min() >= 0 and out2.max() < 32
    for row in out2:
        assert len(set(row.tolist())) == 5  # without replacement: distinct
    # replacement=True: shape only (draws may repeat)
    out3 = np.array(mlx_vulkan._multinomial(mx.array(probs2), 5, replacement=True))
    assert out3.shape == (4, 5) and out3.dtype == np.int64
    print("test_multinomial PASS")


def test_dispatch_registered():
    assert torch.topk in mlx_vulkan.FUNC_TABLE
    assert torch.multinomial in mlx_vulkan.FUNC_TABLE
    assert "topk" in mlx_vulkan.ATEN_TABLE
    assert "multinomial" in mlx_vulkan.ATEN_TABLE
    print("test_dispatch_registered PASS")


def test_where_masked_fill():
    # PT-AA: aten.where.self (PAR-41 census rank 25, 900 uses, already
    # lowered) and aten.masked_fill.Tensor/Scalar/_ (registered but no
    # Tensor-variant coverage). Prove lowering + dtype promotion + parity
    # at fp32/fp16/bf16 with a shared bool-mask digest.
    cond = torch.tensor([True, False, True, False])
    x32, y32 = torch.tensor([1.0, -2.0, 3.5, -4.0]), torch.tensor([0.5, 2.0, -1.0, 4.0])
    cases = [
        (torch.ops.aten.where.self, (cond, x32, y32)),
        (torch.ops.aten.where.self, (cond, x32, torch.tensor([1, 2, 3, 4]))),
        (torch.ops.aten.where.self, (cond, x32.half(), y32)),
        (torch.ops.aten.where.self, (cond, x32.bfloat16(), y32)),
        (torch.ops.aten.where.self, (cond, torch.tensor([1, 2, 3, 4]),
                                    torch.tensor([5, 6, 7, 8], dtype=torch.int32))),
        (torch.ops.aten.masked_fill.Scalar, (x32, cond, 9.0)),
        (torch.ops.aten.masked_fill.Tensor, (x32, cond, torch.tensor(9.0))),
        (torch.ops.aten.masked_fill.Scalar, (x32.bfloat16(), cond, 2.5)),
        (torch.ops.aten.masked_fill.Scalar, (x32.half(), cond, 2.5)),
        (torch.ops.aten.masked_fill_.Scalar, (x32.clone(), cond, 9.0)),
        (torch.ops.aten.masked_fill.Scalar, (torch.randn(2, 3),
                                             torch.tensor([[True], [False]]), 7.0)),
    ]
    for target, ins in cases:
        ref = target(*ins)
        g = torch.fx.Graph()
        targs = [a for a in ins if torch.is_tensor(a)]
        phs = [g.placeholder(f"p{i}") for i in range(len(targs))]
        pit = iter(phs)
        args = tuple(next(pit) if torch.is_tensor(a) else a for a in ins)
        n = g.call_function(target, args, {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*targs)
        assert stats["cpu"] == [], (stats, str(target))
        fam = "where" if "where" in str(target) else "masked_fill"
        assert any(fam in s for s in stats["mlx"]), (stats, str(target))
        got = torch.as_tensor(out)
        assert got.dtype == ref.dtype, (str(target), got.dtype, ref.dtype)
        assert torch.equal(got.float(), ref.float()), (str(target), got, ref)
    print("test_where_masked_fill PASS")


if __name__ == "__main__":
    test_topk()
    test_multinomial()
    test_dispatch_registered()
    test_where_masked_fill()
    print("ALL PASS")
