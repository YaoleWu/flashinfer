"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

flashinfer public-API tests of the Cake Kimi-K3 AttnRes package: the
``flashinfer.kimi_k3_attn_res`` facade.  Its entry points are pass-throughs to
``flashinfer.experimental.cake_kimi_k3_attn_res.cake_backend`` that carry the
experimental marking and accept ``backend="cake"`` only; this module checks that
identity (the same plan and the same bits as the backend, the caller's ``out``
returned, the ``backend`` keyword) and composition with RMSNorm at a BF16 mixture
boundary. Backend-only numerics live in ``test_cake_kimi_k3_attn_res.py``.
"""

import pytest
import torch

import flashinfer.kimi_k3_attn_res as facade
from flashinfer.experimental.cake_kimi_k3_attn_res import cake_backend as backend
from flashinfer.norm import rmsnorm


def _device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    capability = tuple(torch.cuda.get_device_capability(device))
    if backend.SUPPORTED_COMPUTE_CAPABILITIES.get(capability) is None:
        pytest.skip("SM100 / SM103 required")
    return device


def _make_inputs(M, K, *, seed=31000):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = backend.HIDDEN_SIZE
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    return dict(
        prefix=torch.empty((M, hidden), **bf16).uniform_(-1.0, 1.0, generator=g),
        delta=torch.empty((M, hidden), **bf16).uniform_(
            -0.015625, 0.015625, generator=g
        ),
        blocks=torch.empty((M, backend.MAX_BLOCKS, hidden), **bf16).uniform_(
            -1.0, 1.0, generator=g
        ),
        norm_weight=torch.empty(hidden, **bf16).normal_(1.0, 0.05, generator=g),
        qk_weight=torch.empty(hidden, **bf16).normal_(0.0, 0.02, generator=g),
        output_norm_weight=torch.empty(hidden, **bf16).normal_(1.0, 0.05, generator=g),
        out=torch.full((M, hidden), float("nan"), **bf16),
    )


def _clone(inputs):
    return {k: v.clone() for k, v in inputs.items()}


def _assert_same_bits(inputs, mirror):
    for name in ("out", "prefix", "blocks"):
        assert torch.equal(
            inputs[name].view(torch.int16), mirror[name].view(torch.int16)
        ), name


def test_facade_rejects_other_backends():
    _device()
    inputs = _make_inputs(2, 4)
    with pytest.raises(ValueError, match="backend='cake'"):
        facade.prepare_kimi_k3_attn_res(**inputs, num_blocks=4, backend="triton")
    with pytest.raises(ValueError, match="backend='cake'"):
        facade.kimi_k3_attn_res(**inputs, num_blocks=4, backend="triton")


def test_facade_prepare_passes_through_to_the_backend():
    device = _device()
    M, K = 64, 4
    if not backend.generated_program_available(device, M, K):
        pytest.skip("program not registered for this cell")
    assert facade.generated_program_available(
        device, M, K
    ) == backend.generated_program_available(device, M, K)
    inputs = _make_inputs(M, K)
    mirror = _clone(inputs)
    runner = facade.prepare_kimi_k3_attn_res(**inputs, num_blocks=K)
    expected = backend.prepare_kimi_k3_attn_res(**mirror, num_blocks=K)
    assert type(runner) is backend.KimiK3AttnResRunner
    assert runner.plan == expected.plan
    assert runner.launch() is inputs["out"]
    expected.launch()
    torch.cuda.synchronize()
    _assert_same_bits(inputs, mirror)


@pytest.mark.parametrize("pdl", [False, True])
def test_facade_one_shot_call_matches_the_backend(pdl):
    device = _device()
    M, K = 64, 4
    if not backend.generated_program_available(device, M, K, enable_pdl=pdl):
        pytest.skip("program not registered for this cell")
    inputs = _make_inputs(M, K, seed=31001)
    mirror = _clone(inputs)
    out = facade.kimi_k3_attn_res(**inputs, num_blocks=K, enable_pdl=pdl)
    assert out is inputs["out"]
    backend.kimi_k3_attn_res(**mirror, num_blocks=K, enable_pdl=pdl)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    _assert_same_bits(inputs, mirror)


def test_bf16_mixture_then_rmsnorm_in_eager_and_graph():
    """Compose the existing APIs with an explicit BF16 pre-norm boundary."""
    device = _device()
    M, K = 2, 8
    if not facade.generated_program_available(
        device, M, K, common_path=False, apply_output_norm=False
    ):
        pytest.skip("no-output-norm program not registered for this cell")
    inputs = _make_inputs(M, K)
    seed = (
        torch.arange(backend.HIDDEN_SIZE, device=device)
        .remainder(31)
        .sub(15)
        .bfloat16()
        .expand(M, -1)
        .contiguous()
    )
    inputs["prefix"].copy_(seed)
    inputs["delta"].fill_(1.0)
    inputs["blocks"].zero_()
    inputs["norm_weight"].fill_(1.0)
    inputs["qk_weight"].zero_()
    weight = inputs["output_norm_weight"]
    weight.fill_(1.0)
    inputs["output_norm_weight"] = None
    scratch, out = inputs["out"], torch.empty_like(seed)
    runner = facade.prepare_kimi_k3_attn_res(**inputs, num_blocks=K)

    def launch():
        runner.launch()
        rmsnorm(scratch, weight, eps=1e-5, out=out, enable_pdl=False)

    launch()  # Load both programs before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for run in (launch, graph.replay):
        for sign in (1, -1):
            current = seed * sign
            inputs["prefix"].copy_(current)
            scratch.fill_(float("nan"))
            out.fill_(float("nan"))
            updated = (current.float() + 1.0).bfloat16()
            # Uniform scores, eight zero snapshots: the exact mixture is prefix / 9.
            expected_mix = (updated.double() / (K + 1)).bfloat16()
            rounded = expected_mix.double()
            expected = (
                rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-5)
            ).bfloat16()
            run()
            torch.cuda.synchronize()
            assert torch.equal(scratch, expected_mix)
            torch.testing.assert_close(out, expected, rtol=1 / 128, atol=1 / 64)
            assert torch.equal(inputs["prefix"], updated)
            assert torch.count_nonzero(inputs["blocks"]) == 0
