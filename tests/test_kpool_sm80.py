import math

import torch

from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    fwht128_quant_fp8,
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
    kpool_seed_tail_cache,
)
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)


def fwht_reference(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    stride = 1
    while stride < 128:
        grouped = x.reshape(x.shape[0], -1, 2, stride)
        a = grouped[:, :, 0, :]
        b = grouped[:, :, 1, :]
        x = torch.cat((a + b, a - b), dim=-1).reshape(x.shape[0], 128)
        stride *= 2
    return (x / math.sqrt(128)).to(torch.bfloat16).float()


torch.manual_seed(53)
device = torch.device("cuda")
q = torch.randn(33, 128, dtype=torch.bfloat16, device=device)
q_fp8, q_scale = fwht128_quant_fp8(q.contiguous())
rotated = fwht_reference(q)
ref_scale = torch.exp2(
    torch.ceil(torch.log2(torch.clamp(rotated.abs().amax(dim=1), min=1e-4) / 448.0))
).unsqueeze(1)
ref_fp8 = torch.clamp(rotated / ref_scale, -448.0, 448.0).to(
    torch.float8_e4m3fn
)
assert q_fp8.dtype == torch.float8_e4m3fn
torch.testing.assert_close(q_scale, ref_scale, rtol=0, atol=0)
assert torch.equal(q_fp8.view(torch.uint8), ref_fp8.view(torch.uint8))

pool_size = 4
slot_k = torch.randn(2, pool_size, 128, dtype=torch.bfloat16, device=device)
slot_score = torch.randn_like(slot_k) * 0.1
ape = torch.randn(pool_size, 128, dtype=torch.float32, device=device) * 0.1
loc = torch.tensor([0, 1], dtype=torch.int64, device=device)
kv_cache = torch.zeros(1, 64, 1, 132, dtype=torch.uint8, device=device)
compressed_k, compressed_scale = kpool_compress_and_write_cache(
    kv_cache,
    slot_k,
    slot_score,
    ape,
    loc,
    pool_size,
    return_compressed=True,
    write_cache=True,
)
assert compressed_k.dtype == torch.float8_e4m3fn
assert torch.isfinite(compressed_scale).all()

weights = torch.ones(1, 1, dtype=torch.float32, device=device)
ks = torch.zeros(1, dtype=torch.int32, device=device)
ke = torch.full((1,), 2, dtype=torch.int32, device=device)
prefill_logits = fp8_mqa_logits_triton(
    q_fp8[:1].view(1, 1, 128),
    (compressed_k, compressed_scale),
    weights,
    ks,
    ke,
)
assert torch.isfinite(prefill_logits).all()

tail_cache = torch.randn(
    1, 2, pool_size, 128, dtype=torch.bfloat16, device=device
)
seed_key = torch.randn(6, 128, dtype=torch.bfloat16, device=device)
seed_score = torch.randn_like(seed_key)
kpool_seed_tail_cache(
    tail_cache,
    seed_key,
    seed_score,
    torch.tensor([0, 1, 2, 3, 999, -1], dtype=torch.int32, device=device),
    pool_size,
)
kpool_decode_update_and_maybe_write_cache_batched(
    kv_cache,
    tail_cache,
    torch.tensor([[3]], dtype=torch.int32, device=device),
    torch.randn(1, 1, 128, dtype=torch.bfloat16, device=device),
    torch.randn(1, 1, 128, dtype=torch.bfloat16, device=device) * 0.1,
    ape,
    torch.tensor([[0]], dtype=torch.int32, device=device),
    torch.tensor([[3]], dtype=torch.int32, device=device),
    pool_size,
)
decode_logits = fp8_paged_mqa_logits_triton(
    q_fp8[:1].view(1, 1, 1, 128),
    kv_cache,
    weights,
    torch.tensor([1], dtype=torch.int32, device=device),
    torch.tensor([[0]], dtype=torch.int32, device=device),
    max_model_len=1,
)
assert torch.isfinite(decode_logits).all()
print("GLM KPool SM80 encode/prefill/decode checks PASS")
