import torch

from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
    kpool_seed_tail_cache,
)


torch.manual_seed(53)
device = torch.device("cuda:0")
pool_size = 4
head_dim = 128


def make_kv_cache():
    return torch.zeros((1, 64, 1, 132), dtype=torch.uint8, device=device)


# Seed must ignore both negative and positive out-of-range physical slots.
seed_tail = torch.zeros(
    (2, 2, pool_size, head_dim), dtype=torch.bfloat16, device=device
)
seed_key = torch.randn((6, head_dim), dtype=torch.bfloat16, device=device)
seed_score = torch.randn_like(seed_key)
seed_slots = torch.tensor([0, 1, 2, 3, 999, -1], dtype=torch.int32, device=device)
kpool_seed_tail_cache(seed_tail, seed_key, seed_score, seed_slots, pool_size)
torch.cuda.synchronize()
torch.testing.assert_close(seed_tail[0, 0], seed_key[:4], rtol=0, atol=0)
torch.testing.assert_close(seed_tail[0, 1], seed_score[:4], rtol=0, atol=0)
assert torch.count_nonzero(seed_tail[1]).item() == 0


# Build an exact reference for a valid completion while two invalid requests
# exercise the bounds checks in the same Triton launch.
ape = torch.randn((pool_size, head_dim), dtype=torch.float32, device=device) * 0.1
tail = torch.zeros((2, 2, pool_size, head_dim), dtype=torch.bfloat16, device=device)
tail[0, :, :3] = seed_tail[0, :, :3]
key = torch.randn((3, 1, head_dim), dtype=torch.bfloat16, device=device)
score = torch.randn_like(key)
tail_slots = torch.tensor([[3], [999], [-1]], dtype=torch.int32, device=device)
slot_mapping = torch.tensor([[0], [1], [2]], dtype=torch.int32, device=device)
positions = torch.tensor([[3], [3], [3]], dtype=torch.int32, device=device)

expected_kv = make_kv_cache()
expected_pool_k = tail[0:1, 0].clone()
expected_pool_s = tail[0:1, 1].clone()
expected_pool_k[0, 3] = key[0, 0]
expected_pool_s[0, 3] = score[0, 0]
kpool_compress_and_write_cache(
    expected_kv,
    expected_pool_k,
    expected_pool_s,
    ape,
    torch.tensor([0], dtype=torch.int64, device=device),
    pool_size,
)

actual_kv = make_kv_cache()
kpool_decode_update_and_maybe_write_cache_batched(
    actual_kv,
    tail,
    tail_slots,
    key,
    score,
    ape,
    slot_mapping,
    positions,
    pool_size,
)
torch.cuda.synchronize()
assert torch.equal(actual_kv, expected_kv)
torch.testing.assert_close(tail[0, 0, 3], key[0, 0], rtol=0, atol=0)
torch.testing.assert_close(tail[0, 1, 3], score[0, 0], rtol=0, atol=0)
assert torch.count_nonzero(tail[1]).item() == 0


# Warm up both launch shapes before capture. Capturing seed and decode proves
# there are no data-dependent-size operations such as torch.nonzero left.
graph_seed_tail = torch.zeros_like(seed_tail)
for _ in range(3):
    kpool_seed_tail_cache(
        graph_seed_tail, seed_key, seed_score, seed_slots, pool_size
    )
torch.cuda.synchronize()
seed_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(seed_graph):
    kpool_seed_tail_cache(
        graph_seed_tail, seed_key, seed_score, seed_slots, pool_size
    )
seed_graph.replay()

graph_kv = make_kv_cache()
graph_tail = torch.zeros_like(tail)
graph_tail[0, :, :3] = seed_tail[0, :, :3]
for _ in range(3):
    kpool_decode_update_and_maybe_write_cache_batched(
        graph_kv,
        graph_tail,
        tail_slots,
        key,
        score,
        ape,
        slot_mapping,
        positions,
        pool_size,
    )
torch.cuda.synchronize()
decode_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(decode_graph):
    kpool_decode_update_and_maybe_write_cache_batched(
        graph_kv,
        graph_tail,
        tail_slots,
        key,
        score,
        ape,
        slot_mapping,
        positions,
        pool_size,
    )
decode_graph.replay()
torch.cuda.synchronize()

assert torch.isfinite(graph_tail.float()).all()
print("KPool bounded Triton seed/decode correctness + CUDA Graph PASS")
