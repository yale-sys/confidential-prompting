from __future__ import annotations
import torch
from sortedcontainers import SortedList
from safetensors import safe_open
from safetensors.torch import save_file


def get_relocation_map(free_ids: SortedList, allocated_ids: SortedList) -> tuple[list[int], list[int]]:
    free_ids = list(reversed(free_ids))
    allocated_ids = list(allocated_ids)

    relocation_ids = allocated_ids[-len(free_ids):]
    src = []
    dst = []
    while len(free_ids) > 0 and len(relocation_ids) > 0:

        if free_ids[-1] > relocation_ids[-1]:
            break

        src.append(relocation_ids.pop())
        dst.append(free_ids.pop())

    return src, dst


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# token-level attention buffer
class AttentionBuffer:
    num_batch: int
    capacity: int
    num_layers: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: str

    free_indices: SortedList
    used_indices: SortedList

    k: list[torch.Tensor]
    v: list[torch.Tensor]

    #

    def __init__(self, num_batch: int, capacity: int, num_layers: int, num_heads: int, head_dim: int,
                 dtype=torch.float,
                 device: str = "cuda"):
        self.num_batch = num_batch
        self.capacity = capacity
        self.free_indices = SortedList(range(capacity))
        self.used_indices = SortedList()

        self.k = [torch.empty((num_batch, num_heads, capacity, head_dim), dtype=dtype, device=device) for _ in
                  range(num_layers)]
        self.v = [torch.empty((num_batch, num_heads, capacity, head_dim), dtype=dtype, device=device) for _ in
                  range(num_layers)]

        self.dtype = dtype
        self.device = device

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __repr__(self):
        # print used indices series
        return ",".join([str(i) for i in self.used_indices])

    def allocate(self, size: int) -> list[int]:

        if len(self.free_indices) < size:
            raise RuntimeError("Out of buffer capacity")

        # free indices are sorted in ascending order
        allocated = [self.free_indices.pop(0) for _ in range(size)]
        self.used_indices.update(allocated)

        return allocated

    def free(self, indices: list[int]):
        for i in indices:
            self.used_indices.remove(i)
            self.free_indices.add(i)

    def clear(self):
        self.free_indices.clear()
        self.used_indices.clear()
        self.free_indices.update(range(self.capacity))

    def size(self):
        if len(self.used_indices) == 0:
            return 0
        else:
            return self.used_indices[-1] + 1
        
    def memory_consumption(self):
        return self.size() * self.num_layers *self.num_batch * self.num_heads * self.head_dim * 2 * 4 

    def optimize(self):
        src, dst = get_relocation_map(self.free_indices, self.used_indices)

        for i in range(len(src)):
            for j in range(len(self.k)):
                self.k[j][:, :, dst[i], :].copy_(self.k[j][:, :, src[i], :], non_blocking=True)
                self.v[j][:, :, dst[i], :].copy_(self.v[j][:, :, src[i], :], non_blocking=True)

            self.free_indices.add(src[i])
            self.free_indices.remove(dst[i])
            self.used_indices.add(dst[i])
            self.used_indices.remove(src[i])

    def sink(self, layer_id: int, indices: list[int], k: torch.Tensor, v: torch.Tensor):
        # shape of k and v: (1, num_heads, size,  head_dim)
        num_batch, num_heads, size, head_dim = k.shape
        # assert num_batch == 1

        # self.k[layer_id][:, indices, :].copy_(k.squeeze(0), non_blocking=True)
        # self.v[layer_id][:, indices, :].copy_(v.squeeze(0), non_blocking=True)

        # print(k.shape)
        # print(v.shape)

        for i, j in enumerate(indices):
            self.k[layer_id][:, :, j, :].copy_(k[:, :, i, :], non_blocking=True)
            self.v[layer_id][:, :, j, :].copy_(v[:, :, i, :], non_blocking=True)

    def excerpt(self, indices: list[int]) -> AttentionBuffer:

        res = AttentionBuffer(
            num_batch=self.num_batch, 
            capacity=len(indices), 
            num_layers=self.num_layers, 
            num_heads=self.num_heads, 
            head_dim=self.head_dim, 
            dtype=self.dtype,
            device=self.device)
        
        res.allocate(len(indices))
        for i in range(len(self.k)):
            res.k[i][:, :, :, :].copy_(self.k[i][:, :, indices, :], non_blocking=True)
            res.v[i][:, :, :, :].copy_(self.v[i][:, :, indices, :], non_blocking=True)

        return res

    def save(self, path: str):
        # make sure the buffer is optimized before saving
        self.optimize()

        tensors = {}
        for i in range(len(self.k)):
            tensors[f"k_{i}"] = self.k[i][:, :, :self.size(), :].contiguous()
            tensors[f"v_{i}"] = self.v[i][:, :, :self.size(), :].contiguous()

        save_file(tensors, filename=path)

    def load(self, path: str):

        # clear the buffer before loading
        self.clear()

        tensors = {}
        num_batch, num_heads, size, head_dim = 0, 0, 0, 0
        with safe_open(path, framework="pt", device=self.device) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
                if size == 0:
                    num_batch, num_heads, size, head_dim = tensors[k].shape

        assert num_batch == self.num_batch or num_batch == 1
        assert num_heads == self.num_heads
        assert head_dim == self.k[0].shape[3]
        assert size <= self.capacity

        # print(f"Loading buffer of size {size}, head_dim {head_dim}, num_heads {num_heads}")

        self.allocate(size)

        for i in range(len(self.k)):
            self.k[i][:, :, :self.size(), :].copy_(tensors[f"k_{i}"], non_blocking=True)
            self.v[i][:, :, :self.size(), :].copy_(tensors[f"v_{i}"], non_blocking=True)

            # print(torch.sum(self.k[i][:, :self.size(), :]))
            # print(torch.sum(tensors[f"k_{i}"]))
            # print('----')

        # see if the values has been changed
        # for i in range(len(self.k)):
        #     assert torch.allclose(self.k[i][:, :self.size(), :], tensors[f"k_{i}"].to(self.device), atol=1e-5)

    def cache(self, layer_id: int, repeat: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.k[layer_id][:, :, :self.size(), :]
        v = self.v[layer_id][:, :, :self.size(), :]

        if repeat > 1:
            k = repeat_kv(k, repeat)
            v = repeat_kv(v, repeat)
        
        return k, v

    def __len__(self) -> int:
        return self.size()
