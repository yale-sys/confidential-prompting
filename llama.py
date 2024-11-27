import math
import time
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig

from attention import AttentionBuffer
import logger

from concurrent.futures import ThreadPoolExecutor, as_completed
executor = ThreadPoolExecutor(max_workers=64)  # Set max workers based on your requirements

def wait_for_works_with_deadline(works, multiplier=1.5):
    future_to_worker = {executor.submit(work.wait): worker_id for worker_id, work in enumerate(works)}
    completed_times = []
    timeout = None

    start_time = time.time()
    
    # process each work completion
    for _ in as_completed(future_to_worker):
        completion_time = time.time()
        completed_times.append(completion_time)

        # Set the timeout when the first work completes
        if len(completed_times) == 1:
            timeout = completion_time + ((completion_time-start_time) * multiplier)

        # Check for timeout exceeded
        if timeout and time.time() > timeout:
            #print("Deadline exceeded. Ignoring remaining works.")
            break  # Stop processing lagging works

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _compute_default_rope_parameters(
    base:int, dim:int, device:torch.device
) -> torch.Tensor:
   

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq


def _compute_llama3_parameters(
    base:int,
    dim:int,
    factor:int,
    low_freq_factor:int,
    high_freq_factor:int,
    old_context_len:int,
    device:torch.device,
) -> torch.Tensor:
   
    inv_freq = _compute_default_rope_parameters(base, dim, device)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()

        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        #print( config.rope_scaling)
        if config.rope_scaling is not None and config.rope_scaling["rope_type"] == "llama3":
            inv_freq = _compute_llama3_parameters(
                base=self.base,
                dim=self.dim,
                factor=config.rope_scaling["factor"],
                low_freq_factor=config.rope_scaling["low_freq_factor"],
                high_freq_factor=config.rope_scaling["high_freq_factor"],
                old_context_len=config.rope_scaling["original_max_position_embeddings"],
                device=device,
            )
        
        else:
            inv_freq = _compute_default_rope_parameters(self.base, self.dim, device)
            #inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, device=self.inv_freq.device
        )

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        # print(self.max_seq_len_cached, device, self.inv_freq.dtype)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].float(), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].float(), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(x.dtype),
        )
    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb1(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed



class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.config
        )
        
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None = None,
            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
            layer_id: int | None = 0,

    ) -> torch.Tensor:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if buffer is not None:
            kv_seq_len = len(buffer)

        cos, sin = self.rotary_emb(value_states, seq_len=torch.max(position_ids) + 1)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if buffer is not None:
            buffer.sink(layer_id, buffer_sink_ids, key_states, value_states)
            key_states, value_states = buffer.cache(layer_id)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output

def softmax(x: torch.Tensor, dim: int):
    m = torch.max(x, dim=dim, keepdim=True).values
    up = torch.exp((x - m).float())
    down = torch.sum(up, dim=dim, keepdim=True)
    return m, down.to(x.dtype), (up / down).to(x.dtype)


class ConfidentialLlamaAttention(LlamaAttention):
    pvt_buffer: torch.Tensor | None

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.pvt_buffer = None

    def forward(self, hidden_states: torch.Tensor,
                position_ids: torch.LongTensor,
                attention_mask: torch.Tensor | None = None,
                buffer: AttentionBuffer | None = None,
                buffer_sink_ids: list[int] | None = None,
                layer_id: int | None = 0,
                confidential: bool = False,
                num_users: int = 1
            ):
        
        logger.start_measure()
        bsz, q_len, _ = hidden_states.size()

        if self.pvt_buffer is None:
            self.pvt_buffer = torch.empty((bsz, self.num_heads, 1, self.head_dim + 2), dtype=torch.half)
            self.pvt_buffer_gpu = torch.empty_like(self.pvt_buffer, device=buffer.device)

        q_new = self.q_proj(hidden_states)
        
        q_new = q_new.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # print the

        cos, sin = self.rotary_emb(q_new, seq_len=torch.max(position_ids) + 1)
        
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # print('cos', cos.shape)
        # print('sin', sin.shape)
        # print('position_ids', position_ids.shape)
        # print('q_new', q_new.shape)
        # print('k_new', k_new.shape)
        # print('v_new', v_new.shape)

        q_new = apply_rotary_pos_emb1(q_new, cos, sin, position_ids)
        q_new_cpu = q_new.half().cpu()
        
        
        works = []
        if confidential:
            # compute the private states asynchronously.
            # send before view() because send only works with contiguous tensors.
            # print(torch.sum(q_new))
            batch_per_user = bsz // num_users
            for i in range(num_users):
                
                # check if nan
                
                
                #print('send shape', q_new_cpu[i * batch_per_user:(i + 1) * batch_per_user].shape)
                #print('recv shape', self.pvt_buffer[i * batch_per_user:(i + 1) * batch_per_user].shape)
                torch.distributed.isend(q_new_cpu[i * batch_per_user:(i + 1) * batch_per_user], 1 + i)
                # print('recv shape', self.pvt_buffer[i * batch_per_user:(i + 1) * batch_per_user].shape)
                work = torch.distributed.irecv(self.pvt_buffer[i * batch_per_user:(i + 1) * batch_per_user], 1 + i)
                #works.append((work, time.time()))
                works.append(work)
                
            logger.log_measure('comm_overhead1')
            
        
        k_new = self.k_proj(hidden_states)
        v_new = self.v_proj(hidden_states)
        k_new = k_new.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        k_new = apply_rotary_pos_emb1(k_new, cos, sin, position_ids)

        # compute the public states synchronously
        buffer.sink(layer_id, buffer_sink_ids, k_new, v_new)
        k_pub, v_pub = buffer.cache(layer_id, self.num_key_value_groups)
        logger.log_measure('qkv_projection')

        
        # print dtypes
        # print('q_new', q_new.dtype)
        # print('k_pub', k_pub.dtype)
        # print('v_pub', v_pub.dtype)
        # print('attention_mask', attention_mask.dtype)
        # print('buffer', self.pvt_buffer.dtype)
        # print('buffer_sink_ids', buffer_sink_ids)
        # print('layer_id', layer_id)
        
        score_pub = torch.matmul(q_new, k_pub.transpose(2, 3)) / math.sqrt(self.head_dim)

        
        
        if attention_mask is not None:
            score_pub = score_pub + attention_mask

        max_pub, sum_pub, score_pub = softmax(score_pub, dim=-1)
        
        attn_pub = torch.matmul(score_pub.to(q_new.dtype), v_pub)
        logger.log_measure('pub_attention')
        if confidential:
            # wait for the private states to be computed
            #
            
            ############################################################################################################
                
            #wait_for_works_with_deadline(works, 1.5)
            
            for work in works:
                work.wait()
            ############################################################################################################

                
                
            self.pvt_buffer_gpu.copy_(self.pvt_buffer)
            logger.log_measure('comm_overhead2')

            aa = self.pvt_buffer_gpu if hidden_states.dtype == torch.half else self.pvt_buffer_gpu.float()

            max_pvt, sum_pvt, attn_pvt = torch.split(aa, [1, 1, self.head_dim], dim=-1)
            # combine the public and private states
            alpha = torch.exp(max_pvt - max_pub)

            # print('q_new', torch.sum(q_new))
            # print('max_pvt', torch.sum(max_pvt))
            # print('sum_pvt', torch.sum(sum_pvt))
            # print('attn_pvt', torch.sum(attn_pvt))
            # #
            # exit()
            c_pvt = sum_pvt / (sum_pvt + sum_pub / alpha)
            c_pub = sum_pub / (sum_pub + sum_pvt * alpha)

            attn = c_pvt * attn_pvt + c_pub * attn_pub
            
            
            logger.log_measure('pvt_attention')
        else:
            attn = attn_pub

        # project the output and return
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.reshape(bsz, q_len, self.hidden_size)
        attn = self.o_proj(attn)
        
        logger.log_measure('o_projection')
        #exit()
        
        return attn


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = ConfidentialLlamaAttention(config=config)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None = None,
            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
            layer_id: int | None = 0,
            confidential: bool = False,
            num_users: int = 1
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            buffer=buffer,
            buffer_sink_ids=buffer_sink_ids,
            layer_id=layer_id,
            confidential=confidential,
            num_users=num_users
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlamaModel(LlamaPreTrainedModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None = None,
            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
            confidential: bool = False,
            num_users: int = 1,
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape

        position_ids = position_ids.view(-1, seq_length).long()
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if buffer is not None:
            seq_length_with_past = len(buffer)
            past_key_values_length = len(buffer) - seq_length

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )

            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        else:
            inverted_mask = 1.0 - attention_mask.float()
            attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool),
                                                       torch.finfo(inputs_embeds.dtype).min)

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            
            #print('idx', idx)
            #print('hidden_states', hidden_states.)

            # check if there is any nan in hidden states 
            # if torch.isnan(hidden_states).any():
            #     print('idx', idx)
            
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                buffer=buffer,
                buffer_sink_ids=buffer_sink_ids,
                layer_id=idx,
                confidential=confidential,
                num_users=num_users
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None = None,
            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
            confidential: bool = False,
            num_users: int = 1,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            buffer=buffer,
            buffer_sink_ids=buffer_sink_ids,
            confidential=confidential,
            num_users=num_users
        )

        global executor
        

        logits = self.lm_head(hidden_states).float()

        #executor.shutdown(wait=False)
        
        return logits
