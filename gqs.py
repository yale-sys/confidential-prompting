from __future__ import annotations

import math
import random
import re

import numpy as np
import torch
import copy
from transformers import AutoTokenizer

from attention import AttentionBuffer


SPIECE_UNDERLINE = 29871
SEQ_PAD = 4


class Seq:
    token_ids: list[int]
    pos_ids: list[int]
    attn_ids: list[int]

    def __init__(self, token_ids: list[int] = None, pos_ids: list[int] = None, attn_ids: list[int] = None):
        self.token_ids = token_ids if token_ids is not None else []
        self.pos_ids = pos_ids if pos_ids is not None else []
        self.attn_ids = attn_ids if attn_ids is not None else []

    def clear(self):
        self.token_ids.clear()
        self.pos_ids.clear()
        self.attn_ids.clear()

    def __len__(self):
        return len(self.token_ids)


class Sample:
    completed: Seq
    pending: Seq
    prob: float

    def __init__(self):
        self.completed = Seq()
        self.pending = Seq()
        self.prob = 1.0

    def get_inputs(self) -> tuple[list[int], list[int], list[int], np.ndarray]:
        attn_ids = self.completed.attn_ids + self.pending.attn_ids

        mask = np.zeros((len(self.pending.token_ids), max(attn_ids) + 1), dtype=np.bool_)
        mask[:, self.completed.attn_ids] = True

        for i, (pos_id, attn_id) in enumerate(zip(self.pending.pos_ids, self.pending.attn_ids)):
            for pos_id_, attn_id_ in zip(self.pending.pos_ids, self.pending.attn_ids):
                if pos_id_ <= pos_id:
                    mask[i, attn_id_] = True

        return self.pending.token_ids, self.pending.pos_ids, self.pending.attn_ids, mask

    def complete(self):
        self.completed.token_ids.extend(self.pending.token_ids)
        self.completed.pos_ids.extend(self.pending.pos_ids)
        self.completed.attn_ids.extend(self.pending.attn_ids)
        self.pending.clear()

    def clone(self) -> Sample:
        return copy.deepcopy(self)


def filter_probs_abs(probs, tgt_prob, num_intervals):
    lower = math.floor(tgt_prob * num_intervals) / num_intervals + 1e-3  # to prevent 0.0
    upper = lower + 1.0 / num_intervals
    
    qualifying_token_ids = ((probs >= lower) & (probs < upper)).nonzero().squeeze(-1).cpu().numpy()

    return qualifying_token_ids

def filter_probs_rel(probs, tgt_prob, magnitude:int = 1):
    
    rel_dist = torch.abs(probs/tgt_prob - 1)
    
    qualifying_token_ids = (rel_dist <= magnitude).nonzero().squeeze(-1).cpu().numpy()
    
    return qualifying_token_ids



class GreedyQuantizedSampler:
    tokenizer: AutoTokenizer
    model: callable

    buffer: AttentionBuffer
    temperature: float
    top_k: int

    def __init__(self, tokenizer, model, buffer: AttentionBuffer, temperature=1.0):
        self.tokenizer = tokenizer
        self.model = model
        self.buffer = buffer
        self.temperature = temperature

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_id: list[int] | int) -> str:
        return self.tokenizer.decode(token_id, skip_special_tokens=True)

    @torch.inference_mode()
    def sample(self, gamma: int, prefix: str, suffix: str, num_intervals=10, dist:str='rel') -> list[tuple[str, float]]:

        # clear the buffer
        self.buffer.clear()

        # strip special tokens
        prefix = strip_special_tokens(self.tokenizer, prefix)
        
        prefix_token_ids = self.tokenizer.encode(prefix)
        #suffix_so_token_ids = self.encode(suffix)
        combined_token_ids = self.encode(prefix + " " + suffix)
        suffix_token_ids = combined_token_ids[len(prefix_token_ids)-1:] 

        
        #print(f"prefix: {self.tokenizer.decode(prefix_token_ids)}<end>")
        #print(f"suffix: {self.tokenizer.decode(suffix_token_ids)}<end>")
        #print(f"real suffix: {self.tokenizer.decode(real_suffix_token_ids)}<end>")
        #print(f"combin: {self.tokenizer.decode(combined_token_ids)}<end>")
        
        #print(prefix_token_ids)
        #print(suffix_token_ids)
        #print(suffix_token_ids)
        #print(f"suffix: {self.tokenizer.decode(suffix_token_ids)}<end>")
        #print(f"suffix: {self.tokenizer.decode(suffix_token_ids[1:])}<end>")

        
        prefix_offset = len(prefix_token_ids)
        original_suffix = suffix
        root = Sample()
        root.pending.token_ids = prefix_token_ids
        root.pending.pos_ids = list(range(len(prefix_token_ids)))
        root.pending.attn_ids = self.buffer.allocate(len(prefix_token_ids))

        max_token_len = len(suffix_token_ids)
        done_list = []

        ref = root
        candidates = []

        for i in range(max_token_len):
            if ref is not None:
                candidates = [ref] + candidates

            if len(candidates) == 0:
                break

            input_ids = []
            position_ids = []
            sink_attn_ids = []
            attn_mask = np.zeros((1, 1,
                                  sum(len(can.pending) for can in candidates),
                                  len(self.buffer)),
                                 dtype=np.bool_)

            # sort candidates so that target comes first
            offset = 0
            for can in candidates:
                token_ids, pos_ids, attn_ids, mask = can.get_inputs()

                # print(self.decode(can.completed.token_ids + can.pending.token_ids))
                input_ids.extend(token_ids)
                position_ids.extend(pos_ids)
                attn_mask[0, 0, offset:offset + len(token_ids), :len(mask[0])] = mask
                sink_attn_ids.extend(attn_ids)
                offset += len(token_ids)

            #print(f"input_ids: {input_ids}")
            #print(f"position_ids: {position_ids}")
            #print(f"attn_mask: {attn_mask}")
            
            logits = self.model(
                input_ids=torch.as_tensor([input_ids], device=self.model.device, dtype=torch.long),
                position_ids=torch.as_tensor([position_ids], device=self.model.device, dtype=torch.long),
                attention_mask=torch.as_tensor(attn_mask, device=self.model.device, dtype=torch.bool),
                buffer_sink_ids=sink_attn_ids,
                buffer=self.buffer,
            )

            # convert logits to probabilities
            probs = torch.nn.functional.softmax(logits / self.temperature, dim=-1).squeeze(0)

            lower = 0.0
            upper = 1.0

            # # select top-10 tokens
            # probs10, indices10 = torch.topk(probs[-1,:], k=10, dim=-1)
            
            # # decode the top-10 tokens
            # for j in range(10):
            #     print(f"top-{j}: {self.decode(indices10[j])}, prob: {probs10[j]}")
            
            
            if ref is not None:
                tgt = candidates[0]
                tgt_token_prob = probs[len(tgt.pending) - 1, suffix_token_ids[i]].cpu().item()
                
                #print(f"tgt: {self.decode(suffix_token_ids[i])}, prob: {tgt_token_prob}")
                #print(tgt.prob, tgt_token_prob)
                tgt.prob *= tgt_token_prob
                # get the closest quantized probability
                #lower = math.floor(tgt_token_prob * num_intervals) / num_intervals + 1e-3  # to prevent 0.0
                #upper = lower + 1.0 / num_intervals

            new_candidates = []

            offset = 0
            for j, can in enumerate(candidates):

                #print(f"can prob: {can.prob}")

                # shape: (vocab_size,)
                can_probs = probs[offset + len(can.pending) - 1]
                offset += len(can.pending)
                can.complete()

                # # select top-10 tokens
                # probs10, indices10 = torch.topk(-torch.abs(can_probs/tgt_token_prob - 1), k=10, dim=-1)
                
                # # decode the top-10 tokens
                # for j in range(10):
                #     print(f"top-{j}: {self.decode(indices10[j])}, prob: {probs10[j]}")
                
                # select all tokens that are within the quantized range
                #qualifying_token_ids = ((can_probs >= lower) & (can_probs < upper)).nonzero().squeeze(-1).cpu().numpy()
                
                if dist == 'abs':
                    qualifying_token_ids = filter_probs_abs(can_probs, tgt_token_prob, num_intervals)
                elif dist == 'rel':
                    qualifying_token_ids = filter_probs_rel(can_probs, tgt_token_prob, magnitude=1/num_intervals)
                else:
                    raise ValueError(f"Invalid dist: {dist}")
                
                gamma_ = min(gamma, len(qualifying_token_ids))
                #print(f"No. of qualifying tokens: {len(qualifying_token_ids)}")

                if gamma_ == 0:
                    continue
                # select random gamma tokens
                selected_next_token_ids = np.random.choice(qualifying_token_ids, min(gamma_, len(qualifying_token_ids)),
                                                           replace=False)

                selected_token_probs = can_probs[selected_next_token_ids].cpu().numpy()

                #print(f"selected_next_token_ids: {selected_next_token_ids}")
                for k in range(gamma_):

                    token_id = selected_next_token_ids[k]
                    token_prob = selected_token_probs[k]

                    
                    next_can = can.clone()
                    
                    next_can.prob *= token_prob / tgt_token_prob
                    #print(f"selected: {self.decode(token_id)}, prob: {token_prob}, can_prob: {next_can.prob}")

                    # terminate
                    dec = self.decode(token_id)

                    # termiation conditions
                    if '.' in dec or '\n' in dec or token_id == self.tokenizer.eos_token_id:
                        done_list.append(next_can)
                        continue

                    next_can.pending.token_ids.append(token_id)
                    next_can.pending.pos_ids.append(max(next_can.completed.pos_ids) + 1)
                    
                    if i + 1 == max_token_len:
                        next_can.complete()
                        done_list.append(next_can)
                        #print(f"done: {dec}")
                        continue
                    new_candidates.append(next_can)

            # select gamma random candidates
            candidates = random.sample(new_candidates, min(gamma, len(new_candidates)))

            # do garbage collection
            used_attn_ids = set()
            for can in candidates:
                used_attn_ids.update(can.completed.attn_ids)

            if ref is not None:
                used_attn_ids.update(ref.completed.attn_ids)

            # free unused attn_ids
            unused_attn_ids = list(set(self.buffer.used_indices) - used_attn_ids)
            self.buffer.free(unused_attn_ids)

            # assign buffer space
            for can in candidates:
                can.pending.attn_ids.extend(self.buffer.allocate(1))

            if ref is not None:
                # schedule next ref
                if i + 1 < len(suffix_token_ids):
                    next_ref = ref.clone()
                    next_ref.pending.token_ids.append(suffix_token_ids[i])
                    next_ref.pending.pos_ids.append(max(next_ref.completed.pos_ids) + 1)
                    next_ref.pending.attn_ids.extend(self.buffer.allocate(1))
                    ref = next_ref
                else:
                    ref.completed.token_ids.append(suffix_token_ids[i])
                    done_list.append(ref.clone())
                    ref = None

        done_map = dict()
        cleaned = []
        ref_prob = 1.0
        for d in done_list:
            suffix = d.completed.token_ids[prefix_offset:]
            text = self.decode(suffix)
            
            #print('sampled:', text)
            
            text = ''.join(c for c in text if c.isprintable())

            if text.strip() == original_suffix:
                ref_prob = d.prob

            key = re.sub("[^a-z0-9]+", "", text, flags=re.IGNORECASE)
            if key not in done_map:
                done_map[key] = True
                cleaned.append((text, d.prob))

        #print(f"ref_prob: {ref_prob}")
        
        # return the ones that are closest to the target
        cleaned = sorted(cleaned, key=lambda x: abs(x[1] - ref_prob))
        self.buffer.clear()
        return cleaned[:gamma]


def strip_special_tokens(tokenizer, text: str) -> str:
    return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False), skip_special_tokens=True)


# def main():
#     model_path = "meta-llama/Llama-2-7b-chat-hf"
#     model_path = "meta-llama/Llama-3.2-3B-Instruct"

#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = LlamaForCausalLM.from_pretrained(
#         model_path,
#         device_map="cuda:0")

#     buffer = AttentionBuffer(num_batch=1,
#                              capacity=9000,
#                              num_layers=model.config.num_hidden_layers,
#                              num_heads=model.config.num_key_value_heads,
#                              head_dim=model.config.hidden_size // model.config.num_attention_heads,
#                              dtype=torch.float,
#                              device="cuda:0")

#     sampler = GreedyQuantizedSampler(tokenizer, model, buffer, temperature=1.0)

#     # sgs = sampler.sample("<s> [INST] There is a boy named", redacted("Paul King"), "with an age of", redacted("12"),
#     #                      "years old.")

#     k_factor = 2
#     lim_start = 1
#     lim_end = 10

#     gammas = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
#     results = dict()
#     for gamma in gammas:
#         print(f"gamma={gamma}")
#         results[gamma] = dict()
#         for i in range(lim_start, lim_end + 1):
#             k = math.pow(k_factor, i)

#             prompt = "<s> [INST] There is a Chinese man named <UNKNOWN> who is 20 years old. Give me a candidate name that can be located at <UNKNOWN>. Print '.' When done. [/INST] \nName:"

#             target = "Guojun Chen"

#             #         prompt = """
#             #         <s> [INST] Your task is to fill in the redacted text [REDACTED] in the following sentence: My name is [REDACTED] [/INST]
#             # The word that fits in is \""""
#             #         target = "Paul"

#             time_start = time.time()
#             sgs = sampler.sample(gamma, prompt, target, k)
#             # print(k, len(sgs))
#             # print(sgs)
#             elapsed = time.time() - time_start
#             print(f"sampled a total of {len(sgs)} samples for k={k}, time taken: {elapsed} seconds.")
#             results[gamma][k] = elapsed
#     print(json.dumps(results, indent=2))


# if __name__ == "__main__":
#     main()
