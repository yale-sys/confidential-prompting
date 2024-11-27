from __future__ import annotations

import fire
import os
import pickle

import numpy as np
import torch
from transformers import AutoTokenizer

from attention import AttentionBuffer
from llama import LlamaForCausalLM
from prompt import Prompt, Confidential, Redacted, PublicMeta, PrivateMeta, Replacement
from gqs import GreedyQuantizedSampler


@torch.inference_mode()
def prefill(
    prompt:str,
    gamma:int,
    epsilon:float,
    temperature:float,
    prob_dist:str,
    state_dir:str,
    model_path:str, 
    device:str,
    verbose:bool,
    capacity:int = 1024 * 3,
    ):

    # step1. parse prompt.
    prompt = Prompt(prompt.strip())

    # step2. create virtual prompts.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float,
        #quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0),
        device_map=device)

    buffer = AttentionBuffer(num_batch=1, 
                             capacity=capacity,
                             num_layers=model.config.num_hidden_layers,
                             num_heads=model.config.num_key_value_heads,
                             head_dim=model.config.hidden_size // model.config.num_attention_heads,
                             dtype=torch.float,
                             device=device)

    sampler = GreedyQuantizedSampler(tokenizer, model, buffer, temperature=temperature)

    num_intervals = 1.0 / epsilon
    replacements = {}

    for sampling_prompt, target in prompt.get_all_ar_sampling_prompts():
        
        #print(sampling_prompt, target.text)
        
        samples = sampler.sample(gamma * 2, sampling_prompt, target.text, num_intervals=num_intervals, dist=prob_dist)
        
        # only select samples that are not empty
        samples = [s for s in samples if len(s[0].strip()) > 1]
        replacements[target.uid] = [Replacement(s[0], s[1]) for s in samples]
        
        if verbose:
            print(f"Sampled for {target.text}: {samples}")

    # adjust gamma
    gamma = len(replacements[list(replacements.keys())[0]])

    mask_info = []

    public_token_ids = []
    public_pos_ids = []
    private_token_ids = []
    private_pos_ids = []

    pos_offset = 0

    # do encoding
    for e in prompt.elements:
        if isinstance(e, str):
            token_ids = tokenizer.encode(e)
            size = len(token_ids)

            public_token_ids.extend(token_ids)
            public_pos_ids.extend(range(pos_offset, pos_offset + size))
            pos_offset += size

        elif isinstance(e, Confidential):
            for ce in e.elements:
                if isinstance(ce, str):
                    token_ids = tokenizer.encode(ce, add_special_tokens=False)
                    size = len(token_ids)

                    private_token_ids.extend(token_ids)
                    private_pos_ids.extend(range(pos_offset, pos_offset + size))
                    mask_info.append((0, size))
                    pos_offset += size

                elif isinstance(ce, Redacted):

                    candidates = replacements[ce.uid]  # the first is the ground truth
                    max_size = 0
                    for i, can in enumerate(candidates):

                        token_ids = tokenizer.encode(can.text.strip(), add_special_tokens=False)
                        size = len(token_ids)
                        buffer_offset = len(private_token_ids)

                        pos_ids = list(range(pos_offset, pos_offset + size))
                        buffer_ids = list(range(buffer_offset, buffer_offset + size))

                        private_token_ids.extend(token_ids)
                        private_pos_ids.extend(pos_ids)

                        can.buffer_ids = buffer_ids
                        can.token_ids = token_ids
                        can.pos_ids = pos_ids

                        if i == 0:
                            mask_info.append((1, size))
                        else:
                            mask_info.append((2, size))

                        max_size = max(max_size, size)

                    pos_offset += max_size

    all_token_ids = private_token_ids + public_token_ids
    all_pos_ids = private_pos_ids + public_pos_ids

    # build mask
    mask = np.zeros((len(all_token_ids), len(all_token_ids)), dtype=np.bool_)
    mask_gt = np.zeros(len(all_token_ids), dtype=np.bool_)
    offset = 0

    for mask_type, mask_size in mask_info:
        # 0: global, 1: local (gt), 2: local (fake)
        if mask_type == 0:
            mask[:, offset:offset + mask_size] = True
            mask[offset:offset + mask_size, :] |= mask_gt  # optional
        else:
            mask[offset:offset + mask_size, offset:offset + mask_size] = True
            if mask_type == 1:
                mask_gt[offset:offset + mask_size] = True
        offset += mask_size

    # public parts are always shared
    mask[:, len(private_token_ids):] = True

    # causal masking
    for i, pos in enumerate(all_pos_ids):
        for j, pos_ in enumerate(all_pos_ids):
            if pos_ > pos:
                mask[i, j] = False

    # aa = np.array(all_token_ids)
    # for i in range(len(mask)):
        
    #     # find indices of Trues in mask[i]
    #     indices = np.where(mask[i])[0].astype(int)
    #     print('visible', tokenizer.decode(aa[indices]))
    #     print('pos', all_pos_ids[i])

    # step 2. fill in the buffer
    buffer.clear()
    sink_ids = buffer.allocate(len(all_token_ids))
    _ = model(
        input_ids=torch.as_tensor([all_token_ids], device=device),
        position_ids=torch.as_tensor([all_pos_ids], device=device),
        buffer=buffer,
        buffer_sink_ids=sink_ids,
        attention_mask=torch.as_tensor(mask, device=device).unsqueeze(0),
        confidential=False
    )

    # step 3. separate private and public states
    print('all tokens', all_token_ids)
    print('all pos', all_pos_ids)
    
    # detokenize
    print('detokenized all tokens', tokenizer.decode(all_token_ids))    
    

    private_buffer = buffer.excerpt(list(range(len(private_token_ids))))
    public_buffer = buffer.excerpt(list(range(len(private_token_ids), len(all_token_ids))))

    print('public buffer size', public_buffer.size())
    print('private buffer size', private_buffer.size())

    public_buffer.save(os.path.join(state_dir, "public.safetensors"))
    private_buffer.save(os.path.join(state_dir, "private.safetensors"))

    private_meta = PrivateMeta(
        prompt=prompt,
        gamma=gamma,
        len_private=len(private_token_ids),
        len_public=len(public_token_ids),
        token_ids=all_token_ids,
        pos_ids=all_pos_ids,
        mask_info=mask_info,
        replacements=replacements,
        path="private.safetensors"
    )

    public_meta = PublicMeta(
        gamma=gamma,
        len_public=len(public_token_ids),
        initial_token_ids=[tokenizer.encode(" ")[-1]] * gamma,
        pos_offset=max(all_pos_ids),
        path="public.safetensors"
    )

    # save private meta with pickle
    with open(os.path.join(state_dir, "private.meta"), "wb") as f:
        pickle.dump(private_meta, f)

    # save public meta with pickle
    with open(os.path.join(state_dir, "public.meta"), "wb") as f:
        pickle.dump(public_meta, f)
        
            
    # ---------------------
    
    # virtual_prompt_buffer_ids = [[] for _ in range(private_meta.gamma)]

    # for reps in private_meta.replacements.values():
    #     for i in range(private_meta.gamma):
    #         j = i % len(reps)

    #         rel_ids = reps[j].buffer_ids
    #         virtual_prompt_buffer_ids[i].extend(rel_ids)

    # common_mask = np.zeros(private_meta.len_private, dtype=np.bool_)
    # offset = 0
    # for mask_type, mask_size in private_meta.mask_info:
    #     if mask_type == 0:
    #         common_mask[offset:offset + mask_size] = True
    #     offset += mask_size

    # mask = np.zeros((private_meta.gamma, private_meta.len_private), dtype=np.bool_)
    # mask[:, :] = common_mask

    # for i in range(private_meta.gamma):
    #     mask[i, virtual_prompt_buffer_ids[i]] = True
    
    
    # # get nonzero indices of mask[0]
    # indices = np.where(mask[0])[0].astype(int)
    
    # # get tokens
    # print('tokens', tokenizer.decode(np.array(private_token_ids)[indices]))
    
    # # get attention states
    # print('attention states', private_buffer.k[0][0, :, indices, :].sum())
    
    # ---------------------


def main(prompt:str,
         gamma: int = 5,
         epsilon: float = 0.1,
         temperatue: float = 1.0,
         prob_dist:str = "abs",
         model="meta-llama/Llama-3.2-3B-Instruct",
         device="cuda:0",
         states_dir="./states",
         verbose=True):
    
    # read the prompt from the file
    with open(prompt, "r") as f:
        prompt = f.read()

    # Ensure the save directory exists (optional, but useful)
    os.makedirs(states_dir, exist_ok=True)
    
    prefill(
        prompt=prompt,
        gamma=gamma,
        epsilon=epsilon,
        temperature=temperatue,
        prob_dist=prob_dist,
        state_dir=states_dir,
        model_path=model,
        device=device,
        verbose=verbose
    )
    

if __name__ == "__main__":
    fire.Fire(main)