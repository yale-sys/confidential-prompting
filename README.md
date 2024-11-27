# Confidential Prompting

This repository provides a proof-of-concept (PoC) implementation of Secure Multi-Party Decoding (SMD) and Prompt Obfuscation (PO) for confidential prompting, as described in the paper [Confidential Prompting: Protecting User Prompts from Cloud LLM Providers](https://arxiv.org/abs/2409.19134).

## Overview

The PoC demonstrates the fundamental mechanisms of confidential prompting using Llama models (Llama 2, 3, and 3.2). Although the scheme is designed for confidential virtual machines (CVMs) with GPUs, this implementation allows testing in a non-CVM environment by spawning prompt vaults as processes.

## Prerequisites

- **Python**: Version 3.10 or later
- **Dependencies**: Install the necessary packages:

  ```bash
  pip install -r requirements.txt
  ```

## Usage

### 1. Preparing Prompts for Obfuscation

Create a file (e.g., `input.txt`) with prompts, using the `<confidential>` tag to mark sections stored in CVMs and `<redacted>` to mark sensitive words to be obfuscated. Here’s an example using Llama 3 instruction model format:

```text
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|> 
you are a helpful assistant 
<|eot_id|>
<confidential>
<|start_header_id|>user<|end_header_id|>
Where is the capital city of <redacted>Uganda</redacted>?
Also, what is the population of <redacted>Korea</redacted>?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
</confidential>
```

Run the obfuscation command with relevant parameters:

- `gamma`: Number of virtual prompts to sample
- `epsilon`: Privacy budget
- `temperature`: Sampling temperature
- `prob_dist`: Probability difference metric (`abs` for absolute, `rel` for relative)
- `states_dir`: Directory to save intermediate states

```bash
python po.py \
  --prompt input.txt \
  --gamma 5 \
  --epsilon 0.1 \
  --temperature 1.0 \
  --prob_dist "abs" \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --device "cuda:0" \
  --states_dir "./states" \
  --verbose
```

Public and private attention states will be saved in `states_dir`.

### 2. Starting the LLM Serving Server

To run the LLM serving server, use the following command. The server loads the model and intermediate states from `states_dir`.

- `MASTER_ADDR` and `MASTER_PORT`: Define the server's IP and port
- `num_users`: Number of expected users
- `timeout_sec`: Server response timeout
- `max_num_tokens`: Maximum tokens to generate per request
- `print_idx`: Specifies which prompt version to display; set to `0` for the authentic prompt or to a higher value to show an obfuscated version. (note: In a real implementation, the index of the authentic prompt should be randomized and hidden from the server, but we use 0 for demonstration purposes.)

```bash
MASTER_ADDR=localhost MASTER_PORT=29501 python smd.py \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --device "cuda:0" \
  --states_dir "./states" \
  --num_users 1 \
  --timeout_sec 15 \
  --max_num_tokens 2048 \
  --standalone_master \
  --print_idx 0
```

### 3. Running the Secure Prompt Vault

For each user (i.e. CVM), run the secure prompt vault to connect to the LLM serving server and decode tokens securely. The `MASTER_ADDR` and `MASTER_PORT` must match the server, and each `user_id` should be unique.

```bash
MASTER_ADDR=localhost MASTER_PORT=29501 python smd.py \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --device "cuda:0" \
  --states_dir "./states" \
  --timeout_sec 15 \
  --standalone_worker \
  --user_id 0
```

To disable memory efficient mode in CVM, set `--disable_multiplexing`.

### 4. Simulated Multi-User Testing

To simulate multiple users in a non-CVM environment, you can remove the `--standalone_master` flag and run the following command. This will start the LLM server and spawn secure prompt vaults on the same host:

```bash
MASTER_ADDR=localhost MASTER_PORT=29501 python smd.py \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --device "cuda:0" \
  --states_dir "./states" \
  --num_users 2 \
  --timeout_sec 15 \
  --max_num_tokens 2048 \
  --print_idx 0
```

## License
MIT License