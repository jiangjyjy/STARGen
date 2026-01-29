import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel
from transformers import (
    # LlamaForCausalLM,
    # LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    default_data_collator,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
import deepspeed
from deepspeed.runtime.activation_checkpointing import checkpointing
from typing import Dict, List
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import datetime
from safetensors.torch import save_file
import random
from torch.optim import AdamW
from sklearn.metrics import f1_score
from deepspeed.ops.adam import DeepSpeedCPUAdam
from Bio import pairwise2
from collections import Counter
from Bio.Align import PairwiseAligner
import re
import math


SYSTEM_PROMPT1 = """You are an expert RNA inverse folding model.
                    Given an RNA structure, your task is to output the original RNA sequence.

                    STRICT RULES (must be followed exactly):
                    1. The input contains RNA structure tokens wrapped by <rna_pos_begin> and <rna_pos_end>.
                    2. You MUST output ONLY the RNA sequence.
                    3. The output sequence MUST:
                    - consist ONLY of the characters A, U, C, G
                    - have EXACTLY the same length as the number of RNA positions in the input structure
                    4. Do NOT output any explanations, comments, whitespace, or extra tokens.
                    5. Do NOT include <rna_begin>, <rna_end>, or any other tags in the output.
                    6. Output the sequence as a single continuous string (no spaces, no newlines).

                    The output must be a valid RNA sequence matching the given structure.
                """


def parse_args():
    parser = argparse.ArgumentParser(description='RNA Inverse Folding Training')
    
    parser.add_argument('--model_path', type=str, default="",
                      help='Path to pretrained model')
    parser.add_argument('--train_data', type=str, default="",
                      help='Path to training data')
    parser.add_argument('--output_dir', type=str, default="",
                      help='Output directory for checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42,
                  help='Random seed for reproducibility')
    
    parser.add_argument('--num_epochs', type=int, default=20,
                      help='Number of training epochs')
    parser.add_argument('--micro_batch_size', type=int, default=1,
                      help='Micro batch size per GPU')
    parser.add_argument('--grad_accum_steps', type=int, default=8,
                      help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-7, #1e-6, #5e-5,
                      help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=5000,
                      help='Number of warmup steps')
    parser.add_argument('--max_length', type=int, default=1024,
                      help='Maximum sequence length')
    parser.add_argument('--bf16', default=True,
                  help='Whether to use bf16 training')
    parser.add_argument('--tf32', default=True,
                  help='Whether to use tf32 training')
    
    parser.add_argument('--do_eval', action='store_true',
                      help='Whether to perform evaluation')
    parser.add_argument('--in_domain_eval', action='store_true',
                      help='Whether to evaluate on training data')
    parser.add_argument('--eval_steps', type=int, default=1000,
                      help='Number of steps between evaluations')
    parser.add_argument('--eval_ratio', type=float, default=0.1,
                      help='Ratio of evaluation set size to total dataset size')
    parser.add_argument('--max_eval_samples', type=int, default=3000,
                      help='Maximum number of evaluation samples to use')
    
    parser.add_argument('--save_steps', type=int, default=20,
                      help='Number of steps between checkpoint saves')
    parser.add_argument('--save_total_limit', type=int, default=5,
                      help='Maximum number of checkpoints to keep')
    
    parser.add_argument('--local_rank', type=int, default=-1,
                      help='Local rank for distributed training')
    
    return parser.parse_args()



def expand_tokenizer_vocabulary(tokenizer):

    new_tokens = []
    
    special_tokens = ['<rna_begin>', '<rna_end>', '<rna_pos_begin>', '<rna_pos_end>']
    new_tokens.extend(special_tokens)


    rna_tokens = [f'<RNA_{nt}>' for nt in ['A', 'U', 'C', 'G', 'N']]
    new_tokens.extend(rna_tokens)
    
    numbers = [f'<{i}>' for i in range(0, 4097)]
    new_tokens.extend(numbers)
    
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added_tokens} tokens to the vocabulary.")
    return tokenizer


class RnaDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024, task_mode="inverse_folding"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_mode = task_mode
        
        print(f"Loading data from {data_path}")
        if data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r') as file:
                for line in file:
                    data.append(json.loads(line))
            self.data = data
        else:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
            
        self.system_prompt_tokens1 = self.tokenizer(
            SYSTEM_PROMPT1, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        self.system_prompt_tokens2 = self.tokenizer(
            SYSTEM_PROMPT2, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        
        print(f"Loaded {len(self.data)} samples.")
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]

        sequence = item.get('sequence', item.get('rna_sequence'))
        structure = item.get('structure', item.get('rna_position'))

        if sequence is None or structure is None:
            raise KeyError(f"Missing keys: {item.keys()}")

        if self.task_mode is not None:
            if self.task_mode == "inverse_folding":
                sequence_text = f"\n\nRNA Structure: {structure}\nPredict the sequence:"
                target = sequence
                sys_prompt = self.system_prompt_tokens2
                sys_prompt_len = self.system_prompt_len2
            else:
                raise ValueError("Invalid task_mode. Choose 'structure_prediction' or 'inverse_folding'.")

        prompt_tokens_part = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=False,
            truncation=False
        )["input_ids"][0]
        
        prompt_ids = torch.cat([sys_prompt, prompt_tokens_part])
        
        if len(prompt_ids) > self.max_length - 256: 
            prompt_ids = prompt_ids[:self.max_length - 256]

        return {
            "prompt_ids": prompt_ids, 
            "target_text": target_text  
        }


def setup_training(args, total_steps):
    # Adjust total_steps and warmup_steps for gradient accumulation
    total_steps = total_steps // args.grad_accum_steps
    warmup_num_steps = min(args.warmup_steps // args.grad_accum_steps, total_steps)
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    global_batch_size = args.micro_batch_size * args.grad_accum_steps * world_size

    ds_config = {
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,

        "optimizer": {
            "type": "None"
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": warmup_num_steps,
                "total_num_steps": total_steps,
                "last_batch_iteration": -1
            }
        },
        "bf16": {
            "enabled": args.bf16
        },
        "fp16": {
            "enabled": not args.bf16
        },
        "tensor_parallel": {
            # "enabled": True,
            # "tp_size": world_size
        },
        "zero_optimization": {
            "stage": 1,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
        "gradient_clipping": 1.0,
        "train_batch_size": global_batch_size,
        "wall_clock_breakdown": False,
        "flops_profiler": {
            "enabled": False
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None
        }
    }
    return ds_config


def seq_sim(seq0, seq1):
    """Sequence similarity between two sequences using local alignment"""
    alignments = pairwise2.align.localxx(seq0, seq1)

    if len(alignments) > 0:

        best_alignment = alignments[0]
        
        c = 0

        for i in range(len(best_alignment.seqA)):
            if best_alignment.seqA[i] == best_alignment.seqB[i] and best_alignment.seqA[i] != '-':
                c += 1
        
        score = c / len(best_alignment.seqA)
        return score
    else:
        return 0.0 


aligner = PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 1.0     
aligner.mismatch_score = 0.0  
aligner.open_gap_score = 0.0   
aligner.extend_gap_score = 0.0

def extract_rna(seq: str) -> str:

    seq = seq.upper()
    return "".join(re.findall(r"[AUCG]", seq))


def align_pred_to_gt(pred_seq: str, gt_seq: str):

    L = len(gt_seq)
    return pred_seq[:L], gt_seq


def calculate_recovery_rate_alignment(pred_text: str, target_text: str) -> float:

    if not pred_text or not target_text:
        return 0.0

    def clean_sequence(text):
        if not text: return ""

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        unwanted = ["<rna_begin>", "<rna_end>", "<|im_start|>", "<|im_end|>", "<pad>", " ", "\n"]
        for t in unwanted:
            text = text.replace(t, "")

        return "".join(re.findall(r"[AUCGaucg]", text)).upper()

    gen_rna = clean_sequence(pred_text)
    tgt_rna = clean_sequence(target_text)

    aligned_gen_rna, aligned_tgt_rna = align_pred_to_gt(
                    gen_rna,
                    tgt_rna
                )

    if not aligned_tgt_rna:
        return 0.0
    if not aligned_gen_rna:
        return 0.0

    min_len = min(len(aligned_gen_rna), len(aligned_tgt_rna))
    

    match_count = 0
    for i in range(min_len):
        if aligned_gen_rna[i] == aligned_tgt_rna[i]:
            match_count += 1

    recovery_rate = match_count / len(aligned_tgt_rna)

    return min(max(recovery_rate, 0.0), 1.0)


def refined_reward(seq, target):
    raw_seq = seq
    # 1. 格式分
    format_score = 0.0
    # if "<rna_begin>" in raw_seq: format_score += 0.1
    # if "<rna_end>" in raw_seq: format_score += 0.1
    
    # 2. Think 惩罚
    think_penalty = 0.0
    # if "<think>" in raw_seq or "</think>" in raw_seq:
    #     think_penalty = -0.5

    gen_rna = extract_rna(raw_seq)
    tgt_rna = extract_rna(target)
    
    rr_score = 0.0
    len_penalty = 0.0

    if len(gen_rna) > 0:

        len_ratio = len(gen_rna) / len(tgt_rna)
        len_penalty = -min(abs(len_ratio - 1.0), 2.0) * 0.5 

        rr_score = calculate_recovery_rate_alignment(gen_rna, tgt_rna)
        total_reward = rr_score + format_score + len_penalty + think_penalty
    else:

        total_reward = -1.0 + format_score

    return {
        "total": min(max(total_reward, -1.0), 1.5),
        "rr": rr_score,
        "format": format_score,
        "len_penalty": len_penalty
    }


@torch.no_grad()
def evaluate(model_engine, dataloader, tokenizer, epoch, writer, args, max_new_tokens=512):

    torch.cuda.empty_cache()
    model_engine.eval()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device) 

    total_recovery = 0.0
    total_count = 0
    
    rank = dist.get_rank()
    disable_tqdm = (rank != 0)
    
    dist.barrier()


    if rank == 0:
        print(f"\n🚀 Starting Evaluation Epoch {epoch}...")
        
    pbar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", disable=disable_tqdm)

    for batch in dataloader:

        prompts = batch["prompt_ids"].to(device)      # [B, L]
        targets = batch["target_text"]                # List[str]
        

        gen_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
        
        outputs = gen_model.generate(
            input_ids=prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=(prompts != tokenizer.pad_token_id).long()
        )

        prompt_len = prompts.shape[1]
        gen_ids = outputs[:, prompt_len:]
        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


        for pred, tgt in zip(gen_texts, targets):
            rr = calculate_recovery_rate_alignment(pred, tgt)
            total_recovery += rr
            total_count += 1

        pbar.update(1)

    if rank == 0:
        pbar.close()

    stats = torch.tensor(
        [float(total_recovery), float(total_count)], 
        device=device, 
        dtype=torch.float64
    )
    

    dist.barrier()
    
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    global_total_recovery = stats[0].item()
    global_total_count = stats[1].item()
    avg_recovery = global_total_recovery / max(global_total_count, 1)


    if rank == 0:
        print(f"\n [Validation @ Epoch {epoch}] Result:")
        print(f"   - Samples Processed: {int(global_total_count)}")
        print(f"   - Global Avg Recovery Rate: {avg_recovery:.4f}\n")

        if writer is not None:

            step = getattr(args, 'global_step', epoch) 
            writer.add_scalar("Eval/RecoveryRate", avg_recovery, step)

    dist.barrier() 
    model_engine.train()
    torch.cuda.empty_cache()
    
    return avg_recovery

def split_dataset(dataset, args):
    """
    Split dataset into training and validation sets
    
    Args:
        dataset: The full dataset
        args: ArgumentParser args containing evaluation parameters
    
    Returns:
        train_dataset, val_dataset: Split datasets
    """
    if not args.do_eval:
        return dataset, None
        
    total_size = len(dataset)
    
    eval_size = int(total_size * args.eval_ratio)
    
    if args.max_eval_samples is not None:
        eval_size = min(eval_size, args.max_eval_samples)
        
    if args.in_domain_eval:
        indices = torch.randperm(total_size)[:eval_size]
        val_dataset = torch.utils.data.Subset(dataset, indices)
        train_dataset = dataset 
    else:
        train_size = total_size - eval_size
        train_dataset, temp_val = random_split(
            dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        if args.max_eval_samples and eval_size > args.max_eval_samples:
            val_indices = torch.randperm(eval_size)[:args.max_eval_samples]
            val_dataset = torch.utils.data.Subset(temp_val, val_indices)
        else:
            val_dataset = temp_val
    
    if dist.get_rank() == 0:
        print(f"Dataset split complete:")
        print(f"  Total samples: {total_size}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Evaluation samples: {len(val_dataset)}")
        print(f"  Evaluation mode: {'in-domain' if args.in_domain_eval else 'out-of-domain'}")
    
    return train_dataset, val_dataset

def monitor_training_status(loss, lr, step, epoch, global_step, model_engine=None):
    """Monitor training status and detect anomalies"""
    status = {
        'is_valid': True,
        'messages': []
    }
    
    # Check loss
    if torch.isnan(loss) or torch.isinf(loss):
        status['is_valid'] = False
        status['messages'].append(f"Invalid loss value: {loss.item()}")
    
    # Check learning rate
    if lr < 1e-8:
        status['messages'].append(f"Learning rate too small: {lr}")
    
    # Check gradient norm if model is available
    if model_engine is not None:
        try:
            grad_norm = model_engine.get_grad_norm()
            if grad_norm > 100:
                status['messages'].append(f"Large gradient norm: {grad_norm}")
        except:
            pass  # Ignore if get_grad_norm is not available
    
    return status

def save_training_state(model_engine, save_path, epoch, step, global_step):
    """Save complete training state including RNG states"""
    client_state = {
        'epoch': epoch,
        'step': step,
        'global_step': global_step,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'np_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }
    print("get client states")
    # model_engine.save_checkpoint(save_path, client_state=client_state)
    lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_engine.module.state_dict())
    model_engine.module.save_pretrained(save_path, state_dict=lean_state_dict)
    
def restore_training_state(client_state):
    """Restore complete training state including RNG states"""
    if not client_state:
        return
        
    if 'rng_state' in client_state:
        torch.set_rng_state(client_state['rng_state'])
    if 'cuda_rng_state' in client_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(client_state['cuda_rng_state'])
    if 'np_rng_state' in client_state:
        np.random.set_state(client_state['np_rng_state'])
    if 'python_rng_state' in client_state:
        random.setstate(client_state['python_rng_state'])

def get_latest_checkpoint(checkpoint_dir):

    if not os.path.exists(checkpoint_dir):
        return None
        
    def is_valid_checkpoint(path):
        required_files = ['model.safetensors', 'config.json']
        return all(os.path.exists(os.path.join(path, f)) for f in required_files)
        
    checkpoints = [d for d in os.listdir(checkpoint_dir) 
                  if os.path.isdir(os.path.join(checkpoint_dir, d)) 
                  and is_valid_checkpoint(os.path.join(checkpoint_dir, d))]
    if not checkpoints:
        return None
        
    def parse_checkpoint_name(ckpt):

        if ckpt.startswith('epoch_'):

            epoch = int(ckpt.split('_')[1])

            return (epoch, float('inf'))
        elif ckpt.startswith('step_'):

            parts = ckpt.split('_')
            epoch = int(parts[1])
            step = int(parts[2])
            return (epoch, step)
        else:
            return (-1, -1)  
            
    sorted_checkpoints = sorted(checkpoints, 
                              key=lambda x: parse_checkpoint_name(x),
                              reverse=True)  
                              
    if not sorted_checkpoints:
        return None
        
    latest = sorted_checkpoints[0]
    return os.path.join(checkpoint_dir, latest)

def clean_old_checkpoints(checkpoint_dir, save_total_limit):

    if not os.path.exists(checkpoint_dir):
        return
        
    checkpoints = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(checkpoints) <= save_total_limit:
        return
        
    def parse_checkpoint_name(ckpt):
        if ckpt.startswith('epoch_'):
            epoch = int(ckpt.split('_')[1])
            return (epoch, float('inf'))
        elif ckpt.startswith('step_'):
            parts = ckpt.split('_')
            epoch = int(parts[1])
            step = int(parts[2])
            return (epoch, step)
        else:
            return (-1, -1)
            
    sorted_checkpoints = sorted(checkpoints,
                              key=lambda x: parse_checkpoint_name(x),
                              reverse=True) 
                              
    checkpoints_to_delete = sorted_checkpoints[save_total_limit:]
    

    for ckpt in checkpoints_to_delete:
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        try:
            import shutil
            shutil.rmtree(ckpt_path)
            if dist.get_rank() == 0:
                print(f"Removed old checkpoint: {ckpt}")
        except Exception as e:
            print(f"Error deleting checkpoint {ckpt}: {str(e)}")


def riga_data_collator(features, tokenizer):

    prompt_ids_list = [f["prompt_ids"] for f in features]
    target_texts = [f["target_text"] for f in features]

    max_len = max(len(p) for p in prompt_ids_list)
    
    padded_prompts = []
    attention_masks = []
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    for p in prompt_ids_list:
        len_p = len(p)
        pad_len = max_len - len_p
        
        padded_p = torch.cat([
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
            p
        ])
        padded_prompts.append(padded_p)
        
        mask = torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            torch.ones(len_p, dtype=torch.long)
        ])
        attention_masks.append(mask)
        
    return {
        "prompt_ids": torch.stack(padded_prompts),
        "attention_mask": torch.stack(attention_masks),
        "target_text": target_texts
    }

def calculate_reward(generated_seq, target_seq):

    gen = generated_seq.replace(" ", "").strip()
    tgt = target_seq.replace(" ", "").strip()
    
    if len(tgt) == 0: return 0.0

    min_len = min(len(gen), len(tgt))
    if min_len == 0: return 0.0
    
    matches = sum(1 for g, t in zip(gen[:min_len], tgt[:min_len]) if g == t)
    return matches / len(tgt)

def reward_base_composition(gen: str, tgt: str) -> float:

    gen = gen.replace(" ", "").strip()
    tgt = tgt.replace(" ", "").strip()

    if len(gen) == 0 or len(tgt) == 0:
        return 0.0

    gen_cnt = Counter(gen)
    tgt_cnt = Counter(tgt)

    common = 0
    for b in "AUCG":
        common += min(gen_cnt.get(b, 0), tgt_cnt.get(b, 0))

    return common / len(tgt)

def reward_length(gen: str, tgt: str) -> float:

    gen_len = len(gen)
    tgt_len = len(tgt)

    if tgt_len == 0:
        return 0.0

    diff = abs(gen_len - tgt_len) / tgt_len
    return math.exp(-diff)  # [0, 1]

def length_guard(gen: str, tgt: str) -> float:
    if len(tgt) == 0:
        return 0.0

    diff = abs(len(gen) - len(tgt)) / len(tgt)
    return math.exp(-2 * diff)


def reward_gc_content(gen: str, tgt: str) -> float:

    def gc_ratio(seq):
        if len(seq) == 0:
            return 0.0
        return (seq.count("G") + seq.count("C")) / len(seq)

    return 1.0 - abs(gc_ratio(gen) - gc_ratio(tgt))

def reward_soft_alignment(gen: str, tgt: str) -> float:

    gen = gen.replace(" ", "").strip()
    tgt = tgt.replace(" ", "").strip()

    min_len = min(len(gen), len(tgt))
    if min_len == 0:
        return 0.0

    match = sum(1 for g, t in zip(gen[:min_len], tgt[:min_len]) if g == t)
    return match / len(tgt)

def calculate_total_reward(gen: str, tgt: str) -> float:
    r_rec = soft_recovery_reward(gen, tgt)
    r_len = length_guard(gen, tgt)

    reward = 0.8 * r_rec + 0.2 * r_len

    return float(max(reward, 0.05))


def calculate_recovery_rate(gen: str, tgt: str) -> float:

    gen = gen.replace(" ", "").strip()
    tgt = tgt.replace(" ", "").strip()

    if len(gen) == 0 or len(tgt) == 0:
        return 0.0

    min_len = min(len(gen), len(tgt))
    matches = sum(
        1 for g, t in zip(gen[:min_len], tgt[:min_len]) if g == t
    )
    return matches / len(tgt)

def soft_recovery_reward(gen: str, tgt: str) -> float:
    rr = calculate_recovery_rate(gen, tgt)

    reward = math.sqrt(rr + 1e-8)

    return max(reward, 0.05)


from functools import partial 

def main():

    global_step = 0
    start_epoch = 0

    # ===== Recovery Rate Tracking =====
    best_recovery_global = 0.0

    step_recovery_sum = 0.0
    step_recovery_count = 0


    REC_LOG_INTERVAL = 3

    args = parse_args()


    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        raise RuntimeError("LOCAL_RANK not set by DeepSpeed")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision('high')
    

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    try:
        if not torch.distributed.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=1800))
    except Exception as e:
        print(f"Failed to initialize process group: {str(e)}")
        raise
    
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))
    else:
        writer = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=True,
            trust_remote_code=True,
            padding_side='left' 
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        tokenizer = expand_tokenizer_vocabulary(tokenizer)
        
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise


    try:
        dataset = RnaDataset(args.train_data, tokenizer, args.max_length)
        train_dataset, val_dataset = split_dataset(dataset, args)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True
    )
    
    my_collator = partial(riga_data_collator, tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size, 
        sampler=train_sampler,
        collate_fn=my_collator, 
        pin_memory=True,
        num_workers=16
    )

    if val_dataset is not None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.micro_batch_size,
            sampler=val_sampler,
            collate_fn= partial(riga_data_collator, tokenizer=tokenizer), #default_data_collator,
            pin_memory=True,
            num_workers=16,
            persistent_workers=True,
            prefetch_factor=4
        )
    else:
        val_dataloader = None
    
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = False 
    config.pad_token_id = tokenizer.pad_token_id
    
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,                   
                bnb_4bit_use_double_quant=True,      
                bnb_4bit_quant_type="nf4",           
                bnb_4bit_compute_dtype=torch.bfloat16,  
            )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable() 

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    ref_model.resize_token_embeddings(len(tokenizer))
    ref_model.eval()
    ref_model.requires_grad_(False)
    ref_model.to(device)

    total_steps = len(train_dataloader) * args.num_epochs
    ds_config = setup_training(args, total_steps)
    
    # Get latest checkpoint
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint and dist.get_rank() == 0:
        print(f"Found latest checkpoint: {latest_checkpoint}")

    try:    
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )

        GROUP_SIZE = 8     
        BETA = 1.3         
        CLIP_EPS = 0.2     
        
        global_step = 0
        start_epoch = 0 

        if latest_checkpoint:
            print(f"Loading from latest checkpoint: {latest_checkpoint}")
            _, client_state = model_engine.load_checkpoint(latest_checkpoint)
            
            # Restore RNG states and training state
            restore_training_state(client_state)
            
            if client_state:
                start_epoch = client_state.get('epoch', 0)
                resume_step = client_state.get('step', 0)
                global_step = client_state.get('global_step', 0)
                
                if resume_step > 0:
                    print(f"Resuming from epoch {start_epoch}, step {resume_step}")
                else:
                    print(f"Starting epoch {start_epoch} from beginning")
                    resume_step = 0
            else:
                # Parse from checkpoint name as fallback
                if os.path.basename(latest_checkpoint).startswith('epoch_'):
                    start_epoch = int(os.path.basename(latest_checkpoint).split('_')[1]) + 1
                    resume_step = 0
                else:  # step_epoch_step format
                    start_epoch = int(os.path.basename(latest_checkpoint).split('_')[1])
                    resume_step = int(os.path.basename(latest_checkpoint).split('_')[2]) + 1
                
            print(f"Resuming from epoch {start_epoch}, step {resume_step}, global_step {global_step}")
            
    except Exception as e:
        print(f"Error initializing DeepSpeed or loading checkpoint: {str(e)}")
        raise

    rank = dist.get_rank()
    
    for epoch in range(start_epoch, args.num_epochs):
        
        # ===== epoch-level recovery =====
        epoch_recovery_sum = 0.0
        epoch_recovery_count = 0
        epoch_best_recovery = 0.0
        
        train_sampler.set_epoch(epoch)
        model_engine.train()
        
        pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}", disable=(rank != 0))

        for step, batch in enumerate(train_dataloader):

            prompts = batch["prompt_ids"].to(device) 
            target_texts = batch["target_text"]     
            
            input_ids = prompts.repeat_interleave(GROUP_SIZE, dim=0)
            
            torch.cuda.empty_cache()
            with torch.no_grad():

                generation_output = model_engine.module.generate(
                    input_ids=input_ids,
                    max_new_tokens=512, 
                    do_sample=True,      
                    temperature=1.0,     
                    top_k=800,
                    top_p=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    attention_mask=(input_ids != tokenizer.pad_token_id).long()
                )
            

            prompt_length = input_ids.shape[1]
            full_sequences = generation_output 
            generated_sequences = full_sequences[:, prompt_length:]
            
                       
            rewards = []
            rr_list = []
            format_list = []

            decoded_seqs = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

            for i, seq in enumerate(decoded_seqs):

                target = target_texts[i // GROUP_SIZE]

                res = refined_reward(seq, target)
                
                rewards.append(res["total"])
                rr_list.append(res["rr"])
                format_list.append(res["format"])                

                # rewards.append(rr_score)

            rewards_tensor = torch.tensor(rewards, device=device)

            rr_tensor = torch.tensor(rr_list, device=device)
            format_tensor = torch.tensor(format_list, device=device)

            avg_total_reward = rewards_tensor.mean().item()
            avg_rr = rr_tensor.mean().item()
            avg_format = format_tensor.mean().item()

            avg_recovery = rewards_tensor.mean().item()
            step_recovery_sum += avg_recovery
            step_recovery_count += 1
            
            if avg_recovery > best_recovery_global:
                best_recovery_global = avg_recovery

            epoch_recovery_sum += avg_recovery
            epoch_recovery_count += 1
            if avg_recovery > epoch_best_recovery:
                epoch_best_recovery = avg_recovery

            rewards_view = rewards_tensor.view(-1, GROUP_SIZE)
            
            mean_rewards = rewards_view.mean(dim=1, keepdim=True)
            std_rewards = rewards_view.std(dim=1, keepdim=True) + 1e-8
            
            advantages = (rewards_view - mean_rewards) / std_rewards
            
            advantages = advantages.view(-1).unsqueeze(1) 

            ref_model.eval()
            with torch.no_grad():
                ref_outputs = ref_model(full_sequences, attention_mask=(full_sequences != tokenizer.pad_token_id))
                ref_logits = ref_outputs.logits[:, :-1, :] 
            
            policy_outputs = model_engine(full_sequences, attention_mask=(full_sequences != tokenizer.pad_token_id))
            policy_logits = policy_outputs.logits[:, :-1, :] 

            shifted_input_ids = full_sequences[:, 1:]
            
            mask = (shifted_input_ids != tokenizer.pad_token_id).float()

            mask[:, :prompt_length-1] = 0.0
            

            policy_log_probs = torch.gather(policy_logits.log_softmax(-1), -1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
            ref_log_probs = torch.gather(ref_logits.log_softmax(-1), -1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

            old_log_probs = ref_log_probs.detach()
            
            ratio = torch.exp(policy_log_probs - old_log_probs)
            
            ratio_ref = torch.exp(ref_log_probs - policy_log_probs)
            kl_div = torch.exp(ref_log_probs - policy_log_probs) - (ref_log_probs - policy_log_probs) - 1 
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            
            loss_per_token =  (torch.min(surr1, surr2) - BETA * kl_div)
            
            loss = -(loss_per_token * mask).sum() / (mask.sum() + 1e-8)

            mean_kl = (kl_div * mask).sum() / mask.sum()

            model_engine.backward(loss)
            model_engine.step()
            
            global_step += 1

            if rank == 0:
                pbar.update(1)
                if global_step % REC_LOG_INTERVAL == 0:
                    pbar.set_postfix({
                        "Loss": f"{loss.item():.4f}",
                        "RR": f"{avg_rr:.3f}",      
                        "Fmt": f"{avg_format:.2f}", 
                        "Rew": f"{avg_total_reward:.2f}" 
                    })
                
                if writer:
                    writer.add_scalar("Train/Loss", loss.item(), global_step)
                    writer.add_scalar("Train/Total_Reward", avg_total_reward, global_step)
                    writer.add_scalar("Train/Pure_RecoveryRate", avg_rr, global_step)
                    writer.add_scalar("Train/Format_Score", avg_format, global_step)
                    writer.add_scalar("Train/KL", mean_kl.item(), global_step)


            if global_step % args.save_steps  == 0:

                dist.barrier()
                if rank == 0:
                    save_path = os.path.join(checkpoint_dir, f"step_{epoch}_{global_step}")
                    model_engine.module.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    save_training_state(model_engine, save_path, epoch, step, global_step)
                dist.barrier()


            if global_step % args.eval_steps == 0:

                dist.barrier()

                evaluate(model_engine, val_dataloader, tokenizer, epoch, writer, args)
                dist.barrier()

                model_engine.train()


        if dist.get_rank() == 0:
            pbar.close()
            epoch_avg_recovery = epoch_recovery_sum / max(epoch_recovery_count, 1)

            print(
                f"\n===== Epoch {epoch} Summary =====\n"
                f"Avg Recovery:  {epoch_avg_recovery:.4f}\n"
                f"Best Recovery: {epoch_best_recovery:.4f}\n"
                f"Best Global:  {best_recovery_global:.4f}\n"
                f"===============================\n"
            )

            if writer:
                writer.add_scalar("Train/RecoveryRate_EpochAvg", epoch_avg_recovery, epoch)
                writer.add_scalar("Train/RecoveryRate_EpochBest", epoch_best_recovery, epoch)
        
            save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
            model_engine.module.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        
        if args.do_eval and val_dataloader is not None:
            print(f"\n[Epoch {epoch} Validation]")
            evaluate(model_engine, val_dataloader, tokenizer, epoch, writer, args)

    if dist.get_rank() == 0:
        writer.close()
        
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
