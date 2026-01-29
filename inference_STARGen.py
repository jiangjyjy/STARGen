#!/usr/bin/env python
import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    default_data_collator,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
import deepspeed
from typing import Dict, List
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import datetime
import math
from safetensors.torch import save_file
import random
from torch.optim import AdamW
from sklearn.metrics import f1_score
import numpy as np
from Bio import pairwise2
import csv
import subprocess
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    parser = argparse.ArgumentParser(description='RNA Inverse Folding Evaluation')
    
    # 基础训练参数
    parser.add_argument('--model_path', type=str, default="",
                      help='Path to pretrained model')
    parser.add_argument('--train_data', type=str, default="",
                      help='Path to training data')
    parser.add_argument('--output_dir', type=str, default="",
                      help='Output directory for checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42,
                  help='Random seed for reproducibility')
    
    # 训练配置
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--micro_batch_size', type=int, default=1,
                      help='Micro batch size per GPU')
    parser.add_argument('--grad_accum_steps', type=int, default=8,
                      help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
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
  
    
    # 评估参数
    parser.add_argument('--do_eval', action='store_true',
                      help='Whether to perform evaluation during training')
    parser.add_argument('--in_domain_eval', action='store_true',
                      help='Whether to evaluate on training data')
    parser.add_argument('--eval_steps', type=int, default=10000,
                      help='Number of steps between evaluations')
    parser.add_argument('--eval_ratio', type=float, default=0.1,
                      help='Ratio of evaluation set size to total dataset size')
    parser.add_argument('--max_eval_samples', type=int, default=3000,
                      help='Maximum number of evaluation samples to use')
    
    # 保存参数
    parser.add_argument('--save_steps', type=int, default=1000,
                      help='Number of steps between checkpoint saves')
    parser.add_argument('--save_total_limit', type=int, default=5,
                      help='Maximum number of checkpoints to keep')
    
    # 分布式训练
    parser.add_argument('--local_rank', type=int, default=-1,
                      help='Local rank for distributed training')
    
    # New flag to run evaluation only (skips training loop)
    # parser.add_argument('--eval_only', action='store_true',
    #                   help='Run evaluation only mode (do not perform training) and report perplexity for both tasks')
    
    return parser.parse_args()


# def expand_tokenizer_vocabulary(tokenizer: LlamaTokenizer):
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

    def __init__(self, data_path, tokenizer, max_length=2048, task_mode = "inverse_folding"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_mode = task_mode
        
        print(f"Loading and preprocessing data from {data_path}")
        if data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r') as file:
                for line in file:
                    data.append(json.loads(line))
            self.data= data
        else:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
            
        self.system_prompt_tokens1 = self.tokenizer(
            SYSTEM_PROMPT1,
            add_special_tokens=False,
            return_tensors="pt",
            padding=False,
            truncation=False
        )["input_ids"][0]
        
        self.system_prompt_len1 = len(self.system_prompt_tokens1)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
            
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]

        sequence = item.get('sequence', item.get('rna_sequence'))
        structure = item.get('structure', item.get('rna_position'))

        if sequence is None or structure is None:
            raise KeyError(f"Missing keys in data item: {item.keys()}")

        if self.task_mode is not None:
            if self.task_mode == "inverse_folding":
                sequence_text = f"\n\nRNA Structure: {structure}\nPredict the sequence:"
                target = sequence
                sys_prompt = self.system_prompt_tokens2
                sys_prompt_len = self.system_prompt_len2
            else:
                raise ValueError("Invalid task_mode. Choose 'structure_prediction' or 'inverse_folding'.")
        
        # Tokenize inputs
        sequence_tokens = self.tokenizer(
            sequence_text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=False,
            truncation=False
        )["input_ids"][0]
        
        target_tokens = self.tokenizer(
            target,
            add_special_tokens=False,
            return_tensors="pt",
            padding=False,
            truncation=False
        )["input_ids"][0]

        # Combine tokens
        input_ids = torch.cat([
            sys_prompt,
            sequence_tokens,
            target_tokens
        ])
        
        # Simple truncation if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            
        # Add padding if necessary
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = torch.cat([
                torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long),
                input_ids
            ])
            
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # Create labels (mask prompt part)
        labels = input_ids.clone()
        prompt_len = sys_prompt_len + len(sequence_tokens)
        labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_ids": input_ids,
            "target_text": target,   
            "structure_text": structure 
        }

def custom_data_collator(features):

    batch = {}
    
    for key in ["input_ids", "attention_mask", "labels", "prompt_ids"]:
        if key in features[0]:
            batch[key] = torch.stack([f[key] for f in features])

    for key in ["target_text", "structure_text"]:
        if key in features[0]:
            batch[key] = [f[key] for f in features]
            
    return batch

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
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": args.weight_decay
            }
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

def extract_rna(seq: str) -> str:

    if not seq: return ""
    seq = seq.upper()
    return "".join(re.findall(r"[AUCG]", seq))

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

    if not tgt_rna:
        return 0.0
    if not gen_rna:
        return 0.0

    min_len = min(len(gen_rna), len(tgt_rna))
    

    match_count = 0
    for i in range(min_len):
        if gen_rna[i] == tgt_rna[i]:
            match_count += 1

    recovery_rate = match_count / len(tgt_rna)

    return min(max(recovery_rate, 0.0), 1.0)

def parse_dot_bracket(structure):

    stack = []
    pairs = set()
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.add(tuple(sorted((j, i))))
    return pairs

def calculate_structure_f1(pred_structure, gt_structure):

    if not pred_structure and not gt_structure:
        return 1.0
    if not pred_structure or not gt_structure:
        return 0.0

    pred_pairs = parse_dot_bracket(pred_structure)
    gt_pairs = parse_dot_bracket(gt_structure)
    

    if len(pred_pairs) == 0 and len(gt_pairs) == 0:
        return 1.0
    
    tp = len(pred_pairs.intersection(gt_pairs))
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def predict_structure_with_rnafold(sequence):

    try:
        clean_seq = extract_rna(sequence)
        if not clean_seq:
            return ""

        process = subprocess.Popen(
            ['RNAfold', '--noPS'], 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, _ = process.communicate(input=clean_seq)
        
        lines = output.strip().split('\n')
        

        for line in lines:

            if any(c in line for c in '().') and not line.startswith('>'):
                return line.split()[0]
        return ""
    except Exception as e:
        # print(f"RNAfold error: {e}") 
        return ""

def align_seq_length(cand_seq, gt_seq, mode="truncate", pad_char="N"):

    len_cand = len(cand_seq)
    len_gt = len(gt_seq)
    
    if mode == "truncate":
        target_len = min(len_cand, len_gt)
    else:
        target_len = max(len_cand, len_gt)
    
    if len_cand == target_len:
        aligned_cand = cand_seq
    elif len_cand > target_len:
        if mode == "truncate" or mode == "left":
            aligned_cand = cand_seq[:target_len]
        elif mode == "center":
            start = (len_cand - target_len) // 2
            aligned_cand = cand_seq[start:start+target_len]
    else:
        pad_len = target_len - len_cand
        if mode == "pad" or mode == "left":
            aligned_cand = cand_seq + pad_char * pad_len
        elif mode == "center":
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
            aligned_cand = pad_char * pad_left + cand_seq + pad_char * pad_right
    
    if len_gt == target_len:
        aligned_gt = gt_seq
    elif len_gt > target_len:
        if mode == "truncate" or mode == "left":
            aligned_gt = gt_seq[:target_len]
        elif mode == "center":
            start = (len_gt - target_len) // 2
            aligned_gt = gt_seq[start:start+target_len]
    else:
        pad_len = target_len - len_gt
        if mode == "pad" or mode == "left":
            aligned_gt = gt_seq + pad_char * pad_len
        elif mode == "center":
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
            aligned_gt = pad_char * pad_left + gt_seq + pad_char * pad_right
    
    return aligned_cand, aligned_gt


def align_pred_to_gt(pred_seq: str, gt_seq: str):

    L = len(gt_seq)
    return pred_seq[:L], gt_seq


@torch.no_grad()
def evaluate(model_engine, eval_dataloader, tokenizer, epoch, writer, args, csv_path=None):

    torch.cuda.empty_cache()
    model_engine.eval()

    if dist.get_rank() == 0 and csv_path is not None:
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        fieldnames = [
            'Index',
            'Original_Sequence',
            'Generated_Sequence',
            'Recovery_Rate',
            'Seq_Sim',
            'Refolded_F1'
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
    
    current_device_id = dist.get_rank()
    current_device = torch.device(f"cuda:{current_device_id}")

    # --- Best-of-N  ---
    NUM_SAMPLES = 5     
    TEMPERATURE = 1.0   
    
    WEIGHT_RR = 0.5
    WEIGHT_F1 = 0.5
    
    total_best_rr = 0.0
    total_best_f1 = 0.0
    total_seq_sim = 0.0
    total_samples = 0
    
    all_sample_results = []
    global_sample_index = 0
    
    if dist.get_rank() == 0:
        eval_pbar = tqdm(total=len(eval_dataloader), desc=f"Evaluating Epoch {epoch} (Best-of-{NUM_SAMPLES})")

    for batch in eval_dataloader:

        if (batch["labels"] != -100).sum() == 0:
            if dist.get_rank() == 0: eval_pbar.update(1)
            continue


        prompts = batch["prompt_ids"].to(current_device)
        target_seqs = batch["target_text"] 
        
        gen_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
        
        repeated_prompts = prompts.repeat_interleave(NUM_SAMPLES, dim=0)
        
        outputs = gen_model.generate(
            input_ids=repeated_prompts,
            max_new_tokens=512,
            do_sample=True,     
            temperature=TEMPERATURE,
            top_p=0.7,
            top_k=800,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=(repeated_prompts != tokenizer.pad_token_id).long()
        )
        
        prompt_len = repeated_prompts.shape[1]
        gen_ids = outputs[:, prompt_len:]
        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        
        batch_size = len(target_seqs)
        
        for i in range(batch_size):
            gt_seq_raw = target_seqs[i]
            gt_rna = extract_rna(gt_seq_raw)
            
            if len(gt_rna) > 0:
                gt_struct_ref = predict_structure_with_rnafold(gt_rna)
            else:
                gt_struct_ref = ""

            candidates = gen_texts[i*NUM_SAMPLES : (i+1)*NUM_SAMPLES]
            
            best_score = -1.0
            best_metrics = None
            
            for cand_text in candidates:
                cand_rna = extract_rna(cand_text)

                aligned_cand_rna, aligned_gt_rna = align_pred_to_gt(
                    cand_rna,
                    gt_rna
                )

                rr = calculate_recovery_rate_alignment(aligned_cand_rna, aligned_gt_rna)
                
                f1 = 0.0
                pred_struct = ""
                
                if len(aligned_cand_rna) > 0 and gt_struct_ref:

                    pred_struct = predict_structure_with_rnafold(aligned_cand_rna)

                    aligned_gt_struct = gt_struct_ref[:len(pred_struct)]

                    f1 = calculate_structure_f1(pred_struct, aligned_gt_struct)
                
                composite_score = (WEIGHT_RR * rr) + (WEIGHT_F1 * f1)
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_metrics = {
                        "seq": aligned_cand_rna,
                        "gt": aligned_gt_rna,
                        "rr": rr,
                        "f1": f1,
                        "struct": pred_struct
                    }
            
            if best_metrics:
                total_best_rr += best_metrics["rr"]
                total_best_f1 += best_metrics["f1"]
                
                sim = seq_sim(best_metrics["seq"], best_metrics["gt"])
                total_seq_sim += sim
                
                if csv_writer is not None:
                    csv_writer.writerow({
                        'Index': global_sample_index,
                        'Original_Sequence': best_metrics["gt"],
                        'Generated_Sequence': best_metrics["seq"],
                        'Recovery_Rate': best_metrics["rr"],
                        'Seq_Sim': sim,
                        'Refolded_F1': best_metrics["f1"]
                    })
                    csv_file.flush()

                print(
                    f"[Eval][{global_sample_index:05d}] "
                    f"RR={best_metrics['rr']:.4f} | "
                    f"SeqSim={sim:.4f} | "
                    f"F1={best_metrics['f1']:.4f} | "
                    f"Len(pred/gt)={len(best_metrics['seq'])}/{len(best_metrics['gt'])}"
                )
            
            total_samples += 1
            global_sample_index += 1
            
        if dist.get_rank() == 0:
            eval_pbar.update(1)

    if dist.get_rank() == 0:
        eval_pbar.close()

    stats = torch.tensor([
        total_best_rr, 
        total_best_f1, 
        total_seq_sim, 
        float(total_samples)
    ], device=current_device)
    
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
    global_samples = max(stats[3].item(), 1)
    avg_rr = stats[0].item() / global_samples
    avg_f1 = stats[1].item() / global_samples
    avg_sim = stats[2].item() / global_samples

    if dist.get_rank() == 0:
        print(f"\n [Best-of-{NUM_SAMPLES} Refolded Eval Results]")
        print(f"   - Samples: {int(global_samples)}")
        print(f"   - Avg Recovery Rate: {avg_rr:.4f}")
        print(f"   - Avg Refolded F1  : {avg_f1:.4f}")
        print(f"   - Avg Seq Sim      : {avg_sim:.4f}")
        
        if writer:
            writer.add_scalar("Eval/RR_BestOfN", avg_rr, epoch)
            writer.add_scalar("Eval/F1_Refolded", avg_f1, epoch)
    
    if dist.get_rank() == 0 and csv_file is not None:
        csv_file.close()

    model_engine.train()
    
    if dist.get_rank() == 0:
        return 0.0, [avg_rr, avg_f1, avg_sim]
    else:
        return 0.0, [0, 0, 0]



def pad_lists(list1, list2, pad_value=0):
    # Find the maximum length of the two lists
    max_len = max(len(list1), len(list2))
    
    # Pad both lists to the same length
    padded_list1 = list1 + [pad_value] * (max_len - len(list1))
    padded_list2 = list2 + [pad_value] * (max_len - len(list2))
    
    return padded_list1, padded_list2

def split_dataset(dataset, args):

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
    """获取最新的checkpoint路径
    按照epoch和step的数字大小排序,返回最新的checkpoint路径
    """
    if not os.path.exists(checkpoint_dir):
        return None
        
    def is_valid_checkpoint(path):
        # required_files = ['pytorch_model.bin', 'config.json', 'optimizer.pt']
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


def main():
    args = parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision('high')
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if args.tf32:
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    
    try:
        if not torch.distributed.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=datetime.timedelta(seconds=1800)
            )
        if dist.get_rank() == 0:
            print(f"Initialized process group - backend={dist.get_backend()}, world_size={dist.get_world_size()}")
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
            padding_side='right'
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        special_tokens_dict = {
            'pad_token': tokenizer.pad_token,
            'eos_token': tokenizer.eos_token,
            'bos_token': tokenizer.bos_token if tokenizer.bos_token else '<s>',
            'unk_token': tokenizer.unk_token if tokenizer.unk_token else '<unk>'
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        
        tokenizer = expand_tokenizer_vocabulary(tokenizer)
        
        config = AutoConfig.from_pretrained(args.model_path)
        config.use_cache = False
        config.pad_token_id = tokenizer.pad_token_id
        config.use_flash_attention = True  
        
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
            low_cpu_mem_usage=True,
            device_map="auto",        
            quantization_config=bnb_config,
        )

        
        model.gradient_checkpointing_enable()
        
    except Exception as e:
        print(f"Error in model/tokenizer loading: {str(e)}")
        raise

    # Initialize optimizer and DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(), #optimizer=optimizer,
        config=setup_training(args, total_steps=1)  # total_steps is not used in eval-only mode
    )
        
    start_epoch = 0
    resume_step = 0
    global_step = 0

    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint and dist.get_rank() == 0:
        print(f"Found latest checkpoint: {latest_checkpoint}")
    if latest_checkpoint:
        print(f"Loading from latest checkpoint: {latest_checkpoint}")
        _, client_state = model_engine.load_checkpoint(latest_checkpoint)
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
            if os.path.basename(latest_checkpoint).startswith('epoch_'):
                start_epoch = int(os.path.basename(latest_checkpoint).split('_')[1]) + 1
                resume_step = 0
            else:
                start_epoch = int(os.path.basename(latest_checkpoint).split('_')[1])
                resume_step = int(os.path.basename(latest_checkpoint).split('_')[2]) + 1
                
        print(f"Resuming from epoch {start_epoch}, step {resume_step}, global_step {global_step}")
    
    # Check if we want to run evaluation only
    if True:

        eval_dataset_structure = RnaDataset(args.train_data, tokenizer, args.max_length, task_mode="structure_prediction")

        eval_dataset_inverse = RnaDataset(args.train_data, tokenizer, args.max_length, task_mode="inverse_folding")

        eval_sampler_structure = torch.utils.data.distributed.DistributedSampler(
            eval_dataset_structure,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False
        )
        eval_dataloader_structure = DataLoader(
            eval_dataset_structure,
            batch_size=args.micro_batch_size,
            sampler=eval_sampler_structure,
            collate_fn=default_data_collator,
            pin_memory=True,
            num_workers=16,
            persistent_workers=True,
            prefetch_factor=4
        )
        
        eval_sampler_inverse = torch.utils.data.distributed.DistributedSampler(
            eval_dataset_inverse,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False
        )

        eval_dataloader_inverse = DataLoader(
            eval_dataset_inverse,
            batch_size=args.micro_batch_size,
            sampler=eval_sampler_inverse,
            collate_fn=custom_data_collator, #default_data_collator,
            pin_memory=True,
            num_workers=16,
            persistent_workers=True,
            prefetch_factor=4,
        )

        if dist.get_rank() == 0:
            all_runs_metrics = []
        
            for run_id in range(3):
                print(f"\n >>> Starting Run {run_id + 1} / 3 ...")
                
                current_seed = args.seed + run_id
                random.seed(current_seed)
                np.random.seed(current_seed)
                torch.manual_seed(current_seed)
                torch.cuda.manual_seed_all(current_seed)

                step_name = os.path.basename(args.model_path.strip("/"))
                run_csv_path = os.path.join(args.output_dir, f"eval_run{run_id}_{step_name}.csv")

                _, metrics = evaluate(
                    model_engine,
                    eval_dataloader_inverse,
                    tokenizer,
                    epoch=0,
                    writer=None,
                    args=args,
                    csv_path=run_csv_path
                )
                
                all_runs_metrics.append(metrics) 

            metrics_array = np.array(all_runs_metrics)
            means = np.mean(metrics_array, axis=0)
            stds = np.std(metrics_array, axis=0)

            print("\n" + "="*40)
            print("FINAL ERROR BAR RESULTS (n=3)")
            print(f"Recovery Rate: {means[0]:.4f} ± {stds[0]:.4f}")
            print(f"Refolded F1  : {means[1]:.4f} ± {stds[1]:.4f}")
            print(f"Sequence Sim : {means[2]:.4f} ± {stds[2]:.4f}")
            print("="*40)
            
        
        if writer is not None and dist.get_rank() == 0:
            writer.close()
        torch.distributed.destroy_process_group()
        return  # Exit main after evaluation

    

if __name__ == "__main__":
    main()

