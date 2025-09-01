#!/usr/bin/env python3
"""
NEFF Pruning for Large Language Models - Multi-GPU B200 Production Script
Optimized for 8x B200 90GB GPUs on HiPerGator
Includes LLaMA family models and comprehensive task evaluation
"""

import os
import sys
import json
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import numpy as np
from tqdm import tqdm
from datetime import datetime
import gc
import argparse
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import time
import warnings
warnings.filterwarnings('ignore')

# HuggingFace imports
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    LlamaForCausalLM, LlamaTokenizer,
    BitsAndBytesConfig
)
from datasets import load_dataset
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

# Import evaluation libraries
try:
    from evaluate import load as load_metric
except ImportError:
    from datasets import load_metric

# =====================================
# Multi-GPU Configuration for B200s
# =====================================

@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU experiments on HiPerGator B200s"""
    # Storage paths for HiPerGator
    data_dir: str = field(default_factory=lambda: f"/blue/{os.environ.get('GROUP', 'default')}/{os.environ.get('USER', 'user')}")
    cache_dir: str = field(default_factory=lambda: f"/blue/{os.environ.get('GROUP', 'default')}/{os.environ.get('USER', 'user')}/cache")
    results_dir: str = field(default_factory=lambda: f"/blue/{os.environ.get('GROUP', 'default')}/{os.environ.get('USER', 'user')}/results")
    
    # GPU configuration for B200s
    available_gpus: List[int] = None
    use_all_gpus: bool = True
    gpu_memory_limit: float = 85.0  # GB per GPU (B200 has 90GB, leave headroom)
    
    # Model configs - LLaMA family focus
    models: List[str] = None
    
    # Evaluation configs
    calibration_samples: int = 256
    eval_samples: int = 512
    batch_size: int = 4  # Increased for B200s
    max_length: int = 2048
    stride: int = 1024
    
    # Pruning configs
    methods: List[str] = None
    skip_modules: List[str] = None
    
    # Task evaluation settings
    eval_tasks: List[str] = None
    task_batch_size: int = 8
    
    # Parallel processing
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    
    # B200 specific optimizations
    use_flash_attention: bool = True
    use_bfloat16: bool = True
    
    def __post_init__(self):
        # Detect available GPUs
        if self.available_gpus is None:
            self.available_gpus = self._detect_free_gpus()
        
        if self.models is None:
            self.models = [
                # LLaMA-1 family
                "huggyllama/llama-7b",
                "huggyllama/llama-13b",
                "huggyllama/llama-30b",
                "huggyllama/llama-65b",
                
                # OpenLLaMA family
                "openlm-research/open_llama_3b",
                "openlm-research/open_llama_7b",
                "openlm-research/open_llama_13b",
                
                # LLaMA-3.1
                "meta-llama/Meta-Llama-3.1-8B",
            ]
        
        if self.methods is None:
            self.methods = ["weight_only", "emp", "structured_emp"]
        
        if self.skip_modules is None:
            self.skip_modules = ['lm_head', 'embed_tokens', 'embed_positions', 'embeddings']
        
        if self.eval_tasks is None:
            self.eval_tasks = [
                "wic", "mrpc", "hellaswag", "arc_easy", 
                "arc_challenge", "winogrande", "boolq", "rte", "mmlu"
            ]
        
        # Create directories
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    def _detect_free_gpus(self) -> List[int]:
        """Detect GPUs - limit to 4 for resource constraints"""
        try:
            # Check CUDA_VISIBLE_DEVICES first
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
                gpus = [int(g) for g in visible if g.strip().isdigit()]
                if gpus:
                    print(f"Using GPUs from CUDA_VISIBLE_DEVICES: {gpus[:4]}")
                    return gpus[:4]  # Limit to 4 GPUs
            
            # Otherwise detect available GPUs
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'],
                capture_output=True, text=True
            )
            
            free_gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                if len(parts) == 2:
                    idx, mem_used = parts
                    if float(mem_used) < 10000:  # Less than 10GB used
                        free_gpus.append(int(idx))
            
            # Limit to 4 GPUs maximum
            free_gpus = free_gpus[:4]
            print(f"Detected free GPUs (limited to 4): {free_gpus}")
            return free_gpus if free_gpus else [0]
            
        except Exception as e:
            print(f"Error detecting GPUs: {e}")
            # Default to first 4 GPUs
            return list(range(min(4, torch.cuda.device_count())))

# =====================================
# Multi-GPU Model Loading with B200 Optimizations
# =====================================

class MultiGPUModelLoader:
    """Load and distribute models across multiple B200 GPUs"""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self.device_map = None
    
    def estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in GB based on name"""
        size_map = {
            "3b": 12,   # OpenLLaMA 3B
            "7b": 28,   # LLaMA-7B, OpenLLaMA-7B
            "8b": 32,   # LLaMA-3.1-8B
            "13b": 52,  # LLaMA-13B, OpenLLaMA-13B
            "30b": 120, # LLaMA-30B (needs 2+ GPUs)
            "65b": 260, # LLaMA-65B (needs 4 GPUs)
            "70b": 280, # OpenLLaMA-70B (needs 4 GPUs with quantization)
        }
        
        for key, size in size_map.items():
            if key in model_name.lower():
                return size
        return 28  # Default to 7B size
    
    def determine_gpu_allocation(self, model_name: str) -> Dict:
        """Determine optimal GPU allocation for model on 4 B200s"""
        model_size = self.estimate_model_size(model_name)
        num_gpus = len(self.config.available_gpus)
        available_memory = num_gpus * self.config.gpu_memory_limit
        
        print(f"Model: {model_name}")
        print(f"Estimated size: {model_size}GB")
        print(f"Available: {available_memory}GB across {num_gpus} GPUs")
        
        # Single GPU models (up to 85GB)
        if model_size <= 85:
            return {
                "device_map": f"cuda:{self.config.available_gpus[0]}",
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            }
        
        # Multi-GPU models (up to 340GB with 4 GPUs)
        elif model_size <= available_memory:
            # For 4 GPUs, distribute evenly
            device_map = "balanced"
            max_memory = {i: "85GB" for i in self.config.available_gpus}
            
            return {
                "device_map": device_map,
                "max_memory": max_memory,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                "offload_folder": f"{self.config.cache_dir}/offload"
            }
        
        # Large models needing quantization
        else:
            return self._get_quantized_config_4gpu(model_name, model_size)
    
    def _get_quantized_config_4gpu(self, model_name: str, model_size: float) -> Dict:
        """Quantization config for large models on 4 B200s"""
        
        if "65b" in model_name.lower() or "70b" in model_name.lower():
            # 65B/70B models need 4-bit quantization on 4 GPUs
            return {
                "device_map": "balanced",
                "max_memory": {i: "85GB" for i in self.config.available_gpus},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                ),
                "torch_dtype": torch.bfloat16,
                "offload_folder": f"{self.config.cache_dir}/offload"
            }
        elif "30b" in model_name.lower():
            # 30B models can use 8-bit on 4 GPUs
            return {
                "device_map": "balanced", 
                "max_memory": {i: "85GB" for i in self.config.available_gpus},
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16
                ),
                "torch_dtype": torch.bfloat16
            }
        else:
            # Default for other large models
            return {
                "device_map": "auto",
                "max_memory": {i: "85GB" for i in self.config.available_gpus},
                "torch_dtype": torch.bfloat16
            }
    
    def load_model_and_tokenizer(self, model_name: str):
        """Load model with B200 optimizations"""
        
        print(f"\n{'='*60}")
        print(f"Loading {model_name} on B200 GPUs...")
        print(f"{'='*60}")
        
        # Check GPU availability
        for i in range(min(torch.cuda.device_count(), 8)):
            props = torch.cuda.get_device_properties(i)
            mem_free = torch.cuda.mem_get_info(i)[0] / 1024**3
            mem_total = torch.cuda.mem_get_info(i)[1] / 1024**3
            print(f"GPU {i} ({props.name}): {mem_free:.1f}/{mem_total:.1f} GB free")
        
        # Get loading configuration
        load_config = self.determine_gpu_allocation(model_name)
        
        print(f"Loading strategy: {load_config.get('device_map', 'single GPU')}")
        if 'max_memory' in load_config:
            print(f"Memory allocation: {load_config['max_memory']}")
        
        # Determine dtype for B200s
        dtype = torch.bfloat16 if self.config.use_bfloat16 and torch.cuda.is_bf16_supported() else torch.float16
        
        # Load model with optimizations
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_flash_attention_2=self.config.use_flash_attention,
                **load_config
            )
        except Exception as e:
            print(f"Error loading with flash attention, retrying without: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                **load_config
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
            use_fast=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Print model distribution
        if hasattr(model, 'hf_device_map'):
            print("\nModel distribution across GPUs:")
            device_counts = {}
            for name, device in model.hf_device_map.items():
                device_str = str(device)
                device_counts[device_str] = device_counts.get(device_str, 0) + 1
            for device, count in device_counts.items():
                print(f"  {device}: {count} layers")
        
        return model, tokenizer

# =====================================
# Parallel NEFF Pruning with B200 Optimizations
# =====================================

class ParallelNEFFPruner:
    """NEFF pruning with multi-GPU B200 support"""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
    
    def compute_neff_batch(self, weights: torch.Tensor, batch_size: int = 2000) -> torch.Tensor:
        """Compute NEFF in batches for memory efficiency (increased for B200s)"""
        n_rows = weights.shape[0]
        neff_values = []
        
        for i in range(0, n_rows, batch_size):
            batch = weights[i:min(i+batch_size, n_rows)]
            abs_batch = batch.abs()
            batch_sum = abs_batch.sum(dim=1, keepdim=True).clamp_min(1e-12)
            p = abs_batch / batch_sum
            p_squared_sum = (p ** 2).sum(dim=1)
            neff = torch.floor(1.0 / p_squared_sum.clamp_min(1e-12))
            neff = neff.clamp(min=1, max=weights.shape[1])
            neff_values.append(neff)
        
        return torch.cat(neff_values)
    
    @torch.no_grad()
    def collect_activation_statistics_parallel(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_batches: int = 64
    ) -> Dict[nn.Linear, torch.Tensor]:
        """Collect activation statistics using multiple B200 GPUs"""
        
        print("Collecting activation statistics on B200s...")
        
        stats_sum = {}
        stats_count = {}
        handles = []
        
        def register_hook(module: nn.Linear, name: str):
            device = next(module.parameters()).device
            dtype = torch.float32 if self.config.use_bfloat16 else torch.float32
            stats_sum[module] = torch.zeros(module.in_features, device=device, dtype=dtype)
            stats_count[module] = 0
            
            def hook(mod, inp, out):
                x = inp[0].detach()
                
                if x.dim() == 3:
                    x = x.reshape(-1, x.size(-1))
                elif x.dim() != 2:
                    return
                
                stats_sum[mod] += x.abs().sum(dim=0).to(dtype)
                stats_count[mod] += x.size(0)
            
            return module.register_forward_hook(hook)
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if not any(skip in name for skip in self.config.skip_modules):
                    handles.append(register_hook(module, name))
        
        # Collect statistics
        model.eval()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.config.use_bfloat16 else torch.float16):
            for i, batch in enumerate(tqdm(dataloader, desc="Calibration", total=min(num_batches, len(dataloader)))):
                if i >= num_batches:
                    break
                
                try:
                    if isinstance(batch, dict) and 'input_ids' in batch:
                        if hasattr(model, 'hf_device_map'):
                            first_device = list(set(model.hf_device_map.values()))[0]
                            if isinstance(first_device, int):
                                input_ids = batch['input_ids'].to(f'cuda:{first_device}')
                            else:
                                input_ids = batch['input_ids'].to(first_device)
                        else:
                            input_ids = batch['input_ids'].to('cuda:0')
                    else:
                        input_ids = batch.to('cuda:0')
                    
                    _ = model(input_ids, use_cache=False)
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM at batch {i}, clearing cache...")
                    torch.cuda.empty_cache()
                    continue
        
        # Clean up
        for handle in handles:
            handle.remove()
        
        # Compute means
        activation_means = {}
        for module in stats_sum:
            if stats_count[module] > 0:
                activation_means[module] = (stats_sum[module] / stats_count[module]).clamp_min(1e-12)
        
        return activation_means
    
    def prune_model_parallel(
        self,
        model: nn.Module,
        method: str = "weight_only",
        activation_stats: Optional[Dict] = None
    ) -> Tuple[nn.Module, Dict]:
        """Prune model with parallel processing on B200s"""
        
        print(f"\nApplying {method} pruning...")
        
        pruning_info = {
            'method': method,
            'layer_sparsity': {},
            'total_params': 0,
            'pruned_params': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Process layers
        with torch.no_grad():
            for name, module in tqdm(model.named_modules(), desc="Pruning layers"):
                if not isinstance(module, nn.Linear):
                    continue
                if any(skip in name for skip in self.config.skip_modules):
                    continue
                
                weight = module.weight.data
                
                # Get activation statistics
                act_mean = None
                if activation_stats and module in activation_stats:
                    act_mean = activation_stats[module]
                
                # Compute importance
                if method == "weight_only":
                    neff = self.compute_neff_batch(weight)
                    importance = weight.abs()
                
                elif method == "emp" and act_mean is not None:
                    effective_weight = weight.abs() * act_mean.unsqueeze(0)
                    weight_sum = effective_weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    p = effective_weight / weight_sum
                    neff = torch.floor(1.0 / (p.pow(2).sum(dim=1))).clamp(min=1, max=weight.size(1)).long()
                    importance = effective_weight
                
                elif method == "structured_emp" and act_mean is not None:
                    effective_weight = weight.abs() * act_mean.unsqueeze(0)
                    neff = self.compute_neff_batch(effective_weight)
                    group_size = 4
                    neff = (neff // group_size) * group_size
                    neff = neff.clamp(min=group_size, max=weight.size(1))
                    importance = effective_weight
                
                else:
                    neff = self.compute_neff_batch(weight)
                    importance = weight.abs()
                
                # Apply pruning
                mask = torch.zeros_like(weight, dtype=torch.bool)
                
                # Process in chunks for memory efficiency
                chunk_size = 200  # Increased for B200s
                for i in range(0, weight.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, weight.shape[0])
                    chunk_importance = importance[i:end_idx]
                    chunk_neff = neff[i:end_idx] if isinstance(neff, torch.Tensor) else neff
                    
                    _, indices = torch.sort(chunk_importance, dim=1, descending=True)
                    
                    for j, row_idx in enumerate(range(i, end_idx)):
                        if isinstance(chunk_neff, torch.Tensor):
                            n = chunk_neff[j].item()
                        else:
                            n = chunk_neff
                        mask[row_idx, indices[j, :int(n)]] = True
                
                # Apply mask
                module.weight.data *= mask.float()
                
                # Statistics
                num_zeros = (~mask).sum().item()
                num_params = weight.numel()
                sparsity = num_zeros / num_params if num_params > 0 else 0
                
                pruning_info['layer_sparsity'][name] = sparsity
                pruning_info['total_params'] += num_params
                pruning_info['pruned_params'] += num_zeros
        
        pruning_info['overall_sparsity'] = (
            pruning_info['pruned_params'] / pruning_info['total_params'] 
            if pruning_info['total_params'] > 0 else 0
        )
        
        return model, pruning_info

# =====================================
# Perplexity Evaluation
# =====================================

class MultiGPUPerplexityEvaluator:
    """Perplexity evaluation with multi-GPU support"""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        tokenizer,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        split: str = "test"
    ) -> Dict:
        """Evaluate perplexity on WikiText"""
        
        print(f"\nEvaluating perplexity on {dataset_name}...")
        model.eval()
        
        # Load dataset
        dataset = load_dataset(
            dataset_name, 
            dataset_config, 
            split=split,
            cache_dir=self.config.cache_dir
        )
        
        if self.config.eval_samples:
            dataset = dataset.select(range(min(self.config.eval_samples, len(dataset))))
        
        # Prepare text
        text = "\n\n".join(dataset["text"])
        encodings = tokenizer(text, return_tensors="pt")
        
        # Determine device
        if hasattr(model, 'hf_device_map'):
            first_device = list(set(model.hf_device_map.values()))[0]
            if isinstance(first_device, int):
                device = f'cuda:{first_device}'
            else:
                device = first_device
        else:
            device = 'cuda:0'
        
        total_loss = 0
        total_tokens = 0
        
        # Process in windows
        max_length = min(self.config.max_length, 2048)
        stride = self.config.stride
        seq_len = encodings.input_ids.size(1)
        
        progress = tqdm(range(0, seq_len, stride), desc="Computing perplexity")
        
        for begin_idx in progress:
            end_idx = min(begin_idx + max_length, seq_len)
            
            input_ids = encodings.input_ids[:, begin_idx:end_idx].to(device)
            target_ids = input_ids.clone()
            
            if begin_idx != 0:
                target_ids[:, :-stride] = -100
            
            try:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.config.use_bfloat16 else torch.float16):
                    outputs = model(input_ids, labels=target_ids, use_cache=False)
                    num_tokens_in_sequence = (target_ids != -100).sum().item()
                    
                    if begin_idx == 0:
                        num_tokens_to_add = num_tokens_in_sequence
                    else:
                        num_tokens_to_add = min(stride, num_tokens_in_sequence)
                    
                    total_loss += outputs.loss.item() * num_tokens_to_add
                    total_tokens += num_tokens_to_add
                
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at position {begin_idx}, skipping...")
                torch.cuda.empty_cache()
                continue
            
            if total_tokens > 0:
                current_ppl = math.exp(total_loss / total_tokens)
                progress.set_postfix({"ppl": f"{current_ppl:.2f}"})
            
            if end_idx >= seq_len:
                break
        
        perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
        
        return {
            'perplexity': perplexity,
            'loss': total_loss / total_tokens if total_tokens > 0 else float('inf'),
            'total_tokens': total_tokens
        }

# =====================================
# Task Evaluation (for LLaMA-7B)
# =====================================

class TaskEvaluator:
    """Evaluate model on various NLP tasks"""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self.task_configs = {
            'wic': {'dataset': 'super_glue', 'subset': 'wic', 'metric': 'accuracy'},
            'mrpc': {'dataset': 'glue', 'subset': 'mrpc', 'metric': 'accuracy'},
            'hellaswag': {'dataset': 'hellaswag', 'subset': None, 'metric': 'accuracy'},
            'arc_easy': {'dataset': 'ai2_arc', 'subset': 'ARC-Easy', 'metric': 'accuracy'},
            'arc_challenge': {'dataset': 'ai2_arc', 'subset': 'ARC-Challenge', 'metric': 'accuracy'},
            'winogrande': {'dataset': 'winogrande', 'subset': 'winogrande_xl', 'metric': 'accuracy'},
            'boolq': {'dataset': 'super_glue', 'subset': 'boolq', 'metric': 'accuracy'},
            'rte': {'dataset': 'super_glue', 'subset': 'rte', 'metric': 'accuracy'},
            'mmlu': {'dataset': 'cais/mmlu', 'subset': 'all', 'metric': 'accuracy'}
        }
    
    @torch.no_grad()
    def evaluate_task(self, model, tokenizer, task_name: str) -> Dict:
        """Evaluate model on a specific task"""
        
        print(f"\nEvaluating {task_name}...")
        model.eval()
        
        task_config = self.task_configs.get(task_name)
        if not task_config:
            print(f"Unknown task: {task_name}")
            return {'error': f'Unknown task: {task_name}'}
        
        try:
            # Load dataset
            if task_config['subset']:
                dataset = load_dataset(
                    task_config['dataset'],
                    task_config['subset'],
                    split='validation' if task_name != 'mmlu' else 'test',
                    cache_dir=self.config.cache_dir
                )
            else:
                dataset = load_dataset(
                    task_config['dataset'],
                    split='validation',
                    cache_dir=self.config.cache_dir
                )
            
            # Sample if needed
            if len(dataset) > 1000:
                dataset = dataset.select(range(1000))
            
            # Task-specific evaluation logic
            if task_name in ['arc_easy', 'arc_challenge', 'hellaswag', 'winogrande']:
                return self._evaluate_multiple_choice(model, tokenizer, dataset, task_name)
            elif task_name in ['boolq', 'wic', 'rte']:
                return self._evaluate_classification(model, tokenizer, dataset, task_name)
            elif task_name == 'mrpc':
                return self._evaluate_sentence_pair(model, tokenizer, dataset)
            elif task_name == 'mmlu':
                return self._evaluate_mmlu(model, tokenizer, dataset)
            else:
                return {'error': f'Evaluation not implemented for {task_name}'}
                
        except Exception as e:
            print(f"Error evaluating {task_name}: {e}")
            return {'error': str(e)}
    
    def _evaluate_multiple_choice(self, model, tokenizer, dataset, task_name):
        """Evaluate multiple choice tasks"""
        correct = 0
        total = 0
        
        # Determine device
        if hasattr(model, 'hf_device_map'):
            device = f"cuda:{list(model.hf_device_map.values())[0]}"
        else:
            device = 'cuda:0'
        
        for item in tqdm(dataset, desc=f"Evaluating {task_name}"):
            if task_name in ['arc_easy', 'arc_challenge']:
                question = item['question']
                choices = item['choices']['text']
                answer_idx = item['choices']['label'].index(item['answerKey'])
            elif task_name == 'hellaswag':
                question = item['ctx']
                choices = item['endings']
                answer_idx = int(item['label'])
            elif task_name == 'winogrande':
                question = item['sentence']
                choices = [item['option1'], item['option2']]
                answer_idx = int(item['answer']) - 1
            
            # Score each choice
            scores = []
            for choice in choices:
                prompt = f"Question: {question}\nAnswer: {choice}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    scores.append(-outputs.loss.item())
            
            predicted_idx = np.argmax(scores)
            if predicted_idx == answer_idx:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {'accuracy': accuracy, 'correct': correct, 'total': total}

# =====================================
# Main Experiment Runner
# =====================================

class MultiGPUExperimentRunner:
    """Run experiments with multi-GPU B200 support"""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self.model_loader = MultiGPUModelLoader(config)
        self.pruner = ParallelNEFFPruner(config)
        self.perplexity_evaluator = MultiGPUPerplexityEvaluator(config)
        self.task_evaluator = TaskEvaluator(config)
    
    def prepare_calibration_data(self, tokenizer, num_samples: int = 256):
        """Prepare calibration dataset"""
        
        dataset = load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            cache_dir=self.config.cache_dir,
            streaming=True
        )
        
        texts = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            if item["text"].strip():
                texts.append(item["text"][:2048])
        
        # Tokenize
        tokenized = []
        for text in texts:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
                padding=False
            )
            tokenized.append({"input_ids": tokens["input_ids"].squeeze(0)})
        
        return DataLoader(
            tokenized, 
            batch_size=1, 
            shuffle=False,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory
        )
    
    def run_single_model(self, model_name: str, eval_tasks: bool = False) -> Dict:
        """Run experiments on a single model"""
        
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        results = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'num_gpus_used': len(self.config.available_gpus),
            'gpu_ids': self.config.available_gpus
        }
        
        try:
            # Load model
            model, tokenizer = self.model_loader.load_model_and_tokenizer(model_name)
            
            # Get model info
            total_params = sum(p.numel() for p in model.parameters())
            results['total_parameters'] = total_params
            results['model_size_gb'] = total_params * 2 / (1024**3)  # FP16/BF16
            
            print(f"\nModel Statistics:")
            print(f"  Parameters: {total_params:,}")
            print(f"  Size: {results['model_size_gb']:.2f} GB")
            
            # Evaluate WikiText perplexity for all models
            print("\n" + "="*40)
            print("WikiText Perplexity Evaluation")
            print("="*40)
            
            wikitext_results = self.perplexity_evaluator.evaluate(model, tokenizer)
            results['wikitext_perplexity'] = wikitext_results
            print(f"WikiText Perplexity: {wikitext_results['perplexity']:.2f}")
            
            # For LLaMA-7B, also evaluate on tasks
            if eval_tasks and ("7b" in model_name.lower() or "7B" in model_name):
                print("\n" + "="*40)
                print("Task Evaluation for LLaMA-7B")
                print("="*40)
                
                results['task_evaluation'] = {}
                for task in self.config.eval_tasks:
                    print(f"\nEvaluating {task}...")
                    task_result = self.task_evaluator.evaluate_task(model, tokenizer, task)
                    results['task_evaluation'][task] = task_result
                    if 'accuracy' in task_result:
                        print(f"  {task} Accuracy: {task_result['accuracy']:.4f}")
            
            # Test pruning methods
            print("\n" + "="*40)
            print("NEFF Pruning Evaluation")
            print("="*40)
            
            # Prepare calibration data
            calib_loader = self.prepare_calibration_data(tokenizer, self.config.calibration_samples)
            
            for method in self.config.methods:
                print(f"\n{'='*30}")
                print(f"Testing {method.upper()} Pruning")
                print(f"{'='*30}")
                
                # Clone model for pruning
                pruned_model = copy.deepcopy(model)
                
                # Collect activation stats if needed
                activation_stats = None
                if "emp" in method:
                    activation_stats = self.pruner.collect_activation_statistics_parallel(
                        pruned_model,
                        calib_loader,
                        num_batches=min(64, len(calib_loader))
                    )
                
                # Prune model
                pruned_model, pruning_info = self.pruner.prune_model_parallel(
                    pruned_model,
                    method=method,
                    activation_stats=activation_stats
                )
                
                print(f"Overall Sparsity: {pruning_info['overall_sparsity']:.2%}")
                
                # Evaluate pruned model
                pruned_wikitext = self.perplexity_evaluator.evaluate(pruned_model, tokenizer)
                
                # Store results
                results[f'{method}_pruning'] = {
                    'pruning_info': pruning_info,
                    'wikitext_perplexity': pruned_wikitext,
                    'perplexity_increase': pruned_wikitext['perplexity'] - wikitext_results['perplexity'],
                    'relative_increase': (pruned_wikitext['perplexity'] / wikitext_results['perplexity'] - 1) * 100
                }
                
                print(f"Pruned WikiText PPL: {pruned_wikitext['perplexity']:.2f} "
                      f"(+{results[f'{method}_pruning']['perplexity_increase']:.2f}, "
                      f"{results[f'{method}_pruning']['relative_increase']:.1f}% increase)")
                
                # If LLaMA-7B, evaluate pruned model on tasks
                if eval_tasks and ("7b" in model_name.lower() or "7B" in model_name):
                    print(f"\nEvaluating pruned model on tasks...")
                    results[f'{method}_pruning']['task_evaluation'] = {}
                    for task in ['arc_easy', 'boolq']:  # Subset for efficiency
                        task_result = self.task_evaluator.evaluate_task(pruned_model, tokenizer, task)
                        results[f'{method}_pruning']['task_evaluation'][task] = task_result
                        if 'accuracy' in task_result:
                            orig_acc = results['task_evaluation'][task]['accuracy']
                            pruned_acc = task_result['accuracy']
                            print(f"  {task}: {pruned_acc:.4f} (orig: {orig_acc:.4f}, "
                                  f"Δ: {pruned_acc - orig_acc:+.4f})")
                
                # Clean up
                del pruned_model
                torch.cuda.empty_cache()
                gc.collect()
            
            # Clean up original model
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        # Save results
        self.save_results(results, model_name)
        
        return results
    
    def save_results(self, results: Dict, model_name: str):
        """Save results to file"""
        
        safe_name = model_name.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.config.results_dir}/neff_{safe_name}_{timestamp}.json"
        
        # Convert non-serializable objects
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            return obj
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
        
        print(f"\nResults saved to: {filename}")
    
    def run_all_experiments(self):
        """Run experiments on all configured models"""
        
        print("\n" + "="*60)
        print("NEFF MULTI-GPU EXPERIMENT SUITE")
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Models to test: {len(self.config.models)}")
        print(f"GPUs available: {self.config.available_gpus}")
        print("="*60)
        
        all_results = []
        
        for i, model_name in enumerate(self.config.models, 1):
            print(f"\n[{i}/{len(self.config.models)}] Processing {model_name}")
            
            # Check if it's LLaMA-7B for task evaluation
            eval_tasks = "llama-7b" in model_name.lower() or "llama/llama-7b" in model_name.lower()
            
            try:
                result = self.run_single_model(model_name, eval_tasks=eval_tasks)
                all_results.append(result)
            except Exception as e:
                print(f"Failed on {model_name}: {e}")
                all_results.append({
                    'model': model_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                continue
            
            # Memory cleanup between models
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(5)  # Brief pause between models
        
        # Save summary
        summary_file = f"{self.config.results_dir}/neff_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*60)
        print("All experiments completed!")
        print(f"Summary saved to: {summary_file}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

# =====================================
# CLI Interface
# =====================================

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU NEFF Pruning for LLMs on B200s")
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to evaluate (default: all LLaMA family models)"
    )
    
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="GPU IDs to use (default: auto-detect all 8 B200s)"
    )
    
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=256,
        help="Number of calibration samples"
    )
    
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=512,
        help="Number of evaluation samples"
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["weight_only", "emp", "structured_emp"],
        help="Pruning methods to test"
    )
    
    parser.add_argument(
        "--eval-tasks",
        nargs="+",
        default=None,
        help="Tasks to evaluate (for LLaMA-7B)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base directory for data storage"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--single-model",
        type=str,
        default=None,
        help="Run only a single model"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = MultiGPUConfig(
        available_gpus=args.gpus,
        models=args.models,
        calibration_samples=args.calibration_samples,
        eval_samples=args.eval_samples,
        methods=args.methods,
        eval_tasks=args.eval_tasks,
        batch_size=args.batch_size
    )
    
    if args.data_dir:
        config.data_dir = args.data_dir
        config.cache_dir = f"{args.data_dir}/cache"
        config.results_dir = f"{args.data_dir}/results"
    
    # Print system info
    print("\n" + "="*60)
    print("NEFF PRUNING SYSTEM - B200 OPTIMIZED")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Available GPUs: {config.available_gpus}")
    print(f"Data directory: {config.data_dir}")
    print(f"Models to test: {len(config.models)}")
    print(f"Methods: {config.methods}")
    print("="*60)
    
    # Run experiments
    runner = MultiGPUExperimentRunner(config)
    
    if args.single_model:
        # Run single model
        eval_tasks = "7b" in args.single_model.lower()
        runner.run_single_model(args.single_model, eval_tasks=eval_tasks)
    else:
        # Run all models
        runner.run_all_experiments()

if __name__ == "__main__":
    main()
    
    def _evaluate_classification(self, model, tokenizer, dataset, task_name):
        """Evaluate binary classification tasks"""
        correct = 0
        total = 0
        
        # Determine device
        if hasattr(model, 'hf_device_map'):
            device = f"cuda:{list(model.hf_device_map.values())[0]}"
        else:
            device = 'cuda:0'
        
        for item in tqdm(dataset, desc=f"Evaluating {task_name}"):
            if task_name == 'boolq':
                text = f"Question: {item['question']}\nPassage: {item['passage']}\nAnswer:"
                label = item['label']
            elif task_name == 'wic':
                text = f"Sentence 1: {item['sentence1']}\nSentence 2: {item['sentence2']}\nSame meaning?"
                label = item['label']
            elif task_name == 'rte':
                text = f"Premise: {item['premise']}\nHypothesis: {item['hypothesis']}\nEntailment?"
                label = item['label']
            
            # Score True/False
            scores = []
            for answer in ['Yes', 'No']:
                prompt = text + f" {answer}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    scores.append(-outputs.loss.item())
            
            predicted = 1 if scores[0] > scores[1] else 0
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {'accuracy': accuracy, 'correct': correct, 'total': total}
    
    def _evaluate_sentence_pair(self, model, tokenizer, dataset):
        """Evaluate MRPC (sentence pair classification)"""
        correct = 0
        total = 0
        
        # Determine device
        if hasattr(model, 'hf_device_map'):
            device = f"cuda:{list(model.hf_device_map.values())[0]}"
        else:
            device = 'cuda:0'
        
        for item in tqdm(dataset, desc="Evaluating MRPC"):
            text = f"Sentence 1: {item['sentence1']}\nSentence 2: {item['sentence2']}\nAre these paraphrases?"
            label = item['label']
            
            # Score Yes/No
            scores = []
            for answer in ['Yes', 'No']:
                prompt = text + f" {answer}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    scores.append(-outputs.loss.item())
            
            predicted = 1 if scores[0] > scores[1] else 0
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {'accuracy': accuracy, 'correct': correct, 'total': total}
    
    def _evaluate_mmlu(self, model, tokenizer, dataset):
        """Evaluate MMLU (5-shot)"""
        correct = 0
        total = 0
        
        # Determine device
        if hasattr(model, 'hf_device_map'):
            device = f"cuda:{list(model.hf_device_map.values())[0]}"
        else:
            device = 'cuda:0'
        
        # Sample subset for efficiency
        dataset = dataset.select(range(min(500, len(dataset))))
        
        for item in tqdm(dataset, desc="Evaluating MMLU"):
            question = item['question']
            choices = [item['choices'][i] for i in range(len(item['choices']))]
            answer_idx = item['answer']
            
            # Score each choice
            scores = []
            for i, choice in enumerate(choices):
                prompt = f"Question: {question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer: {chr(65+i)}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    scores.append(-outputs.loss.item())
            
            predicted_idx = np.argmax(scores)
            if predicted_idx == answer_idx:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {'accuracy': accuracy, 'correct': correct, 'total': total}