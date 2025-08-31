#!/usr/bin/env python3
"""
NEFF Pruning for Large Language Models - Multi-GPU H100 Production Script
Optimized for 8x H100 80GB GPUs
For ICLR 2026 submission
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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
from datetime import datetime
import gc
import argparse
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# HuggingFace imports
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    LlamaForCausalLM, LlamaTokenizer,
    MistralForCausalLM,
    BitsAndBytesConfig
)
from datasets import load_dataset
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import warnings
warnings.filterwarnings('ignore')

# =====================================
# Multi-GPU Configuration
# =====================================

@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU experiments"""
    data_dir: str = "/mnt/data/wyx"
    cache_dir: str = "/mnt/data/wyx/cache"
    results_dir: str = "/mnt/data/wyx/results"
    
    # GPU configuration
    available_gpus: List[int] = None
    use_all_gpus: bool = False
    gpu_memory_limit: float = 75.0  # GB per GPU
    
    # Model configs - expanded for multi-GPU
    models: List[str] = None
    
    # Evaluation configs
    calibration_samples: int = 256
    eval_samples: int = 512
    batch_size: int = 2
    max_length: int = 2048
    stride: int = 1024
    
    # Pruning configs
    methods: List[str] = None
    skip_modules: List[str] = None
    
    # Parallel processing
    num_workers: int = 4
    prefetch_factor: int = 2
    
    def __post_init__(self):
        # Detect available GPUs
        if self.available_gpus is None:
            # Check which GPUs are free (less than 10GB used)
            self.available_gpus = self._detect_free_gpus()
        
        if self.models is None:
            self.models = [
                # Small models (single GPU)
                "facebook/opt-125m",
                "facebook/opt-1.3b", 
                "facebook/opt-6.7b",
                
                # Medium models (1-2 GPUs)
                "meta-llama/Llama-2-7b-hf",
                "mistralai/Mistral-7B-v0.1",
                "tiiuae/falcon-7b",
                "bigscience/bloom-7b1",
                
                # Large models (2-4 GPUs)
                "meta-llama/Llama-2-13b-hf",
                "EleutherAI/gpt-neox-20b",
                "mistralai/Mixtral-8x7B-v0.1",
                
                # Extra large models (4-8 GPUs)
                "meta-llama/Llama-2-70b-hf",
                "tiiuae/falcon-40b",
            ]
        
        if self.methods is None:
            self.methods = ["weight_only", "emp", "structured_emp"]
        
        if self.skip_modules is None:
            self.skip_modules = ['lm_head', 'embed_tokens', 'embed_positions', 'embeddings']
        
        # Create directories
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    def _detect_free_gpus(self) -> List[int]:
        """Detect GPUs with low memory usage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'],
                capture_output=True, text=True
            )
            
            free_gpus = []
            for line in result.stdout.strip().split('\n'):
                idx, mem_used = line.split(',')
                if float(mem_used) < 10000:  # Less than 10GB used
                    free_gpus.append(int(idx))
            
            print(f"Detected free GPUs: {free_gpus}")
            return free_gpus if free_gpus else [0]  # Default to GPU 0
            
        except Exception as e:
            print(f"Error detecting GPUs: {e}")
            return [0]

# =====================================
# Multi-GPU Model Loading
# =====================================

class MultiGPUModelLoader:
    """Load and distribute models across multiple GPUs"""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self.device_map = None
    
    def estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in GB based on name"""
        size_map = {
            "125m": 0.5, "350m": 1.4, "1.3b": 5.2, "2.7b": 10.8,
            "6.7b": 26.8, "7b": 28, "13b": 52, "20b": 80,
            "30b": 120, "40b": 160, "70b": 280, "175b": 700
        }
        
        for key, size in size_map.items():
            if key in model_name.lower():
                return size
        return 28  # Default to 7B size
    
    def determine_gpu_allocation(self, model_name: str) -> Dict:
        """Determine optimal GPU allocation for model"""
        model_size = self.estimate_model_size(model_name)
        available_memory = len(self.config.available_gpus) * self.config.gpu_memory_limit
        
        if model_size <= self.config.gpu_memory_limit:
            # Single GPU is enough
            return {"device_map": f"cuda:{self.config.available_gpus[0]}"}
        
        elif model_size <= available_memory:
            # Use multiple GPUs
            num_gpus_needed = min(
                math.ceil(model_size / self.config.gpu_memory_limit),
                len(self.config.available_gpus)
            )
            
            # Create device map for model parallelism
            device_map = "auto"
            max_memory = {
                i: f"{self.config.gpu_memory_limit}GB" 
                for i in self.config.available_gpus[:num_gpus_needed]
            }
            
            return {
                "device_map": device_map,
                "max_memory": max_memory,
                "offload_folder": f"{self.config.cache_dir}/offload"
            }
        
        else:
            # Need quantization
            return self._get_quantized_config(model_name, model_size)
    
    def _get_quantized_config(self, model_name: str, model_size: float) -> Dict:
        """Get quantization config for large models"""
        
        if model_size > 200:  # 70B+ models
            config = {
                "device_map": "auto",
                "max_memory": {i: f"{self.config.gpu_memory_limit}GB" 
                              for i in self.config.available_gpus},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                ),
                "offload_folder": f"{self.config.cache_dir}/offload"
            }
        elif model_size > 50:  # 13B-40B models
            config = {
                "device_map": "auto",
                "max_memory": {i: f"{self.config.gpu_memory_limit}GB" 
                              for i in self.config.available_gpus[:4]},
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            }
        else:
            config = {
                "device_map": "auto",
                "max_memory": {i: f"{self.config.gpu_memory_limit}GB" 
                              for i in self.config.available_gpus[:2]}
            }
        
        return config
    
    def load_model_and_tokenizer(self, model_name: str):
        """Load model with optimal GPU distribution"""
        
        print(f"\nLoading {model_name}...")
        print(f"Available GPUs: {self.config.available_gpus}")
        
        # Get loading configuration
        load_config = self.determine_gpu_allocation(model_name)
        
        print(f"Loading strategy: {load_config.get('device_map', 'single GPU')}")
        if 'max_memory' in load_config:
            print(f"Memory allocation: {load_config['max_memory']}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=self.config.cache_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **load_config
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Print model distribution
        if hasattr(model, 'hf_device_map'):
            print("\nModel distribution across GPUs:")
            for name, device in model.hf_device_map.items():
                if isinstance(device, int):
                    print(f"  {name}: GPU {device}")
                else:
                    print(f"  {name}: {device}")
        
        return model, tokenizer

# =====================================
# Parallel NEFF Pruning
# =====================================

class ParallelNEFFPruner:
    """NEFF pruning with multi-GPU support"""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
    
    def compute_neff_batch(self, weights: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
        """Compute NEFF in batches for memory efficiency"""
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
        """Collect activation statistics using multiple GPUs"""
        
        print("Collecting activation statistics (parallel)...")
        
        stats_sum = {}
        stats_count = {}
        handles = []
        
        def register_hook(module: nn.Linear, name: str):
            # Determine device for this module
            device = next(module.parameters()).device
            stats_sum[module] = torch.zeros(module.in_features, device=device, dtype=torch.float32)
            stats_count[module] = 0
            
            def hook(mod, inp, out):
                x = inp[0].detach()
                
                if x.dim() == 3:
                    x = x.reshape(-1, x.size(-1))
                elif x.dim() != 2:
                    return
                
                # Accumulate on the same device as the module
                stats_sum[mod] += x.abs().sum(dim=0)
                stats_count[mod] += x.size(0)
            
            return module.register_forward_hook(hook)
        
        # Register hooks for all Linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if not any(skip in name for skip in self.config.skip_modules):
                    handles.append(register_hook(module, name))
        
        # Collect statistics
        model.eval()
        
        with torch.cuda.amp.autocast():
            for i, batch in enumerate(tqdm(dataloader, desc="Calibration", total=min(num_batches, len(dataloader)))):
                if i >= num_batches:
                    break
                
                try:
                    if isinstance(batch, dict) and 'input_ids' in batch:
                        # Move to first GPU in model's device map
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
                    print(f"OOM at batch {i}, skipping...")
                    torch.cuda.empty_cache()
                    continue
        
        # Clean up hooks
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
        """Prune model with parallel processing"""
        
        print(f"\nApplying {method} pruning (parallel)...")
        
        pruning_info = {
            'method': method,
            'layer_sparsity': {},
            'layer_neff': {},
            'total_params': 0,
            'pruned_params': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Group layers by device for parallel processing
        device_layers = {}
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(skip in name for skip in self.config.skip_modules):
                continue
            
            device = str(next(module.parameters()).device)
            if device not in device_layers:
                device_layers[device] = []
            device_layers[device].append((name, module))
        
        # Process layers in parallel per device
        with torch.no_grad():
            for device, layers in device_layers.items():
                print(f"Processing {len(layers)} layers on {device}")
                
                for name, module in tqdm(layers, desc=f"Pruning on {device}"):
                    weight = module.weight.data
                    
                    # Get activation statistics if available
                    act_mean = None
                    if activation_stats and module in activation_stats:
                        act_mean = activation_stats[module]
                    
                    # Compute importance based on method
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
                    
                    # Create and apply mask
                    mask = torch.zeros_like(weight, dtype=torch.bool)
                    
                    # Process in chunks
                    chunk_size = 100
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
                    
                    # Calculate statistics
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
# Multi-GPU Perplexity Evaluation
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
        """Evaluate perplexity with multi-GPU model"""
        
        print(f"\nEvaluating perplexity on {dataset_name}/{dataset_config}...")
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
        
        # Determine first device
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
        num_sequences = 0
        
        # Process in windows
        max_length = min(self.config.max_length, 2048)
        stride = self.config.stride
        
        seq_len = encodings.input_ids.size(1)
        progress = tqdm(range(0, seq_len, stride), desc="Computing perplexity")
        
        for begin_idx in progress:
            end_idx = min(begin_idx + max_length, seq_len)
            trg_len = end_idx - begin_idx
            
            input_ids = encodings.input_ids[:, begin_idx:end_idx].to(device)
            target_ids = input_ids.clone()
            
            if begin_idx != 0:
                target_ids[:, :-stride] = -100
            
            try:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, labels=target_ids, use_cache=False)
                    num_tokens_in_sequence = (target_ids != -100).sum().item()
                    if begin_idx == 0:
                        num_tokens_to_add = num_tokens_in_sequence
                    else:
                        num_tokens_to_add = min(stride, num_tokens_in_sequence)
                    
                    total_loss += outputs.loss.item() * num_tokens_to_add
                    total_tokens += num_tokens_to_add
                    num_sequences += 1
                
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at position {begin_idx}, skipping...")
                torch.cuda.empty_cache()
                continue
            
            # Update progress
            if num_sequences > 0:
                current_ppl = math.exp(total_loss / total_tokens)
                progress.set_postfix({"ppl": f"{current_ppl:.2f}"})
            
            if end_idx >= seq_len:
                break
        
        perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
        
        return {
            'perplexity': perplexity,
            'loss': total_loss / total_tokens if total_tokens > 0 else float('inf'),
            'total_tokens': total_tokens,
            'num_sequences': num_sequences
        }

# =====================================
# Main Experiment Runner
# =====================================

class MultiGPUExperimentRunner:
    """Run experiments with multi-GPU support"""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self.model_loader = MultiGPUModelLoader(config)
        self.pruner = ParallelNEFFPruner(config)
        self.evaluator = MultiGPUPerplexityEvaluator(config)
    
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
                texts.append(item["text"][:2048])  # Limit length
        
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
            prefetch_factor=self.config.prefetch_factor
        )
    
    def run_single_model(self, model_name: str) -> Dict:
        """Run experiments on a single model"""
        
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
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
            results['model_size_gb'] = total_params * 2 / (1024**3)  # FP16
            
            print(f"Model parameters: {total_params:,}")
            print(f"Model size: {results['model_size_gb']:.2f} GB")
            
            # Evaluate original model
            print("\n1. Evaluating Original Model")
            print("-" * 40)
            orig_results = self.evaluator.evaluate(model, tokenizer)
            results['original'] = orig_results
            print(f"Perplexity: {orig_results['perplexity']:.2f}")
            
            # Prepare calibration data
            calib_loader = self.prepare_calibration_data(tokenizer, self.config.calibration_samples)
            
            # Test different pruning methods
            for method in self.config.methods:
                print(f"\n2. Testing {method.upper()} Pruning")
                print("-" * 40)
                
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
                
                print(f"Sparsity: {pruning_info['overall_sparsity']:.2%}")
                
                # Evaluate pruned model
                pruned_results = self.evaluator.evaluate(pruned_model, tokenizer)
                
                # Store results
                results[method] = {
                    'pruning_info': pruning_info,
                    'evaluation': pruned_results,
                    'perplexity_increase': pruned_results['perplexity'] - orig_results['perplexity'],
                    'relative_increase': (pruned_results['perplexity'] / orig_results['perplexity'] - 1) * 100
                }
                
                print(f"Perplexity: {pruned_results['perplexity']:.2f} "
                      f"(Δ: +{results[method]['perplexity_increase']:.2f}, "
                      f"{results[method]['relative_increase']:.1f}% increase)")
                
                # Clean up
                del pruned_model
                torch.cuda.empty_cache()
            
            # Clean up original model
            del model
            torch.cuda.empty_cache()
            
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
        
        filename = f"{self.config.results_dir}/neff_results_{safe_name}_{timestamp}.json"
        
        # Convert non-serializable objects
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return str(obj)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
        
        print(f"\nResults saved to {filename}")

# =====================================
# CLI Interface
# =====================================

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU NEFF Pruning for LLMs")
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to evaluate"
    )
    
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="GPU IDs to use (e.g., 0 1 2 3)"
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
        default=["weight_only", "emp"],
        help="Pruning methods to test"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/data/wyx",
        help="Base directory for data storage"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = MultiGPUConfig(
        data_dir=args.data_dir,
        cache_dir=f"{args.data_dir}/cache",
        results_dir=f"{args.data_dir}/results",
        available_gpus=args.gpus,
        models=args.models,
        calibration_samples=args.calibration_samples,
        eval_samples=args.eval_samples,
        methods=args.methods
    )
    
    # Print system info
    print("\n" + "="*60)
    print("MULTI-GPU NEFF PRUNING SYSTEM")
    print("="*60)
    print(f"Available GPUs: {config.available_gpus}")
    print(f"Data directory: {config.data_dir}")
    print(f"Models to test: {len(config.models) if config.models else 'default set'}")
    print(f"Methods: {config.methods}")
    print("="*60)
    
    # Run experiments
    runner = MultiGPUExperimentRunner(config)
    
    if args.models:
        for model in args.models:
            runner.run_single_model(model)
    else:
        # Run default model set
        for model in config.models:
            try:
                runner.run_single_model(model)
            except Exception as e:
                print(f"Failed on {model}: {e}")
                continue
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print(f"Results saved in: {config.results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
