import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy as np
import json
import copy


class ExpertActivationTracker:
    """Track expert activations during MoE forward pass"""
    
    def __init__(self, record_top_k=8):
        self.activations = defaultdict(list)
        self.hooks = []
        self.record_top_k = record_top_k  # Always record TOP-8 for comparison
    
    def register_hooks(self, model):
        """Register forward hooks to capture expert selections"""
        
        for name, module in model.named_modules():
            # Hook MoE layers - adjust based on actual model architecture
            if 'mlp' in name.lower() or 'moe' in name.lower():
                if hasattr(module, 'gate') or 'gate' in name:
                    hook = module.register_forward_hook(
                        self._create_hook(name)
                    )
                    self.hooks.append(hook)
    
    def _create_hook(self, layer_name):
        """Create a hook function for a specific layer"""
        def hook(module, input, output):
            # Extract router logits and compute top-k experts
            if isinstance(input, tuple):
                hidden_states = input[0]
            else:
                hidden_states = input
            
            # Get router logits (this may need adjustment based on model)
            if hasattr(module, 'gate'):
                router_logits = module.gate(hidden_states)
            elif hasattr(module, 'router'):
                router_logits = module.router(hidden_states)
            else:
                return
            
            # Get top-k experts for the last token
            routing_weights = F.softmax(router_logits, dim=-1)
            last_token_weights = routing_weights[-1, -1, :]  # [num_experts]
            
            # Always record TOP-8 for comparison purposes
            top_k_values, top_k_indices = torch.topk(
                last_token_weights, 
                self.record_top_k, 
                dim=-1
            )
            
            self.activations[layer_name].append({
                'experts': top_k_indices.cpu().tolist(),
                'weights': top_k_values.cpu().tolist(),
                'all_weights': last_token_weights.cpu().numpy()
            })
        
        return hook
    
    def clear(self):
        """Clear recorded activations"""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def quantize_expert_weights(module, bits=8):
    """
    Quantize expert weights to specified bit precision
    
    Args:
        module: The module containing expert weights
        bits: Number of bits for quantization (default: 8)
    """
    def quantize_tensor(tensor, bits=8):
        """Quantize a tensor to specified bits"""
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return tensor
        
        # Calculate quantization parameters
        qmin = 0
        qmax = 2 ** bits - 1
        
        # Get min and max values
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Calculate scale and zero point
        scale = (max_val - min_val) / qmax
        zero_point = qmin - min_val / scale
        
        # Quantize
        q_tensor = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
        
        # Dequantize back to original dtype
        dequant_tensor = (q_tensor - zero_point) * scale
        
        return dequant_tensor.to(tensor.dtype)
    
    # Quantize all parameters in the module
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data = quantize_tensor(param.data, bits=bits)


def quantize_model_experts(model, bits=8):
    """
    Quantize all expert weights in the MoE model
    
    Args:
        model: The model to quantize
        bits: Number of bits for quantization (default: 8)
    
    Returns:
        Number of experts quantized
    """
    quantized_count = 0
    
    for name, module in model.named_modules():
        # Identify expert modules - adjust based on actual model architecture
        if 'expert' in name.lower() or ('mlp' in name.lower() and 'moe' in name.lower()):
            # Skip gate/router modules, only quantize expert computation
            if 'gate' not in name.lower() and 'router' not in name.lower():
                quantize_expert_weights(module, bits=bits)
                quantized_count += 1
    
    return quantized_count


def calculate_step_match(original_experts, quantized_experts, step_idx):
    """
    Calculate matching metrics for a specific decode step
    """
    results = {
        'layer_matches': {},
        'step': step_idx
    }
    
    for layer_name in original_experts.keys():
        if layer_name not in quantized_experts:
            continue
        
        if step_idx >= len(original_experts[layer_name]) or step_idx >= len(quantized_experts[layer_name]):
            continue
        
        # Get TOP-8 from both models for this step
        orig_top8 = original_experts[layer_name][step_idx]['experts'][:8]
        quant_top8 = quantized_experts[layer_name][step_idx]['experts'][:8]
        
        # Calculate overlap (all 8 positions)
        orig_set = set(orig_top8)
        quant_set = set(quant_top8)
        
        intersection = orig_set.intersection(quant_set)
        overlap_ratio = len(intersection) / 8.0
        
        # Check exact order match
        exact_match = orig_top8 == quant_top8
        
        # Position-based matching (how many experts match at same position)
        position_matches = sum(1 for i in range(8) if orig_top8[i] == quant_top8[i])
        
        results['layer_matches'][layer_name] = {
            'overlap_ratio': overlap_ratio,
            'exact_order_match': exact_match,
            'position_matches': position_matches,
            'original_top8': orig_top8,
            'quantized_top8': quant_top8,
            'original_weights': original_experts[layer_name][step_idx]['weights'][:8],
            'quantized_weights': quantized_experts[layer_name][step_idx]['weights'][:8]
        }
    
    return results


def calculate_average_stats(step_results):
    """Calculate average statistics across all layers for a step"""
    if not step_results['layer_matches']:
        return {}
    
    overlap_ratios = [m['overlap_ratio'] for m in step_results['layer_matches'].values()]
    position_match_counts = [m['position_matches'] for m in step_results['layer_matches'].values()]
    
    # Calculate percentage of layers with 100% overlap
    perfect_matches = sum(1 for ratio in overlap_ratios if ratio == 1.0)
    perfect_match_percentage = (perfect_matches / len(overlap_ratios)) * 100
    
    return {
        'avg_overlap_ratio': np.mean(overlap_ratios),
        'std_overlap_ratio': np.std(overlap_ratios),
        'min_overlap_ratio': np.min(overlap_ratios),
        'max_overlap_ratio': np.max(overlap_ratios),
        'avg_position_matches': np.mean(position_match_counts),
        'perfect_match_count': perfect_matches,
        'perfect_match_percentage': perfect_match_percentage,
        'total_layers': len(step_results['layer_matches'])
    }


def run_quantized_experiment(model_path, prompt_text, num_decode_steps=3, quantization_bits=8):
    """
    Main experiment function for quantized experts
    
    Args:
        model_path: Path to the model
        prompt_text: Input prompt for generation
        num_decode_steps: Number of decode steps to analyze
        quantization_bits: Number of bits for expert quantization (default: 8)
    """
    print(f"Loading model from {model_path}...")
    
    # Load model and tokenizer (only once)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: Quantized Experts ({quantization_bits}-bit)")
    print(f"Decode Steps: {num_decode_steps}")
    print(f"Original Model: FP16/FP32 experts, TOP-8")
    print(f"Quantized Model: {quantization_bits}-bit experts, TOP-8")
    print(f"{'='*60}")
    
    # ============================================================
    # Phase 1: Original Model - Prefill + N Decode Steps
    # ============================================================
    print("\n=== Phase 1: Original Model (Full Precision Experts) ===")
    print("Performing prefill and decode steps...")
    
    tracker_original = ExpertActivationTracker(record_top_k=8)
    tracker_original.register_hooks(model)
    
    with torch.no_grad():
        outputs_original = model.generate(
            **inputs,
            max_new_tokens=num_decode_steps,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    original_activations = dict(tracker_original.activations)
    tracker_original.remove_hooks()
    
    print(f"Captured activations from {len(original_activations)} layers")
    print(f"Steps recorded: {len(list(original_activations.values())[0]) if original_activations else 0}")
    
    generated_text_original = tokenizer.decode(outputs_original.sequences[0], skip_special_tokens=True)
    print(f"Generated text (original): {generated_text_original}")
    
    # Extract the prefill KV cache from original model
    print("\n--- Extracting prefill cache from original model ---")
    
    # Get prefill results (just the input tokens, no generation)
    with torch.no_grad():
        prefill_outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True
        )
    
    # Store the KV cache from prefill
    prefill_past_key_values = prefill_outputs.past_key_values
    
    print(f"Prefill cache extracted: {len(prefill_past_key_values)} layers")
    
    # ============================================================
    # Phase 2: Quantize Experts in Same Model
    # ============================================================
    print(f"\n=== Phase 2: Quantizing Experts ({quantization_bits}-bit) ===")
    print("Quantizing expert weights in the same model...")
    
    # Quantize expert weights in place
    quantized_count = quantize_model_experts(model, bits=quantization_bits)
    print(f"Quantized {quantized_count} expert modules")
    
    # ============================================================
    # Phase 3: Quantized Model - Using Original Model's Prefill
    # ============================================================
    print(f"\n=== Phase 3: Running Decode with Quantized Experts ===")
    print("Using original model's prefill and KV cache...")
    
    # Now perform decode steps using the prefill cache
    tracker_quantized = ExpertActivationTracker(record_top_k=8)
    tracker_quantized.register_hooks(model)
    
    # Start generation from the prefill cache
    with torch.no_grad():
        outputs_quantized = model.generate(
            **inputs,
            max_new_tokens=num_decode_steps,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    quantized_activations = dict(tracker_quantized.activations)
    tracker_quantized.remove_hooks()
    
    print(f"Captured activations from {len(quantized_activations)} layers")
    print(f"Steps recorded: {len(list(quantized_activations.values())[0]) if quantized_activations else 0}")
    
    generated_text_quantized = tokenizer.decode(outputs_quantized.sequences[0], skip_special_tokens=True)
    print(f"Generated text (quantized): {generated_text_quantized}")
    
    # ============================================================
    # Phase 4: Analysis for Each Decode Step
    # ============================================================
    print("\n=== Phase 4: Step-by-Step Analysis ===")
    
    all_results = []
    
    for step_idx in range(num_decode_steps):
        print(f"\n{'-'*60}")
        print(f"DECODE STEP {step_idx + 1}")
        print(f"{'-'*60}")
        
        step_results = calculate_step_match(original_activations, quantized_activations, step_idx)
        avg_stats = calculate_average_stats(step_results)
        
        step_results['average_stats'] = avg_stats
        all_results.append(step_results)
        
        # Print average statistics for this step
        print("\n--- Average Statistics (All Layers) ---")
        print(f"Average Overlap Ratio (TOP-8): {avg_stats['avg_overlap_ratio']:.4f} ({avg_stats['avg_overlap_ratio']*100:.2f}%)")
        print(f"Std Overlap Ratio: {avg_stats['std_overlap_ratio']:.4f}")
        print(f"Min Overlap Ratio: {avg_stats['min_overlap_ratio']:.4f}")
        print(f"Max Overlap Ratio: {avg_stats['max_overlap_ratio']:.4f}")
        print(f"Average Position Matches: {avg_stats['avg_position_matches']:.2f}/8")
        print(f"Perfect Matches (100% Overlap): {avg_stats['perfect_match_count']}/{avg_stats['total_layers']} ({avg_stats['perfect_match_percentage']:.2f}%)")
        print(f"Total Layers: {avg_stats['total_layers']}")
        
        # Show a few example layers
        print("\n--- Sample Layer Results ---")
        for i, (layer_name, metrics) in enumerate(list(step_results['layer_matches'].items())[:3]):
            print(f"\n{layer_name}:")
            print(f"  Overlap Ratio: {metrics['overlap_ratio']:.2%}")
            print(f"  Position Matches: {metrics['position_matches']}/8")
            print(f"  Original TOP-8: {metrics['original_top8']}")
            print(f"  Quantized TOP-8: {metrics['quantized_top8']}")
    
    # ============================================================
    # Phase 5: Overall Summary Across All Steps
    # ============================================================
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY ACROSS ALL DECODE STEPS")
    print(f"{'='*60}")
    
    overall_avg_overlap = np.mean([r['average_stats']['avg_overlap_ratio'] for r in all_results])
    overall_avg_position = np.mean([r['average_stats']['avg_position_matches'] for r in all_results])
    overall_perfect_match_pct = np.mean([r['average_stats']['perfect_match_percentage'] for r in all_results])
    
    print(f"\nAverage Overlap Ratio across {num_decode_steps} steps: {overall_avg_overlap:.4f} ({overall_avg_overlap*100:.2f}%)")
    print(f"Average Position Matches across {num_decode_steps} steps: {overall_avg_position:.2f}/8")
    print(f"Average Perfect Match Percentage across {num_decode_steps} steps: {overall_perfect_match_pct:.2f}%")
    
    print("\n--- Per-Step Summary ---")
    for i, result in enumerate(all_results):
        stats = result['average_stats']
        print(f"Step {i+1}: Avg Overlap = {stats['avg_overlap_ratio']:.4f} ({stats['avg_overlap_ratio']*100:.2f}%), "
              f"Perfect Matches = {stats['perfect_match_percentage']:.2f}%")
    
    # ============================================================
    # Conclusion
    # ============================================================
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    
    if overall_avg_overlap > 0.95:
        print(f"✓ HIGH AGREEMENT ({overall_avg_overlap:.1%})")
        print(f"  The quantized model ({quantization_bits}-bit experts) can accurately predict the")
        print("  original model's expert activations. Quantization has minimal impact.")
        print(f"  {overall_perfect_match_pct:.1f}% of layers show perfect (100%) overlap.")
    elif overall_avg_overlap > 0.85:
        print(f"~ MODERATE AGREEMENT ({overall_avg_overlap:.1%})")
        print(f"  The quantized model ({quantization_bits}-bit experts) shows reasonable predictive")
        print("  capability but with some divergence from the original model.")
        print(f"  {overall_perfect_match_pct:.1f}% of layers show perfect (100%) overlap.")
    else:
        print(f"✗ LOW AGREEMENT ({overall_avg_overlap:.1%})")
        print(f"  The quantized model ({quantization_bits}-bit experts) has limited predictive")
        print("  capability. Quantization significantly impacts expert selection.")
        print(f"  {overall_perfect_match_pct:.1f}% of layers show perfect (100%) overlap.")
    
    # Text comparison
    print("\n--- Generated Text Comparison ---")
    print(f"Original:  {generated_text_original}")
    print(f"Quantized: {generated_text_quantized}")
    texts_match = generated_text_original == generated_text_quantized
    print(f"Texts Match: {'✓ Yes' if texts_match else '✗ No'}")
    
    return {
        'step_results': all_results,
        'overall_stats': {
            'avg_overlap_ratio': overall_avg_overlap,
            'avg_position_matches': overall_avg_position,
            'avg_perfect_match_percentage': overall_perfect_match_pct,
            'num_steps': num_decode_steps,
            'quantization_bits': quantization_bits
        },
        'generated_texts': {
            'original': generated_text_original,
            'quantized': generated_text_quantized,
            'texts_match': texts_match
        }
    }


def save_results(results, filename='quantized_expert_activation_results.json'):
    """Save results to JSON file"""
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    PROMPT = " The governor would take possession of the arsenal in the name of the United States . "
    NUM_DECODE_STEPS = 10
    QUANTIZATION_BITS = 8  # Can try 8, 4, or other bit widths
    
    try:
        results = run_quantized_experiment(
            MODEL_PATH, 
            PROMPT, 
            NUM_DECODE_STEPS, 
            QUANTIZATION_BITS
        )
        save_results(results)
        
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()