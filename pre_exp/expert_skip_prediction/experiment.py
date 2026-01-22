import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy as np
import json


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


def modify_model_for_top_k(model, k=7):
    """
    Modify model to use top-k experts instead of top-8
    This creates a modified forward pass
    """
    modified_count = 0
    for name, module in model.named_modules():
        if 'mlp' in name.lower() or 'moe' in name.lower():
            if hasattr(module, 'num_experts_per_tok'):
                module.num_experts_per_tok = k
                modified_count += 1
            if hasattr(module, 'top_k'):
                module.top_k = k
                modified_count += 1
    return modified_count


def calculate_step_match(original_experts, modified_experts, step_idx):
    """
    Calculate matching metrics for a specific decode step
    """
    results = {
        'layer_matches': {},
        'step': step_idx
    }
    
    for layer_name in original_experts.keys():
        if layer_name not in modified_experts:
            continue
        
        if step_idx >= len(original_experts[layer_name]) or step_idx >= len(modified_experts[layer_name]):
            continue
        
        # Get TOP-8 from both models for this step
        orig_top8 = original_experts[layer_name][step_idx]['experts'][:8]
        mod_top8 = modified_experts[layer_name][step_idx]['experts'][:8]
        
        # Calculate overlap (all 8 positions)
        orig_set = set(orig_top8)
        mod_set = set(mod_top8)
        
        intersection = orig_set.intersection(mod_set)
        overlap_ratio = len(intersection) / 8.0
        
        # Check exact order match
        exact_match = orig_top8 == mod_top8
        
        # Position-based matching (how many experts match at same position)
        position_matches = sum(1 for i in range(8) if orig_top8[i] == mod_top8[i])
        
        results['layer_matches'][layer_name] = {
            'overlap_ratio': overlap_ratio,
            'exact_order_match': exact_match,
            'position_matches': position_matches,
            'original_top8': orig_top8,
            'modified_top8': mod_top8,
            'original_weights': original_experts[layer_name][step_idx]['weights'][:8],
            'modified_weights': modified_experts[layer_name][step_idx]['weights'][:8]
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


def run_experiment(model_path, prompt_text, num_decode_steps=3, modified_top_k=7):
    """
    Main experiment function
    
    Args:
        model_path: Path to the model
        prompt_text: Input prompt for generation
        num_decode_steps: Number of decode steps to analyze
        modified_top_k: Number of experts to use in modified model (default: 7)
    """
    print(f"Loading model from {model_path}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {num_decode_steps} Decode Steps")
    print(f"Original Model: TOP-8 experts")
    print(f"Modified Model: TOP-{modified_top_k} experts (recording TOP-8)")
    print(f"{'='*60}")
    
    # ============================================================
    # Phase 1: Original Model (Top-8 Experts) - Prefill + 3 Decode Steps
    # ============================================================
    print("\n=== Phase 1: Original Model (Top-8 Experts) ===")
    
    tracker_original = ExpertActivationTracker(record_top_k=8)
    tracker_original.register_hooks(model)
    
    with torch.no_grad():
        # Prefill and decode multiple steps
        outputs_original = model.generate(
            **inputs,
            max_new_tokens=num_decode_steps,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            do_sample=False,  # Deterministic for comparison
            pad_token_id=tokenizer.eos_token_id
        )
    
    original_activations = dict(tracker_original.activations)
    tracker_original.remove_hooks()
    
    print(f"Captured activations from {len(original_activations)} layers")
    print(f"Steps recorded: {len(list(original_activations.values())[0]) if original_activations else 0}")
    
    # Decode the generated tokens
    generated_text_original = tokenizer.decode(outputs_original.sequences[0], skip_special_tokens=True)
    print(f"Generated text (original): {generated_text_original}")
    
    # ============================================================
    # Phase 2: Modified Model (Top-k Experts) - Same Prefill + 3 Decode Steps
    # ============================================================
    print(f"\n=== Phase 2: Modified Model (Top-{modified_top_k} Experts, Recording Top-8) ===")
    
    # Modify model to use top-k
    modified_count = modify_model_for_top_k(model, k=modified_top_k)
    print(f"Modified {modified_count} parameters to use top-{modified_top_k} experts")
    
    # Track modified model activations (still record TOP-8 for comparison)
    tracker_modified = ExpertActivationTracker(record_top_k=8)
    tracker_modified.register_hooks(model)
    
    with torch.no_grad():
        # Same prefill and decode with modified model
        outputs_modified = model.generate(
            **inputs,
            max_new_tokens=num_decode_steps,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    modified_activations = dict(tracker_modified.activations)
    tracker_modified.remove_hooks()
    
    print(f"Captured activations from {len(modified_activations)} layers")
    print(f"Steps recorded: {len(list(modified_activations.values())[0]) if modified_activations else 0}")
    
    # Decode the generated tokens
    generated_text_modified = tokenizer.decode(outputs_modified.sequences[0], skip_special_tokens=True)
    print(f"Generated text (modified): {generated_text_modified}")
    
    # ============================================================
    # Phase 3: Analysis for Each Decode Step
    # ============================================================
    print("\n=== Phase 3: Step-by-Step Analysis ===")
    
    all_results = []
    
    for step_idx in range(num_decode_steps):
        print(f"\n{'-'*60}")
        print(f"DECODE STEP {step_idx + 1}")
        print(f"{'-'*60}")
        
        step_results = calculate_step_match(original_activations, modified_activations, step_idx)
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
            print(f"  Modified TOP-8: {metrics['modified_top8']}")
    
    # ============================================================
    # Phase 4: Overall Summary Across All Steps
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
        print(f"  The modified model (top-{modified_top_k}) can accurately predict the original model's")
        print("  expert activations (top-8). Skipping expert calculations is viable.")
        print(f"  {overall_perfect_match_pct:.1f}% of layers show perfect (100%) overlap.")
    elif overall_avg_overlap > 0.85:
        print(f"~ MODERATE AGREEMENT ({overall_avg_overlap:.1%})")
        print("  The modified model shows reasonable predictive capability but with")
        print("  some divergence. Further analysis recommended.")
        print(f"  {overall_perfect_match_pct:.1f}% of layers show perfect (100%) overlap.")
    else:
        print(f"✗ LOW AGREEMENT ({overall_avg_overlap:.1%})")
        print("  The modified model has limited predictive capability for the original")
        print("  model's expert activations. Skipping calculations may not be reliable.")
        print(f"  {overall_perfect_match_pct:.1f}% of layers show perfect (100%) overlap.")
    
    return {
        'step_results': all_results,
        'overall_stats': {
            'avg_overlap_ratio': overall_avg_overlap,
            'avg_position_matches': overall_avg_position,
            'avg_perfect_match_percentage': overall_perfect_match_pct,
            'num_steps': num_decode_steps,
            'modified_top_k': modified_top_k
        },
        'generated_texts': {
            'original': generated_text_original,
            'modified': generated_text_modified
        }
    }


def save_results(results, filename='expert_activation_results.json'):
    """Save results to JSON file"""
    # Convert numpy types to Python types for JSON serialization
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
    # PROMPT = "The capital of France is"
    PROMPT = "In September 2010 , a teaser website was revealed by Sega , hinting at a new Valkyria Chronicles game . In its September issue , Famitsu listed that Senjō no Valkyria 3 would be arriving on the PlayStation Portable . Its first public appearance was at the 2010 Tokyo Game Show ( TGS ) , where a demo was made available for journalists and attendees . During the publicity , story details were kept scant so as not to spoil too much for potential players , along with some of its content still being in flux at the time of its reveal . To promote the game and detail the story leading into the game 's events , an episodic Flash visual novel written by Fujii began release in January 2011 . The game was released January 27 , 2011 . During an interview , the development team said that the game had the capacity for downloadable content ( DLC ) , but that no plans were finalized . Multiple DLC maps , featuring additional missions and recruitable characters , were released between February and April 2011 ."
    # PROMPT = " The governor would take possession of the arsenal in the name of the United States . "
    # PROMPT = "What factors would you consider when designing an inclusive and accessible public transportation system?"
    NUM_DECODE_STEPS = 10
    MODIFIED_TOP_K = 6  # Number of experts for modified model (original uses 8)
    
    try:
        results = run_experiment(MODEL_PATH, PROMPT, NUM_DECODE_STEPS, MODIFIED_TOP_K)
        save_results(results)
        
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()