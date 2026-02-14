"""
Precision alignment tests for ModelRunner-based Qwen3 implementation.
"""

import os
import pytest
import torch

from transformers import AutoModelForCausalLM

from src.core.model import MoEConfig
from src.memory.parameter_loader import ParameterLoader
from src.model.qwen3_runner import Qwen3ModelRunner

QWEN3_MODEL_PATH = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"


@pytest.mark.skipif(
    not os.path.exists(QWEN3_MODEL_PATH),
    reason="Qwen3 model not found",
)
class TestModelRunnerPrecision:
    """Precision checks against transformers using ModelRunner path."""

    @pytest.mark.slow
    def test_embedding_vs_transformers(self):
        config = MoEConfig.from_pretrained(QWEN3_MODEL_PATH)
        param_loader = ParameterLoader(QWEN3_MODEL_PATH)
        param_loader._load_static_parameters()

        runner = Qwen3ModelRunner(config=config, parameter_loader=param_loader)

        hf_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        hf_model.eval()

        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(
            0, min(1000, config.vocab_size),
            (batch_size, seq_len),
            device="cuda",
        )

        with torch.no_grad():
            our_embed = runner.embed(input_ids)
            hf_embed = hf_model.model.embed_tokens(input_ids)

        max_diff = (our_embed - hf_embed).abs().max().item()
        assert torch.allclose(our_embed, hf_embed, atol=1e-6), f"Embedding mismatch: {max_diff:.2e}"

    @pytest.mark.slow
    def test_logits_vs_transformers(self):
        config = MoEConfig.from_pretrained(QWEN3_MODEL_PATH)
        param_loader = ParameterLoader(QWEN3_MODEL_PATH)
        param_loader._load_static_parameters()

        runner = Qwen3ModelRunner(config=config, parameter_loader=param_loader)

        hf_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        hf_model.eval()

        batch_size = 2
        seq_len = 6
        hidden = torch.randn(
            batch_size, seq_len, config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            our_logits = runner.compute_logits(hidden)
            hf_logits = hf_model.lm_head(hf_model.model.norm(hidden))

        max_diff = (our_logits - hf_logits).abs().max().item()
        assert torch.allclose(our_logits, hf_logits, atol=2e-3), f"Logits mismatch: {max_diff:.2e}"
        print(f"Input IDs: {input_ids[0].tolist()}")
        
        # Note: This will fail without loading experts
        # We'll test the structure but expect it to fail
        print("\nAttempting generation (will fail without experts loaded)...")
        
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=5,
                    temperature=1.0,
                )
            
            print(f"\n✓ Generation completed")
            print(f"  Output IDs: {output_ids[0].tolist()}")
            
            # Decode
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"  Output text: {output_text}")
            
        except Exception as e:
            print(f"\n⚠ Generation failed (expected): {e}")
            print("  This is expected without loading all experts")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
