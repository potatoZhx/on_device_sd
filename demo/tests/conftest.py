import os
import sys
from dataclasses import dataclass
from typing import Dict

import pytest
import torch

DEMO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DEMO_ROOT not in sys.path:
    sys.path.insert(0, DEMO_ROOT)

from src.core.model import MoEConfig
from src.core.types import ExpertID, DeviceType
from src.memory.paged_kv_cache import PagedKVCache
from src.model.qwen3_runner import Qwen3ModelRunner


class DummyParameterLoader:
    def __init__(self, config: MoEConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.static_params_gpu: Dict[str, torch.Tensor] = {}
        self.shared_expert_ids = set()
        self.shared_experts_gpu: Dict[ExpertID, Dict[str, torch.Tensor]] = {}
        self.expert_params_gpu: Dict[ExpertID, Dict[str, torch.Tensor]] = {}
        self.expert_params_cpu: Dict[ExpertID, Dict[str, torch.Tensor]] = {}
        self._init_params()

    def _rand(self, shape, device=None):
        device = device or self.device
        return torch.randn(*shape, device=device, dtype=self.config.get_dtype()) * 0.02

    def _init_params(self):
        cfg = self.config
        hidden = cfg.hidden_size
        q_size = cfg.num_attention_heads * cfg.head_dim
        kv_size = cfg.num_key_value_heads * cfg.head_dim
        moe_hidden = cfg.moe_intermediate_size

        self.static_params_gpu["embed_tokens"] = self._rand((cfg.vocab_size, hidden))
        self.static_params_gpu["lm_head"] = self._rand((cfg.vocab_size, hidden))
        self.static_params_gpu["final_layernorm"] = self._rand((hidden,))

        for layer_idx in range(cfg.num_hidden_layers):
            prefix = f"layer_{layer_idx}"
            self.static_params_gpu[f"{prefix}.input_layernorm"] = self._rand((hidden,))
            self.static_params_gpu[f"{prefix}.post_attention_layernorm"] = self._rand((hidden,))
            self.static_params_gpu[f"{prefix}.self_attn.q_proj"] = self._rand((q_size, hidden))
            self.static_params_gpu[f"{prefix}.self_attn.k_proj"] = self._rand((kv_size, hidden))
            self.static_params_gpu[f"{prefix}.self_attn.v_proj"] = self._rand((kv_size, hidden))
            self.static_params_gpu[f"{prefix}.self_attn.o_proj"] = self._rand((hidden, q_size))
            self.static_params_gpu[f"{prefix}.self_attn.q_norm"] = self._rand((cfg.head_dim,))
            self.static_params_gpu[f"{prefix}.self_attn.k_norm"] = self._rand((cfg.head_dim,))
            self.static_params_gpu[f"{prefix}.router"] = self._rand((cfg.num_experts, hidden))

            for expert_idx in range(cfg.num_experts):
                expert_id = ExpertID(layer_idx, expert_idx)
                params_gpu = {
                    "gate_proj": self._rand((moe_hidden, hidden)),
                    "up_proj": self._rand((moe_hidden, hidden)),
                    "down_proj": self._rand((hidden, moe_hidden)),
                }
                params_cpu = {k: v.cpu() for k, v in params_gpu.items()}
                self.expert_params_gpu[expert_id] = params_gpu
                self.expert_params_cpu[expert_id] = params_cpu

    def is_shared_expert(self, expert_id: ExpertID) -> bool:
        return expert_id in self.shared_expert_ids

    def get_shared_expert_params(self, expert_id: ExpertID):
        return self.shared_experts_gpu.get(expert_id)

    def get_expert_params(self, expert_id: ExpertID, device: DeviceType = DeviceType.GPU):
        if device == DeviceType.CPU:
            params = self.expert_params_cpu.get(expert_id)
            return params
        return self.expert_params_gpu.get(expert_id)


class DummyExpertCache:
    def __init__(self):
        self.cached_experts: Dict[ExpertID, Dict[str, torch.Tensor]] = {}
        self.max_experts = 16

    def is_cached(self, expert_id: ExpertID) -> bool:
        return expert_id in self.cached_experts

    def get(self, expert_id: ExpertID):
        return self.cached_experts.get(expert_id)

    def put(self, expert_id: ExpertID, params: Dict[str, torch.Tensor]):
        self.cached_experts[expert_id] = params


class TestPagedKVCache(PagedKVCache):
    def _calculate_num_blocks(self, gpu_memory_utilization: float) -> int:
        return 4


@pytest.fixture(scope="session")
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for model runner tests")
    return True


@pytest.fixture(scope="session")
def small_config():
    return MoEConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=128,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=0,
        moe_intermediate_size=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
        torch_dtype="float16",
        model_type="qwen3_moe",
        draft_top_c=1,
        max_draft_tokens=4,
    )


@pytest.fixture()
def dummy_parameter_loader(cuda_available, small_config):
    return DummyParameterLoader(small_config)


@pytest.fixture()
def dummy_expert_cache():
    return DummyExpertCache()


@pytest.fixture()
def qwen3_runner(cuda_available, small_config, dummy_parameter_loader):
    return Qwen3ModelRunner(config=small_config, parameter_loader=dummy_parameter_loader)


@pytest.fixture()
def kv_cache(cuda_available, small_config):
    return TestPagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )
