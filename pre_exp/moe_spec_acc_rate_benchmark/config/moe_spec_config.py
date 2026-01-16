from dataclasses import dataclass
from typing import List

@dataclass
class MOESpecConfig:
    # 模型配置
    model_path: str = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    device: str = "cuda"
    dtype: str = "float16"
    
    # Expert路由修改配置
    top_k_experts_to_remove: int = 2
    
    # 推测解码配置
    draft_length: int = 1
    max_new_tokens: int = 1024
    temperature: float = 0.0
    do_sample: bool = False
    
    # 评估配置
    bench_name: str = "spec_bench"
    num_choices: int = 1
    num_gpus_per_model: int = 1
    num_gpus_total: int = 1
    
    # 输出配置
    model_id: str = "moe-spec-test"
    answer_file: str = None
