import argparse
import json
import os
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM


DEFAULT_MODEL_PATH = "/data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base"
DEFAULT_OUTPUT_DIR = "/data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base-experts-int4-g128"
EXPERT_LINEAR_NAMES = ("gate_proj", "up_proj", "down_proj")


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    if q.dtype != torch.int8:
        q = q.to(torch.int8)
    unsigned = (q + 8).to(torch.uint8)
    if unsigned.shape[-1] % 2:
        pad = torch.zeros(*unsigned.shape[:-1], 1, dtype=torch.uint8)
        unsigned = torch.cat([unsigned, pad], dim=-1)
    low = unsigned[..., 0::2]
    high = unsigned[..., 1::2] << 4
    return low | high


def quantize_weight_symmetric_int4(weight: torch.Tensor, group_size: int) -> dict:
    weight = weight.detach().float().cpu().contiguous()
    out_features, in_features = weight.shape
    pad = (group_size - (in_features % group_size)) % group_size
    if pad:
        weight = torch.cat([weight, torch.zeros(out_features, pad)], dim=1)

    grouped = weight.view(out_features, -1, group_size)
    scales = grouped.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
    q = torch.round(grouped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    q = q.view(out_features, -1)

    return {
        "qweight": pack_int4(q),
        "scales": scales.to(torch.float16),
        "out_features": out_features,
        "in_features": in_features,
        "group_size": group_size,
        "bits": 4,
        "symmetric": True,
    }


def iter_moe_layers(model):
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "gate"):
            yield layer_idx, mlp


def quantize_model(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    layer_dir = os.path.join(args.output_dir, "layers")
    os.makedirs(layer_dir, exist_ok=True)

    metadata = {
        "source_model": args.model_path,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quantization": {
            "method": "symmetric_groupwise_int4",
            "paper_reference": "MoE-SpeQ draft ratio: INT4 expert linears, group_size=128",
            "bits": 4,
            "group_size": args.group_size,
            "quantized_modules": "all Linear layers inside routed expert modules",
            "fp16_modules": [
                "router/gate",
                "attention",
                "normalization",
                "embeddings",
                "lm_head",
                "shared_experts",
                "non_expert_parameters",
            ],
        },
        "layers": [],
    }

    print(f"Loading source model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    moe_layers = list(iter_moe_layers(model))
    if not moe_layers:
        raise RuntimeError("No MoE layers found. Check model architecture.")

    print(f"Found {len(moe_layers)} MoE layers. Writing quantized expert shards to: {layer_dir}")
    for layer_idx, mlp in tqdm(moe_layers, desc="Quantizing MoE layers"):
        layer_payload = {}
        expert_count = len(mlp.experts)

        for expert_idx, expert in enumerate(tqdm(mlp.experts, desc=f"layer {layer_idx}", leave=False)):
            for linear_name in EXPERT_LINEAR_NAMES:
                linear = getattr(expert, linear_name)
                key = f"experts.{expert_idx}.{linear_name}"
                layer_payload[key] = quantize_weight_symmetric_int4(linear.weight, args.group_size)

        shard_name = f"layer_{layer_idx:03d}.pt"
        shard_path = os.path.join(layer_dir, shard_name)
        torch.save(layer_payload, shard_path)
        metadata["layers"].append(
            {
                "layer_idx": layer_idx,
                "num_experts": expert_count,
                "linear_names": list(EXPERT_LINEAR_NAMES),
                "shard": os.path.join("layers", shard_name),
            }
        )
        del layer_payload

    metadata_path = os.path.join(args.output_dir, "quantization_config.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Quantized expert draft model generated.")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Metadata: {os.path.abspath(metadata_path)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize Qwen3 MoE routed expert linears to INT4 groupwise weights.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive.")
    quantize_model(args)


if __name__ == "__main__":
    main()
