"""Generate answers with MOE speculative decoding.

Usage:
python3 inference_moe_spec.py --model-path /zx_data1/models/Qwen--Qwen3-30B-A3B-Base --model-id moe-spec-test
"""
import argparse
import torch
from fastchat.utils import str_to_torch_dtype
from evaluation.eval import run_eval, reorg_answer_file
from model.moe_spec.moe_model import MOEModelWrapper, ModifiedMOEModel
from model.moe_spec.moe_spec_decoder import MOESpecDecoder

def moe_spec_forward(inputs, model, tokenizer, max_new_tokens, 
                    num_to_modify=2, src_positions=None, dst_positions=None, **kwargs):
    """
    MOE推测解码的前向函数，兼容Spec-Bench接口
    注意：移除了draft_length参数，因为现在固定为1
    """
    input_ids = inputs.input_ids
    
    # 创建修改的MOE模型
    if src_positions is None:
        src_positions = [7, 8]
    if dst_positions is None:
        dst_positions = [9, 10]
    modified_model = ModifiedMOEModel(
        model,
        num_to_modify=num_to_modify,
        src_positions=src_positions,
        dst_positions=dst_positions,
    )
    
    # 创建推测解码器（现在draft_length固定为1）
    spec_decoder = MOESpecDecoder(model, modified_model)
    
    # 执行推测解码
    result = spec_decoder.speculate_decode(input_ids, max_new_tokens)
    
    return (result['output_ids'], 
            result['new_token'], 
            result['step'], 
            result['accept_length_list'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the MOE model"
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="spec_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    # 移除draft-length参数，因为现在固定为1
    parser.add_argument("--num-to-modify", type=int, default=2)
    parser.add_argument("--src-positions", type=str, default="7,8")
    parser.add_argument("--dst-positions", type=str, default="9,10")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    args = parser.parse_args()

    # 设置数据文件路径
    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"results/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    # 加载MOE模型
    model = MOEModelWrapper(
        args.model_path,
        device="cuda",
        dtype=args.dtype
    )

    tokenizer = model.get_tokenizer()

    # 执行评估
    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=moe_spec_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        num_to_modify=args.num_to_modify,
        src_positions=[int(x) for x in args.src_positions.split(',') if x],
        dst_positions=[int(x) for x in args.dst_positions.split(',') if x],
    )

    reorg_answer_file(answer_file)
