import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

def main():
    # 检查 CUDA 可用性
    print("=== GPU 环境检查 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"可用 GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 50)
    
    # 设置模型路径
    model_path = "/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"
    
    print("\n开始加载模型...")
    start_time = time.time()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载模型到GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # 自动分配到可用的GPU
        trust_remote_code=True,
        torch_dtype=torch.float16  # 使用float16以节省显存
    )
    
    load_time = time.time() - start_time
    print(f"模型加载完成，耗时: {load_time:.2f}秒")
    
    # 打印模型信息
    print("\n模型信息:")
    print(f"模型参数量: {model.num_parameters() / 1e9:.2f}B")
    print(f"模型所在设备: {next(model.parameters()).device}")
    
    # 进行简单的推理测试
    test_text = "请介绍一下北京的历史"
    print(f"\n测试输入文本: {test_text}")
    
    # 记录推理时间
    inference_start = time.time()
    
    # 对输入进行编码
    inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    inference_time = time.time() - inference_start
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n生成的回复: {response}")
    print(f"推理耗时: {inference_time:.2f}秒")

if __name__ == "__main__":
    main()
