import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

# ==========================================
# 1. 物理参数配置 (精确对齐 Qwen3-30B-A3B Config)
# ==========================================
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768 
SEQ_LEN = 5  # 模拟 Target 验证时，一次性传入的 Draft Token 数量
DTYPE = torch.float16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自动推导体积
EXPERT_PARAMS = 3 * HIDDEN_SIZE * INTERMEDIATE_SIZE
EXPERT_MB = (EXPERT_PARAMS * 2) / (1024 * 1024) # FP16 占 2 Bytes
ACT_KB = (SEQ_LEN * HIDDEN_SIZE * 2) / 1024

# 想要测试的 n 个专家数量列表
TEST_N_LIST = [1, 2, 3, 4, 5, 6, 7, 8]

# ==========================================
# 2. 模拟专家网络
# ==========================================
class DummyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ==========================================
# 3. 测速工具函数
# ==========================================
def benchmark(func, name, iters=50, warmup=10):
    for _ in range(warmup):
        func()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(iters):
        func()
        
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()
    
    avg_time_ms = ((end_time - start_time) / iters) * 1000
    print(f"| {name:<35} | {avg_time_ms:>10.3f} ms |")
    return avg_time_ms

# ==========================================
# 4. 核心测试逻辑
# ==========================================
def main():
    print("="*60)
    print("🚀 细粒度 MoE 异构推理物理常数验证 (多专家扩展版)")
    print("="*60)
    print(f"模型特征 : Qwen3 细粒度 MoE (128 Experts)")
    print(f"单专家体积: ~{EXPERT_MB:.2f} MB (FP16)")
    print(f"激活值体积: ~{ACT_KB:.2f} KB (Seq={SEQ_LEN})")
    print("-" * 60)

    # 初始化测试数据 (使用锁页内存加速 PCIe)
    cpu_expert = DummyExpert().to(torch.float32) 
    gpu_expert = DummyExpert().to(DTYPE).to(DEVICE)
    
    cpu_input = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, dtype=torch.float32).pin_memory()
    gpu_input = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    
    weight_tensor_cpu = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=DTYPE).pin_memory()
    
    # 限制 CPU 线程数，模拟单实例争抢的情况
    torch.set_num_threads(os.cpu_count() // 2) 

    for n in TEST_N_LIST:
        print(f"\n{'='*20} 测试 n = {n} 个专家 {'='*20}")
        print(f"{'测试项目':<37} | {'平均耗时':>13}")
        print("-" * 60)

        # ------------------------------------------------
        # 测试 1：GPU 计算 n 个专家 (T_gpu)
        # ------------------------------------------------
        def task_gpu_compute(num_experts):
            for _ in range(num_experts):
                _ = gpu_expert(gpu_input)

        benchmark(lambda: task_gpu_compute(n), f"1. GPU Compute {n} Experts (T_gpu)")

        # ------------------------------------------------
        # 测试 2：CPU 计算 n 个专家 (T_cpu)
        # ------------------------------------------------
        def task_cpu_compute(num_experts):
            with torch.no_grad():
                for _ in range(num_experts):
                    _ = cpu_expert(cpu_input)

        benchmark(lambda: task_cpu_compute(n), f"2. CPU Compute {n} Experts (T_cpu)")

        # ------------------------------------------------
        # 测试 3：PCIe 搬运 n 个专家权重 (T_swap)
        # ------------------------------------------------
        def task_weight_swap(num_experts):
            for _ in range(num_experts):
                # 复制 3 次代表 Gate, Up, Down 三个矩阵的 PCIe 传输
                gpu_t1 = weight_tensor_cpu.to(DEVICE, non_blocking=True)
                gpu_t2 = weight_tensor_cpu.to(DEVICE, non_blocking=True)
                gpu_t3 = weight_tensor_cpu.to(DEVICE, non_blocking=True)
            torch.cuda.synchronize()

        benchmark(lambda: task_weight_swap(n), f"3. PCIe Swap Weights (~{EXPERT_MB * n:.1f}MB)")

        # ------------------------------------------------
        # 测试 4：PCIe 搬运激活值 (不随 n 增加，因为一次传完)
        # ------------------------------------------------
        def task_act_swap():
            gpu_act = cpu_input.to(DEVICE, non_blocking=True)
            _ = gpu_act.to('cpu', non_blocking=True)
            torch.cuda.synchronize()

        benchmark(task_act_swap, f"4. PCIe Swap Activations (~{ACT_KB:.1f}KB)")

    print("\n" + "="*60)
    print("💡 数据观测指南：")
    print("1. 观察 T_gpu 是否随 n 线性增长？(通常 GPU 没吃满时，增长很缓慢)")
    print("2. 观察 T_cpu 是否随 n 严格线性增长？(CPU 内存带宽容易打满)")
    print("3. 关键对比：比较 '2. CPU Compute' 和 '3. PCIe Swap Weights' 的耗时！")
    print("   -> 哪边时间短，Target 验证时就应该采用哪种调度策略。")

if __name__ == "__main__":
    main()