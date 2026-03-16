import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. 配置区域
# ==========================================
MODEL_PATH = "/data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base"
PREDICTOR_WEIGHTS = "/data2/group_谈海生/lagin/models/SRDP_Experiments/run_soft_20260128_231323/best_model.pth"
DATA_FILE = "/data2/group_谈海生/lagin/data/mtbench101/mtbench101.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STOP_THRESHOLD = 0.65
MAX_DRAFT_LEN = 10
MAX_SAMPLES = 50  # 为了测试速度，只取前 50 个样本

# ==========================================
# 2. SRDP 预测器
# ==========================================
class SRDP_Predictor(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. CPU-GPU 异构专家调度包装器 (n=2 算法)
# ==========================================
class HeteroQwenExpertBlockWrapper(nn.Module):
    def __init__(self, original_mlp, layer_idx, max_cpu_experts=2):
        super().__init__()
        self.layer_idx = layer_idx
        self.gate = original_mlp.gate
        self.experts = original_mlp.experts
        self.shared_expert = getattr(original_mlp, "shared_expert", None)
        self.shared_expert_gate = getattr(original_mlp, "shared_expert_gate", None)
        
        self.num_experts = len(self.experts)
        self.max_cpu_experts = max_cpu_experts
        
        # 物理分割：一半 GPU，一半 CPU
        self.gpu_expert_indices = set(range(0, self.num_experts // 2))
        self.cpu_expert_indices = set(range(self.num_experts // 2, self.num_experts))
        
        for i in self.cpu_expert_indices:
            self.experts[i].to('cpu')
            
        self.last_routing_info = None

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        router_logits = self.gate(hidden_states_flat)
        all_probs = F.softmax(router_logits, dim=-1)
        
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        hidden_states_cpu = hidden_states_flat.cpu()
        
        orig_ids_batch, orig_weights_batch = [], []
        mod_ids_batch, mod_weights_batch = [], []

        for t in range(hidden_states_flat.shape[0]):
            logits_t = router_logits[t]
            
            # 1. 初始 Top-8
            sorted_logits, sorted_indices = torch.sort(logits_t, descending=True)
            top8_indices = sorted_indices[:8].tolist()
            
            orig_ids_batch.append(top8_indices)
            orig_weights_batch.append(F.softmax(logits_t[top8_indices], dim=-1).tolist())
            
            cpu_needed = [idx for idx in top8_indices if idx in self.cpu_expert_indices]
            gpu_needed = [idx for idx in top8_indices if idx in self.gpu_expert_indices]
            
            # 2. 截断与替补 (n=2)
            if len(cpu_needed) > self.max_cpu_experts:
                cpu_kept = cpu_needed[:self.max_cpu_experts]
                gpu_kept = gpu_needed.copy()
                
                shortage = 8 - len(cpu_kept) - len(gpu_kept)
                for rank in range(8, self.num_experts):
                    candidate_idx = sorted_indices[rank].item()
                    if candidate_idx in self.gpu_expert_indices:
                        gpu_kept.append(candidate_idx)
                        shortage -= 1
                        if shortage == 0: break
                final_indices = cpu_kept + gpu_kept
            else:
                final_indices = top8_indices
                
            # 3. 归一化
            final_indices_tensor = torch.tensor(final_indices, device=DEVICE)
            selected_logits = logits_t[final_indices_tensor]
            final_weights = F.softmax(selected_logits, dim=-1)
            
            mod_ids_batch.append(final_indices)
            mod_weights_batch.append(final_weights.tolist())
            
            # 4. 异构计算
            for idx, weight in zip(final_indices, final_weights):
                if idx in self.gpu_expert_indices:
                    expert_out = self.experts[idx](hidden_states_flat[t].unsqueeze(0))
                    final_hidden_states[t] += (expert_out.squeeze(0) * weight)
                    
            for idx, weight in zip(final_indices, final_weights):
                if idx in self.cpu_expert_indices:
                    expert_out_cpu = self.experts[idx](hidden_states_cpu[t].unsqueeze(0))
                    final_hidden_states[t] += (expert_out_cpu.squeeze(0).to(DEVICE) * weight)
                    
        self.last_routing_info = {
            "orig_ids": np.array(orig_ids_batch), "orig_weights": np.array(orig_weights_batch),
            "mod_ids": np.array(mod_ids_batch), "mod_weights": np.array(mod_weights_batch)
        }
        
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states_flat)
            if self.shared_expert_gate is not None:
                shared_out = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_out
            final_hidden_states += shared_out
            
        return final_hidden_states.view(batch_size, seq_len, hidden_dim)

def apply_hetero_moe(model):
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp = HeteroQwenExpertBlockWrapper(layer.mlp, layer_idx=i, max_cpu_experts=2)
    return model

# ==========================================
# 4. 实时特征提取器
# ==========================================
class RealtimeFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.reset_state()

    def reset_state(self):
        self.state = {
            "min_top1_prob": 1.0, "avg_entropy": 0.0,
            "step_count": 0, "accum_score_loss": 0.0
        }
        self.prev_mlp_pred = 1.0

    def extract(self, logits, hidden_states):
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top10_probs, _ = torch.topk(probs, 10)
        top10_probs = top10_probs / top10_probs.sum()
        entropy = -(top10_probs * torch.log(top10_probs + 1e-9)).sum().item()
        
        top2_vals, _ = torch.topk(probs, 2)
        top1_prob, margin = top2_vals[0].item(), (top2_vals[0] - top2_vals[1]).item()
        
        total_curr_score_loss = 0.0
        max_layer_loss = 0.0
        total_replaced, total_slots = 0, 0
        
        for layer in self.model.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "last_routing_info"):
                info = layer.mlp.last_routing_info
                if info is None: continue
                
                orig_ids, mod_ids, orig_weights = info["orig_ids"][0], info["mod_ids"][0], info["orig_weights"][0]
                mask = (orig_ids != mod_ids)
                total_replaced += np.sum(mask)
                total_slots += len(orig_ids)
                
                layer_loss = np.sum(orig_weights * mask)
                total_curr_score_loss += layer_loss
                max_layer_loss = max(max_layer_loss, layer_loss)
                
        replace_rate = total_replaced / (total_slots + 1e-9)
        
        s = self.state
        s["step_count"] += 1
        s["min_top1_prob"] = min(s["min_top1_prob"], top1_prob)
        s["avg_entropy"] = (s["avg_entropy"] * (s["step_count"] - 1) + entropy) / s["step_count"]
        s["accum_score_loss"] += total_curr_score_loss
        hidden_norm = torch.norm(hidden_states[0, -1, :], p=2).item()
        
        features = [
            total_curr_score_loss, replace_rate, max_layer_loss, s["accum_score_loss"] / 10.0, 0.0,
            top1_prob, entropy, margin,
            s["step_count"] / 10.0, s["min_top1_prob"], s["avg_entropy"], hidden_norm / 100.0, self.prev_mlp_pred
        ]
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ==========================================
# 5. 加载 MTBench 数据
# ==========================================
def prepare_mtbench_data(tokenizer):
    print(f"正在加载 MTBench 数据: {DATA_FILE}")
    samples = []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                history = item.get("history", [])
                
                text = ""
                for turn in history:
                    if "user" in turn: text += f"User: {turn['user']}\n"
                    if "bot" in turn: text += f"Bot: {turn['bot']}\n"
                
                if not text: continue
                encodings = tokenizer(text, return_tensors="pt")
                samples.append(encodings.input_ids)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return []
    print(f"加载了 {len(samples)} 个样本。")
    return samples

# ==========================================
# 6. 批量评测引擎
# ==========================================
def run_mtbench_eval():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    
    print("⏳ 准备草稿模型...")
    draft_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    draft_model = apply_hetero_moe(draft_model)
    
    predictor = SRDP_Predictor(input_dim=13).to(DEVICE)
    if os.path.exists(PREDICTOR_WEIGHTS):
        predictor.load_state_dict(torch.load(PREDICTOR_WEIGHTS, map_location=DEVICE))
    predictor.eval()

    samples = prepare_mtbench_data(tokenizer)[:MAX_SAMPLES]
    extractor = RealtimeFeatureExtractor(draft_model)
    
    TARGET_GEN_LEN = 30 # 每个 prompt 测试生成 30 个 token
    
    total_time_A = 0.0
    total_time_B = 0.0
    total_tokens = 0
    total_accepted = 0

    print("\n" + "="*50)
    print(f"🚀 开始 MTBench 批量评估 (共 {len(samples)} 个 Prompt)")
    print("="*50)

    for idx, input_ids in enumerate(tqdm(samples, desc="Evaluating")):
        input_ids = input_ids.to(DEVICE)
        
        # ------------------- 测试 A：纯自回归 -------------------
        curr_ids_A = input_ids.clone()
        torch.cuda.synchronize()
        start_time_A = time.time()
        
        with torch.no_grad():
            for _ in range(TARGET_GEN_LEN):
                outputs = target_model(curr_ids_A)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
                curr_ids_A = torch.cat([curr_ids_A, next_token], dim=1)
                
        torch.cuda.synchronize()
        total_time_A += (time.time() - start_time_A)
        
        # ------------------- 测试 B：推测解码 -------------------
        curr_ids_B = input_ids.clone()
        torch.cuda.synchronize()
        start_time_B = time.time()
        
        generated_tokens = 0
        with torch.no_grad():
            while generated_tokens < TARGET_GEN_LEN:
                draft_tokens = []
                draft_inputs = curr_ids_B.clone()
                extractor.reset_state()
                
                for _ in range(MAX_DRAFT_LEN):
                    outputs = draft_model(draft_inputs, output_hidden_states=True)
                    logits = outputs.logits
                    hidden_states = outputs.hidden_states[-1]
                    
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                    
                    features = extractor.extract(logits, hidden_states)
                    pred_prob = predictor(features).item()
                    extractor.prev_mlp_pred = pred_prob
                    
                    if pred_prob < STOP_THRESHOLD: break
                        
                    draft_tokens.append(next_token)
                    draft_inputs = torch.cat([draft_inputs, next_token], dim=1)
                
                # Target Verification
                if len(draft_tokens) > 0:
                    draft_tensor = torch.cat(draft_tokens, dim=1)
                    verify_inputs = torch.cat([curr_ids_B, draft_tensor], dim=1)
                else:
                    verify_inputs = curr_ids_B
                    
                verify_outputs = target_model(verify_inputs)
                target_logits = verify_outputs.logits[:, curr_ids_B.shape[1]-1:, :] 
                
                accepted_len = 0
                for i in range(len(draft_tokens)):
                    target_token = torch.argmax(target_logits[:, i, :], dim=-1).unsqueeze(0)
                    if target_token.item() == draft_tokens[i].item():
                        accepted_len += 1
                    else: break
                        
                total_accepted += accepted_len
                final_token_to_add = torch.argmax(target_logits[:, accepted_len, :], dim=-1).unsqueeze(0)
                
                if accepted_len > 0:
                    accepted_tensor = torch.cat(draft_tokens[:accepted_len], dim=1)
                    curr_ids_B = torch.cat([curr_ids_B, accepted_tensor, final_token_to_add], dim=1)
                    generated_tokens += (accepted_len + 1)
                else:
                    curr_ids_B = torch.cat([curr_ids_B, final_token_to_add], dim=1)
                    generated_tokens += 1
                    
        torch.cuda.synchronize()
        total_time_B += (time.time() - start_time_B)
        total_tokens += generated_tokens

    # =================输出报告=================
    print("\n" + "🏆"*20)
    print("🎉 MTBench 最终实验加速报告")
    print("🏆"*20)
    print(f"测试 Prompt 数量  : {len(samples)}")
    print(f"总生成 Token 数量 : {total_tokens}")
    print(f"总计草稿接受数量  : {total_accepted} (接受率: {total_accepted/total_tokens:.2%})")
    print("-" * 40)
    print(f"纯 Target 耗时    : {total_time_A:.2f} s ({total_tokens/total_time_A:.2f} tok/s)")
    print(f"推测解码总耗时    : {total_time_B:.2f} s ({total_tokens/total_time_B:.2f} tok/s)")
    speedup_ratio = total_time_A / total_time_B
    print(f"整体端到端加速比  : {speedup_ratio:.2f} x")

if __name__ == "__main__":
    run_mtbench_eval()