# Expert Skip Prediction Experiment

## 实验目的

验证跳过最后一个 expert 的修改模型能否用来预测原始模型的专家激活。

## 实验流程

1. **加载模型**: 加载 Qwen3-30B-A3B-Base 模型
2. **原始模型推理**: 
   - 执行 prefill 阶段
   - 执行一步 decode
   - 记录每一层激活的 experts 集合
3. **修改模型**: 修改模型使每一层运行时跳过最后一个 expert（top-8 -> top-7）
4. **修改模型推理**:
   - 使用 prefill 阶段的 kv cache 和输出
   - 执行修改模型的一步 decode
   - 记录每一层激活的 experts 集合
5. **统计匹配程度**: 计算原始模型和修改模型激活 experts 的匹配率

## 运行方法

```bash
cd /zx_data1/sparsity/on_device_sd/pre_exp/expert_skip_prediction
python experiment.py
```

## 输出结果

实验结果会保存在 `expert_match_results.json` 文件中，包含：
- 每一层的原始激活 experts
- 每一层的修改激活 experts
- 每一层的匹配率和 Jaccard 相似度
- 总体统计信息

## 注意事项

- 需要确保模型路径正确：`/zx_data1/models/Qwen--Qwen3-30B-A3B-Base`
- 需要足够的 GPU 显存来加载 30B 模型
- 实验可能需要较长时间运行


