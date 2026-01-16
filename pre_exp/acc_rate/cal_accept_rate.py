import numpy as np
import logging

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler("/zx_data1/sparsity/on_device_sd/log/acc_rate/test.log")
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

# 创建流处理器（用于终端输出）
stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(stream_formatter)

# 将处理器添加到日志记录器
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

dir1 =  "/zx_data1/sparsity/on_device_sd/log/logits/Mixtral-8x7B_probs.npz"
dir2 =  "/zx_data1/sparsity/on_device_sd/log/logits/Mixtral-8x7B_probs_1.npz"

data1 = np.load(dir1, allow_pickle=True)
data2 = np.load(dir2, allow_pickle=True)

probs1 = data1["probs"]
probs2 = data2["probs"]
prefix_tokens1 = data1["prefix_tokens"]
prefix_tokens2 = data2["prefix_tokens"]

try:
    assert probs1.shape == probs2.shape, f"Probs shape mismatch: {probs1.shape} vs {probs2.shape}"
except AssertionError as e:
    logger.error(e)
    raise

acc_rate_sum = 0.0
num = 0
acc_rate = 0.0

for i in range(probs1.shape[0]):
    pre_tokens_i1 = prefix_tokens1[i]
    pre_tokens_i2 = prefix_tokens2[i]
    try:
        assert np.array_equal(pre_tokens_i1, pre_tokens_i2), f"Prefix tokens mismatch at index {i}: {pre_tokens_i1} vs {pre_tokens_i2}"
    except AssertionError as e:
        logger.error(e)
        raise

    probs_i1 = probs1[i]
    probs_i2 = probs2[i]
    acc_i = np.sum(np.minimum(probs_i1, probs_i2))
    logger.info(f"accept rate of {i}: {acc_i}; prefix tokens: {pre_tokens_i1}; ")
    acc_rate_sum += float(acc_i)
    num += 1
    acc_rate = acc_rate_sum / num
    logger.info(f"current accept rate: {acc_rate}")
    logger.info("--------------------------")

logger.info(f"finish compute of {num} samples")
logger.info(f"average accept rate: {acc_rate}")
