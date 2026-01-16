import torch

def _speculative_sampling(
    candidate_input_ids: torch.Tensor,
    candidate_logits: torch.Tensor, 
    candidate_length: int,
    new_logits: torch.Tensor,
    last_assistant_token_is_eos: bool = False,
    max_matches: int = None
) -> tuple[torch.Tensor, int]:
    """
    推测采样算法（基于Spec-Bench实现）
    输入:
        candidate_input_ids: 候选输入序列
        candidate_logits: 候选序列的logits
        candidate_length: 候选序列长度
        new_logits: 原模型验证的logits
        last_assistant_token_is_eos: 最后一个候选token是否为EOS
        max_matches: 最大匹配数
    输出:
        valid_tokens: 有效的token序列
        n_matches: 匹配的token数量
    """
    if max_matches is None:
        max_matches = candidate_length
    
    # 确保候选长度不超过输入序列的长度
    actual_candidate_length = min(candidate_length, candidate_input_ids.shape[1])
    new_candidate_input_ids = candidate_input_ids[:, -actual_candidate_length:]
    
    # 更新candidate_length为实际长度
    candidate_length = actual_candidate_length
    
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)
    p = new_logits.softmax(dim=-1)
    
    # 处理Qwen3模型的特殊情况
    # 对于Qwen3，可能 p.shape[1] == 1，这意味着我们只有一个位置的logits
    # 在这种情况下，我们需要特殊处理
    
    # 如果没有足多logits，我们使用直接比较来决定是否接受
    if p.shape[1] == 1 and q.shape[1] == 1:
        # 获取draft token的ID
        draft_token_id = new_candidate_input_ids[0, -1].item()
        
        # 获取两个模型对这个token的概率
        draft_prob = q[0, 0, draft_token_id].item()
        original_prob = p[0, 0, draft_token_id].item()
        
        # 计算概率比率
        ratio = original_prob / max(draft_prob, 1e-8)
        
        # 随机决定是否接受
        if torch.rand(1).item() <= ratio:
            # 接受
            return new_candidate_input_ids, 1
        else:
            # 拒绝，采样一个新token
            p_prime = p[0, 0]
            t = torch.multinomial(p_prime, num_samples=1)[None, :]
            return t, 0
    
    # 处理维度不匹配的情况
    if q.shape[1] == 0 or p.shape[1] == 0:
        # 如果没有有效的logits，返回空结果
        return new_candidate_input_ids[:, :0], 0
    
    # 确保索引不越界
    actual_candidate_length = min(candidate_length, q.shape[1], p.shape[1] - 1)
    if actual_candidate_length <= 0:
        return new_candidate_input_ids[:, :0], 0
    
    # 使用实际可用的长度
    q_i = q[:, torch.arange(actual_candidate_length), new_candidate_input_ids[:, -actual_candidate_length:]].squeeze(0, 1)
    p_i = p[:, torch.arange(actual_candidate_length), new_candidate_input_ids[:, -actual_candidate_length:]].squeeze(0, 1)
    
    # 如果没有有效的候选token，直接返回
    if candidate_length == 0 or q_i.shape[0] == 0 or p_i.shape[0] == 0:
        return new_candidate_input_ids[:, :0], 0
        
    # 避免除零
    q_i = torch.clamp(q_i, min=1e-8)
    probability_ratio = p_i / q_i

    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    # 处理空张量的情况
    if is_accepted.numel() == 0:
        n_matches = 0
    else:
        n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    if last_assistant_token_is_eos and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        n_matches = min(n_matches, max_matches)

        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = min(candidate_logits.shape[1], max_matches)
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t

    return valid_tokens, int(n_matches.item()) if isinstance(n_matches, torch.Tensor) else n_matches

def speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    last_assistant_token_is_eos,
    max_matches,
    deterministic: bool = False,
):
    """
    Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)
    p = new_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)

    if deterministic:
        # 确定性：逐位置与argmax比较，直到首个不匹配
        target_argmax = p[:, torch.arange(candidate_length), :].argmax(dim=-1).squeeze(0)
        candidate_flat = new_candidate_input_ids.squeeze(0)
        equal_mask = (target_argmax[:candidate_length] == candidate_flat[:candidate_length])
        # 连续前缀匹配长度
        if equal_mask.numel() == 0:
            n_matches = 0
        else:
            # 找到第一个False的位置
            first_false = (~equal_mask).float().argmax().item() if (~equal_mask).any() else candidate_length
            n_matches = first_false if first_false != 0 or not equal_mask[0].item() else 0
            if (~equal_mask).any() is False:
                n_matches = candidate_length
    else:
        probability_ratio = p_i / q_i
        r_i = torch.rand_like(probability_ratio)
        is_accepted = r_i <= probability_ratio
        n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    if last_assistant_token_is_eos and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        n_matches = min(n_matches, max_matches)

        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = min(candidate_logits.shape[1], max_matches)
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            denom = p_prime.sum(dim=-1, keepdim=True)
            p_prime = p_prime / torch.clamp(denom, min=1e-12)
        else:
            p_prime = p_n_plus_1
        if deterministic:
            t = p_prime.argmax(dim=-1, keepdim=True)[None, :].squeeze(0)
        else:
            t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t

    return valid_tokens, int(n_matches.item()) if isinstance(n_matches, torch.Tensor) else n_matches