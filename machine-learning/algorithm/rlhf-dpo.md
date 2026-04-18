# RLHF & DPO — Preference Alignment

Preference alignment methods fine-tune a pretrained language model to produce outputs that humans prefer. They sit after supervised fine-tuning (SFT) in the training pipeline and are responsible for making models helpful, harmless, and honest.

```
Pretraining → Supervised Fine-Tuning (SFT) → Preference Alignment (RLHF or DPO)
```

---

## Background — Why Alignment Is Needed

A language model trained only on next-token prediction learns to imitate the training corpus — including its noise, biases, and harmful content. Supervised fine-tuning on curated instruction-response pairs improves this, but the model still does not have a notion of which of its possible outputs a human would prefer.

Preference alignment addresses this by training the model on human preference data: pairs of responses to the same prompt where a human has indicated which response is better.

---

## RLHF — Reinforcement Learning from Human Feedback

RLHF is the original preference alignment approach, used in InstructGPT and early ChatGPT. It involves four stages:

### Stage 1 — Pretraining

A large language model is pretrained on a broad text corpus via next-token prediction. This gives the model general language understanding and generation capability.

### Stage 2 — Supervised Fine-Tuning (SFT)

The pretrained model is fine-tuned on a curated dataset of (prompt, high-quality response) pairs. This teaches the model the desired response format and domain.

```python
# SFT: standard cross-entropy on (prompt, response) pairs
loss = cross_entropy(model(prompt + response), response_tokens)
```

### Stage 3 — Reward Modelling

A separate reward model is trained to predict human preference scores. Human annotators are shown pairs of responses `(y_w, y_l)` to the same prompt and indicate which they prefer (`y_w` = preferred/winner, `y_l` = rejected/loser).

The reward model is trained with a Bradley-Terry preference model:

```
Loss = -E[ log σ(r(x, y_w) - r(x, y_l)) ]
```

where `r(x, y)` is the scalar reward for response `y` to prompt `x`, and `σ` is the sigmoid function.

```python
# Reward model training
reward_winner = reward_model(prompt, response_winner)
reward_loser  = reward_model(prompt, response_loser)
loss = -torch.log(torch.sigmoid(reward_winner - reward_loser)).mean()
```

### Stage 4 — RL Fine-Tuning with PPO

The SFT model is further fine-tuned using Proximal Policy Optimisation (PPO). The reward model provides a scalar signal for each generated response. A KL divergence penalty against the SFT model prevents the policy from drifting too far.

```
Objective = E[ r(x, y) ] - β · KL[ π_θ(y|x) || π_SFT(y|x) ]
```

- `π_θ` — the policy being trained
- `π_SFT` — the frozen SFT reference model
- `β` — KL penalty coefficient (typically 0.1–0.5)
- `r(x, y)` — reward from the reward model

| Pros | Cons |
|------|------|
| Flexible — reward model can encode complex preferences | Complex — requires training and maintaining a separate reward model |
| Can optimise non-differentiable objectives | PPO is unstable — sensitive to hyperparameters |
| Strong empirical results (InstructGPT, ChatGPT) | High compute cost — four models in memory simultaneously |
| Fine-grained control via KL coefficient | Reward hacking — model learns to exploit reward model weaknesses |

---

## DPO — Direct Preference Optimisation

DPO eliminates the reward model and RL entirely. It derives a closed-form training objective directly from the preference data, showing that the optimal policy under the RLHF objective can be expressed as a simple classification loss on the language model itself.

### Key insight

The RLHF objective implicitly defines a relationship between the optimal policy and the reward:

```
r*(x, y) = β · log[ π*(y|x) / π_SFT(y|x) ] + β · log Z(x)
```

Substituting this into the Bradley-Terry preference model and cancelling the partition function `Z(x)` (which is the same for both responses to the same prompt) gives the DPO loss directly:

```
L_DPO = -E[ log σ( β · log[π_θ(y_w|x)/π_SFT(y_w|x)] - β · log[π_θ(y_l|x)/π_SFT(y_l|x)] ) ]
```

This is a binary cross-entropy loss that:
- Increases the likelihood of the preferred response `y_w` relative to the reference model
- Decreases the likelihood of the rejected response `y_l` relative to the reference model
- Uses the log-ratio to the reference model as an implicit reward signal

### Implementation

```python
import torch
import torch.nn.functional as F


def dpo_loss(
    policy_logprobs_winner: torch.Tensor,
    policy_logprobs_loser: torch.Tensor,
    ref_logprobs_winner: torch.Tensor,
    ref_logprobs_loser: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute the DPO loss for a batch of preference pairs.

    Parameters
    ----------
    policy_logprobs_winner : torch.Tensor
        Sum of log-probabilities of the preferred response under the policy, shape (B,).
    policy_logprobs_loser : torch.Tensor
        Sum of log-probabilities of the rejected response under the policy, shape (B,).
    ref_logprobs_winner : torch.Tensor
        Sum of log-probabilities of the preferred response under the reference model, shape (B,).
    ref_logprobs_loser : torch.Tensor
        Sum of log-probabilities of the rejected response under the reference model, shape (B,).
    beta : float
        KL penalty coefficient. Higher beta = stay closer to reference model.

    Returns
    -------
    torch.Tensor
        Scalar DPO loss.

    Notes
    -----
    The log-ratio log[π_θ/π_ref] acts as an implicit reward.
    The loss pushes the policy to assign higher implicit reward to y_w than y_l.
    """
    log_ratio_winner = policy_logprobs_winner - ref_logprobs_winner
    log_ratio_loser  = policy_logprobs_loser  - ref_logprobs_loser

    logits = beta * (log_ratio_winner - log_ratio_loser)
    loss   = -F.logsigmoid(logits).mean()
    return loss


def get_sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_idx: int,
) -> torch.Tensor:
    """
    Compute sum of log-probabilities for the response tokens only.

    Parameters
    ----------
    input_ids : torch.Tensor
        Token IDs for [prompt + response], shape (B, L).
    response_start_idx : int
        Index where the response begins in the sequence.

    Returns
    -------
    torch.Tensor
        Sum of log-probs over response tokens, shape (B,).
    """
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)

    # Sum only over response tokens
    response_log_probs = token_log_probs[:, response_start_idx:]
    return response_log_probs.sum(dim=-1)
```

### Training loop

```python
from torch.utils.data import DataLoader

def train_dpo(
    policy_model,
    reference_model,
    dataloader: DataLoader,
    optimiser,
    beta: float = 0.1,
    n_epochs: int = 1,
) -> None:
    """
    DPO training loop.

    Notes
    -----
    reference_model is frozen — gradients are disabled.
    policy_model is updated to maximise preference alignment.
    """
    reference_model.eval()

    for epoch in range(n_epochs):
        for batch in dataloader:
            # batch contains: prompt, winner_ids, loser_ids, attention masks

            # Policy log-probs (with gradient)
            policy_logprobs_w = get_sequence_logprobs(
                policy_model, batch["winner_ids"], batch["winner_mask"],
                batch["response_start_idx"],
            )
            policy_logprobs_l = get_sequence_logprobs(
                policy_model, batch["loser_ids"], batch["loser_mask"],
                batch["response_start_idx"],
            )

            # Reference log-probs (no gradient — frozen)
            with torch.no_grad():
                ref_logprobs_w = get_sequence_logprobs(
                    reference_model, batch["winner_ids"], batch["winner_mask"],
                    batch["response_start_idx"],
                )
                ref_logprobs_l = get_sequence_logprobs(
                    reference_model, batch["loser_ids"], batch["loser_mask"],
                    batch["response_start_idx"],
                )

            loss = dpo_loss(
                policy_logprobs_w, policy_logprobs_l,
                ref_logprobs_w,    ref_logprobs_l,
                beta=beta,
            )

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
```

### Data format

DPO requires a preference dataset of triplets:

```python
# Each example is a dict with three fields
{
    "prompt":    "Explain gradient descent in simple terms.",
    "winner":    "Gradient descent is an optimisation algorithm...",   # preferred response
    "loser":     "It's a math thing used in ML.",                      # rejected response
}
```

---

## RLHF vs DPO Comparison

| | RLHF (PPO) | DPO |
|---|-----------|-----|
| Reward model | Required — separate model trained on preferences | Not needed — implicit in the loss |
| Training stability | Unstable — PPO is sensitive to hyperparameters | Stable — standard supervised loss |
| Compute | High — 4 models in memory (policy, ref, reward, value) | Low — 2 models (policy, ref) |
| Implementation complexity | High | Low |
| Reward hacking | Yes — model can exploit reward model | Less — no separate reward model to exploit |
| Fine-grained control | High — reward shaping, KL tuning | Medium — only β controls KL |
| Empirical performance | Strong — used in GPT-4, Claude | Competitive — often matches RLHF on benchmarks |
| Best for | Complex reward signals, multi-objective alignment | Simpler alignment tasks, limited compute |

---

## GRPO — Group Relative Policy Optimisation

GRPO is the alignment algorithm used in DeepSeek-R1. It eliminates the value model required by PPO, reducing memory cost while maintaining strong performance on reasoning tasks.

**Core idea:** instead of using a learned value function to estimate the baseline, GRPO generates a group of G responses for each prompt and uses the group’s mean reward as the baseline.

```
For each prompt x:
  Generate G responses {y_1, ..., y_G} from the current policy
  Compute reward r_i for each response
  Baseline = mean(r_1, ..., r_G)
  Advantage_i = r_i - baseline

Objective = E[ (advantage_i / std(advantages)) · clipped_ratio ] - β · KL
```

The clipped ratio is the same PPO clipping trick — `clip(π_θ/π_old, 1-ε, 1+ε)` — preventing large policy updates.

| | PPO | GRPO |
|---|-----|------|
| Baseline estimation | Learned value model (separate network) | Group mean reward (no extra model) |
| Models in memory | 4 (policy, ref, reward, value) | 3 (policy, ref, reward) |
| Best for | General RLHF | Reasoning tasks with verifiable rewards |

**Verifiable rewards:** GRPO works especially well when rewards can be computed automatically — e.g. whether a maths answer is correct, whether code passes unit tests. This removes the need for a learned reward model entirely for these tasks.

---

## SimPO — Simple Preference Optimisation

SimPO is a recent simplification of DPO that removes the reference model entirely. Instead of using log-ratios to the reference model as implicit rewards, SimPO uses the average log-probability of the response directly.

```
L_SimPO = -E[ log σ( (β/|y_w|) · logπ(y_w|x) - (β/|y_l|) · logπ(y_l|x) - γ ) ]
```

- `|y|` — response length (normalises for length bias)
- `γ` — target reward margin (ensures a minimum gap between winner and loser)
- No reference model needed — reduces memory by one model

| | DPO | SimPO |
|---|-----|-------|
| Reference model | Required | Not needed |
| Length normalisation | No | Yes — divides by response length |
| Reward margin | No | Yes — γ enforces minimum gap |
| Memory | 2 models | 1 model |

---

## β (Beta) — The KL Coefficient

β controls the trade-off between preference alignment and staying close to the reference model:

| β value | Effect |
|---------|--------|
| Low (0.01–0.05) | Strong alignment — model moves far from reference, risk of degeneration |
| Medium (0.1–0.3) | Balanced — standard range for most tasks |
| High (0.5–1.0) | Conservative — model stays close to SFT, weaker alignment |

Start with β = 0.1 and tune based on whether the model degenerates (lower β) or alignment is too weak (lower β).

---

## Rules

- Always start with a well-trained SFT model — DPO and RLHF cannot fix a poor base model
- Freeze the reference model completely — it must not update during DPO training
- Use length-normalised log-probs if winner and loser responses have very different lengths — raw sum favours shorter responses
- Monitor the implicit reward margin `β · (log_ratio_winner - log_ratio_loser)` during training — it should increase
- Validate alignment qualitatively — run the model on held-out prompts and inspect outputs, not just loss curves
