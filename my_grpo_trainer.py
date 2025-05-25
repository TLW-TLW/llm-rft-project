from trl.trainer.grpo_trainer import GRPOTrainer
import torch
import torch.nn.functional as F


def safe_get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, no_grad=False):
    if no_grad:
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits
    else:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits
    log_probs = F.log_softmax(logits, dim=-1)
    idx = input_ids.unsqueeze(-1)
    logps_all = log_probs.gather(dim=-1, index=idx).squeeze(-1)
    logps = logps_all[:, -logits_to_keep:]
    return logps


class MyGRPOTrainer(GRPOTrainer):
    def compute_loss(self, model, inputs, **kwargs):
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # 取当前策略 log-probs
        per_token_logps = safe_get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep, no_grad=False
        )

        # 取参考策略 log-probs
        ref_per_token_logps = inputs.get("ref_per_token_logps", None)
        if ref_per_token_logps is None:
            if self.ref_model is not None:
                ref_per_token_logps = safe_get_per_token_logps(
                    self.ref_model, input_ids, attention_mask, logits_to_keep, no_grad=True
                )
            elif hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    ref_per_token_logps = safe_get_per_token_logps(
                        model, input_ids, attention_mask, logits_to_keep
                    )
            else:
                raise RuntimeError("Cannot obtain reference log-probs.")

        # KL（二阶近似）
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps) - 1
        )

        # 位置权重
        seq_len = completion_mask.size(1)
        position_weights = torch.linspace(
            1.0, 0.1, steps=seq_len, device=completion_mask.device
        ).unsqueeze(0).expand_as(per_token_kl)

        # policy + KL
        advantages = inputs["advantages"]
        adv_mean = advantages.mean()
        adv_std  = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        advantages = advantages.unsqueeze(1)
        # policy_term = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        # per_token_loss = -(policy_term - self.beta * position_weights * per_token_kl)
        policy_loss = -(per_token_logps * advantages)
        per_token_loss = policy_loss + self.beta * position_weights * per_token_kl

        loss = (per_token_loss * completion_mask).sum(1) / completion_mask.sum(1)
        loss = loss.mean()

        # logging
        mean_kl = (per_token_kl * completion_mask).sum(1) / completion_mask.sum(1)
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss


