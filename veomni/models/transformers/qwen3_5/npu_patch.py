

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers.activations import ACT2FN


from .generated import patched_modeling_qwen3_5_npu as modeling_qwen3_5
from ..qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_npu as modeling_qwen3_5_moe



def qwen3_next_rms_norm_forward_npu(self, x):
    return torch_npu.npu_rms_norm(x.float(), 1.0 + self.weight.float(), epsilon=self.eps)[0].type_as(x)

def qwen3_next_rms_norm_forward_gated_npu(self, hidden_states, gate=None):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight.float(), epsilon=self.variance_epsilon)[0]
    hidden_states = hidden_states * F.silu(gate.to(torch.float32))
    return hidden_states.to(input_dtype)

def qwen3_next_apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin).to(q.dtype)
    k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin).to(k.dtype)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def apply_qwen3_5_patch():
    # Patches for Qwen3.5 Model
    modeling_qwen3_5.Qwen3_5RMSNormGated.forward = qwen3_next_rms_norm_forward_gated_npu
    modeling_qwen3_5.Qwen3_5RMSNorm.forward = qwen3_next_rms_norm_forward_npu
    modeling_qwen3_5.apply_rotary_pos_emb = qwen3_next_apply_rotary_pos_emb_npu

def apply_qwen3_5_moe_patch():
    # Patches for Qwen3.5 MoE Model
    # modeling_qwen3_5_moe.Qwen3_5MoeExperts.forward = qwen3_5_moe_experts_forward_npu
    modeling_qwen3_5_moe.Qwen3_5MoeRMSNormGated.forward = qwen3_next_rms_norm_forward_gated_npu
    modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm.forward = qwen3_next_rms_norm_forward_npu
    modeling_qwen3_5_moe.apply_rotary_pos_emb = qwen3_next_apply_rotary_pos_emb_npu