import torch
import torch.nn as nn


# class LogitAdapter(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int = 0,
#         vocab_size: int = 0,
#         lm_head: nn.Module = None,
#         dropout: float = 0.0,
#         init_scale: float = 1e-5,
#     ):
#         super().__init__()

#         if lm_head is not None:
#             hidden_dim = lm_head.in_features
#             vocab_size = lm_head.out_features


#         self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2, bias=False)
#         self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2, bias=False)        
#         self.dropout = nn.Dropout(dropout)
#         self.act = nn.SiLU()
#         self.act2 = nn.SiLU()
#         self.ln = nn.LayerNorm(hidden_dim // 2) 
#         self.hidden_dim = hidden_dim // 2
#         self.vocab_size = vocab_size

#         # nn.init.eye_(self.fc1.weight)
#         # nn.init.zeros_(self.fc1.bias)
#         # nn.init.eye_(self.fc2.weight)
#         # nn.init.zeros_(self.fc2.bias)
#         # nn.init.normal_(self.fc1.weight, mean=0.0, std=init_scale)
#         # nn.init.normal_(self.fc1.bias, mean=0.0, std=init_scale)
        
#         self.lm_head = nn.Linear(hidden_dim // 2, vocab_size, bias=False)
#         nn.init.normal_(self.lm_head.weight, mean=0.0, std=init_scale)
#         # if lm_head is not None:
#         #     self.lm_head.weight.data.copy_(lm_head.weight.data)

#     def forward(self, x):
#         h = self.fc1(x)
#         h = self.act(h)
#         h = self.act2(self.fc2(h))
#         # h = self.ln(h)
#         out = self.lm_head(h)
#         # out = self.lm_head(x)
#         return out
    
#     def save(self, path: str):
#         ckpt = {
#             "state_dict": self.state_dict(),
#             "hidden_dim": self.hidden_dim,
#             "vocab_size": self.vocab_size,
#         }
#         torch.save(ckpt, path)

#     @classmethod
#     def load(cls, path: str, map_location="cpu"):
#         ckpt = torch.load(path, map_location=map_location)
#         model = cls(
#             hidden_dim=ckpt["hidden_dim"] * 2,
#             vocab_size=ckpt["vocab_size"],
#         )
#         model.load_state_dict(ckpt["state_dict"])
#         return model


class LogitAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        vocab_size: int = 0,
        lm_head: nn.Module = None,
        dropout: float = 0.0,
        init_scale: float = 1e-5,
    ):
        super().__init__()

        if lm_head is not None:
            hidden_dim = lm_head.in_features
            vocab_size = lm_head.out_features


        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2, bias=False)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2, bias=False)        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        self.act2 = nn.SiLU()
        self.ln = nn.LayerNorm(hidden_dim // 2) 
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # nn.init.eye_(self.fc1.weight)
        # nn.init.zeros_(self.fc1.bias)
        # nn.init.eye_(self.fc2.weight)
        # nn.init.zeros_(self.fc2.bias)
        # nn.init.normal_(self.fc1.weight, mean=0.0, std=init_scale)
        # nn.init.normal_(self.fc1.bias, mean=0.0, std=init_scale)
        
        self.lm_head = nn.Linear(hidden_dim // 2, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=init_scale)
        # if lm_head is not None:
        #     self.lm_head.weight.data.copy_(lm_head.weight.data)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        h = self.act2(self.fc2(h))
        # h = self.ln(h)
        out = self.lm_head(h)
        # out = self.lm_head(x)
        return out
    
    def save(self, path: str):
        ckpt = {
            "state_dict": self.state_dict(),
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
        }
        torch.save(ckpt, path)

    @classmethod
    def load(cls, path: str, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        model = cls(
            hidden_dim=ckpt["hidden_dim"],
            vocab_size=ckpt["vocab_size"],
        )
        model.load_state_dict(ckpt["state_dict"])
        return model


import torch
from transformers import LogitsProcessorList
from transformers.generation import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

def sample_logits(logits, temperature=1.0, top_k=50, top_p=0.95, do_sample=True):
    if logits.dim() == 3:
        logits = logits.squeeze(1)

    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    else:
        batch_size = logits.shape[0]

        warpers = LogitsProcessorList()
        
        if temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1)) 
        if top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

        dummy_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=logits.device)

        processed_logits = warpers(dummy_input_ids, logits)

        probs = torch.nn.functional.softmax(processed_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) # Shape trả về: [Batch_Size, 1]
        
        return next_token

def new_generate(logits_list, original_ids, eos_token_id, temp=1.0, k=50, p=0.95):
    mask = (original_ids == eos_token_id)
    batch_tokens = []
    
    # 1. Loop qua từng step để sample
    for step_idx, current_step_logits in enumerate(logits_list):
        token_id = sample_logits(
            current_step_logits, 
            temperature=temp, 
            top_k=k, 
            top_p=p
        )
        batch_tokens.append(token_id)

    output_sequences = torch.cat(batch_tokens, dim=1)
    
    # 3. Kiểm tra kích thước Mask
    if mask is not None:
        mask = mask[:, -output_sequences.size(1):]
        output_sequences = output_sequences.masked_fill(mask, eos_token_id)
        
    return output_sequences

def token_modify(new_token_ids, original_ids, eos_token_id):
    mask = (original_ids == eos_token_id)
    output_sequences = torch.cat(new_token_ids, dim=1)
    mask = mask[:, -output_sequences.size(1):]
    output_sequences = output_sequences.masked_fill(mask, eos_token_id)
    return output_sequences

def sample_logits_all_positions(logits, mask, temperature=1.0, top_k=50, top_p=0.95, do_sample=True, pad_value=-100):
    batch_size, N, vocab_size = logits.shape
    
    bool_mask = mask.bool()
    
    valid_logits = logits[bool_mask] 
    
    M = valid_logits.shape[0]
    
    final_tokens = torch.full((batch_size, N), fill_value=pad_value, dtype=torch.long, device=logits.device)
    
    if M == 0:
        return final_tokens

    if not do_sample:
        sampled_tokens = torch.argmax(valid_logits, dim=-1) # Shape: (M,)
    else:
        warpers = LogitsProcessorList()
        if temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1)) 
        if top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

        dummy_input_ids = torch.zeros((M, 1), dtype=torch.long, device=valid_logits.device)
        processed_logits = warpers(dummy_input_ids, valid_logits)

        probs = torch.nn.functional.softmax(processed_logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # Shape: (M,)

    final_tokens[bool_mask] = sampled_tokens
    
    return final_tokens