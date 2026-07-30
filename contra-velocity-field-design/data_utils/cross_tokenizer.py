import json
from collections.abc import Mapping
from typing import Dict, List, Sequence, Tuple

import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset

from .hf_data import resolve_data_file


N_SPAN = 1024


def _tokenize_with_padding_side(tokenizer, texts, padding_side: str, **kwargs):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        return tokenizer(texts, **kwargs)
    finally:
        tokenizer.padding_side = original_padding_side


def longest_common_subsequence(a, b, s_i=0, s_j=0) -> List[Tuple[int, int]]:
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    m, n = len(a), len(b)

    i = s_i
    j = s_j
    result = []

    while i < m and j < n:
        if a[i][1] == 0:
            i += 1
            continue
        if b[j][1] == 0:
            j += 1
            continue

        if a[i][1] == b[j][1]:
            result.append((i + 1, j + 1))
            i += 1
            j += 1
        elif a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

        if i + 1 < m and j + 1 < n:
            student_new_span = a[i][1] == 0 and a[i + 1][1] > 0
            teacher_new_span = b[j][1] == 0 and b[j + 1][1] > 0
            if student_new_span or teacher_new_span:
                while j < n and b[j][1] > 0:
                    j += 1
                while i < m and a[i][1] > 0:
                    i += 1
                i += 1
                j += 1
                result.append((i, j))

    if len(result) > N_SPAN:
        step = len(result) / N_SPAN
        return [result[int((idx + 1) * step) - 1] for idx in range(N_SPAN)]

    return result


def get_pooler_tensor(segments_idxs):
    padded_idx_batch = []
    max_seg = 0
    max_len_all = 0

    for seg_idx, max_len in segments_idxs:
        max_seg = max(max_seg, len(seg_idx))
        max_len_all = max(max_len_all, max_len)
        if len(seg_idx) == 0:
            padded_idx_batch.append(torch.full((1, 1), -1, dtype=torch.long))
            continue
        padded = torch.stack(
            [pad(segment, (0, max_len - len(segment)), value=-1) for segment in seg_idx]
        )
        padded_idx_batch.append(padded)

    max_seg = max(max_seg, 1)
    max_len_all = max(max_len_all, 1)

    def pad2d(tensor, height, width):
        return pad(tensor, (0, width - tensor.size(1), 0, height - tensor.size(0)), value=-1)

    padded_idx_batch = torch.stack(
        [pad2d(tensor, max_seg, max_len_all) for tensor in padded_idx_batch]
    )

    mask = padded_idx_batch != -1
    safe_idx = padded_idx_batch.masked_fill(~mask, 0)
    return {"safe_idx": safe_idx, "mask": mask}


def prepare_pooler_v2(
    student_starts,
    student_offset_mapping,
    teacher_starts,
    teacher_offset_mapping,
):
    student_seg_idxs, teacher_seg_idxs = [], []

    for student_start, student_offset, teacher_start, teacher_offset in zip(
        student_starts,
        student_offset_mapping,
        teacher_starts,
        teacher_offset_mapping,
    ):
        student_seg_idx = []
        teacher_seg_idx = []

        student_start = student_start.item()
        teacher_start = teacher_start.item()
        longest_common_offset = [(student_start, teacher_start)] + longest_common_subsequence(
            student_offset,
            teacher_offset,
            student_start,
            teacher_start,
        )

        student_max_len, teacher_max_len = 1, 1
        for idx in range(1, len(longest_common_offset)):
            prev_student, prev_teacher = longest_common_offset[idx - 1]
            cur_student, cur_teacher = longest_common_offset[idx]
            student_seg = torch.arange(prev_student, cur_student, dtype=torch.long)
            teacher_seg = torch.arange(prev_teacher, cur_teacher, dtype=torch.long)
            if student_seg.numel() == 0 or teacher_seg.numel() == 0:
                continue
            student_seg_idx.append(student_seg)
            teacher_seg_idx.append(teacher_seg)
            student_max_len = max(student_max_len, student_seg.numel())
            teacher_max_len = max(teacher_max_len, teacher_seg.numel())

        student_seg_idxs.append((student_seg_idx, student_max_len))
        teacher_seg_idxs.append((teacher_seg_idx, teacher_max_len))

    return get_pooler_tensor(student_seg_idxs), get_pooler_tensor(teacher_seg_idxs)


def pool_hidden_states(hidden_states: Sequence[torch.Tensor], safe_idx: torch.Tensor, pooler_mask: torch.Tensor):
    batch_size = safe_idx.size(0)
    batch_idxs = torch.arange(batch_size, device=safe_idx.device)[:, None, None]
    denom = pooler_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(dtype=torch.float32)

    pooled_hidden_states = []
    for hidden_state in hidden_states:
        gathered = hidden_state[batch_idxs, safe_idx] * pooler_mask.unsqueeze(-1)
        pooled = gathered.sum(dim=2) / denom
        pooled_hidden_states.append(pooled)

    return tuple(pooled_hidden_states)


def pool_hidden_states_sra(
    hidden_states: Sequence[torch.Tensor],
    attentions: Sequence[torch.Tensor],
    safe_idx: torch.Tensor,
    pooler_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices,
    is_causal: bool = True,
):
    batch_size = safe_idx.size(0)
    batch_idxs = torch.arange(batch_size, device=safe_idx.device)[:, None, None]
    pooled_hidden_states = []
    span_weights = []

    if not isinstance(layer_indices, (list, tuple)):
        layer_indices = [layer_indices]

    mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
    mask_4d = mask_2d.unsqueeze(1)

    for layer_idx in layer_indices:
        attn_idx = max(int(layer_idx) - 1, 0)
        attn_tensor = None
        if attentions is not None and len(attentions) > attn_idx:
            attn_tensor = attentions[attn_idx]

        if attn_tensor is None:
            raise ValueError("SRA pooling requires attention tensors. Provide SRA-style ones fallback before calling pool_hidden_states_sra.")
        elif is_causal:
            weights = attn_tensor.sum(dim=1)[:, -1].detach()
        else:
            weights = (attn_tensor * mask_4d).sum(dim=(1, 2)).detach()

        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-5)
        weights = weights.unsqueeze(-1)[batch_idxs, safe_idx] * pooler_mask.unsqueeze(-1)

        gathered = hidden_states[layer_idx][batch_idxs, safe_idx] * pooler_mask.unsqueeze(-1)
        gathered = gathered * weights

        hidden_state_mean = gathered.sum(2) / weights.sum(2).clamp(min=1e-5)
        pooled_hidden_states.append(hidden_state_mean)
        span_weights.append(weights.sum(2))

    return tuple(pooled_hidden_states), torch.stack(span_weights)


class CrossTokenizerDollyDataset(Dataset):
    def __init__(
        self,
        args,
        student_tokenizer,
        teacher_tokenizer,
        path: str,
        split: str,
        num: int,
        ratio: float,
        rng_sample,
    ):
        del ratio, rng_sample
        self.args = args
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.model_type = args.model_type
        self.examples = self._load_examples(path, split)
        self.answers = [example["output"] if isinstance(example["output"], list) else [example["output"]] for example in self.examples]
        self.num = len(self.examples) if num == -1 else min(num, len(self.examples))

    def _load_examples(self, path: str, split: str) -> List[Dict[str, str]]:
        file_path = resolve_data_file(path, f"{split}.jsonl")
        with open(file_path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        sample = self.examples[index]
        output = sample["output"][0] if isinstance(sample["output"], list) else sample["output"]
        return {
            "prompt": sample["prompt"],
            "output": output,
        }

    def _build_student_prompt_batch(self, prompts: List[str]):
        return _tokenize_with_padding_side(
            self.student_tokenizer,
            prompts,
            padding_side="left",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_prompt_length,
        )

    def _build_full_batch(self, tokenizer, texts: List[str], padding_side: str):
        return _tokenize_with_padding_side(
            tokenizer,
            texts,
            padding_side=padding_side,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

    def _build_prompt_lengths(self, tokenizer, prompts: List[str]) -> torch.Tensor:
        tokenized = _tokenize_with_padding_side(
            tokenizer,
            prompts,
            padding_side="right",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return tokenized["attention_mask"].sum(dim=1)

    def _build_student_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lengths: torch.Tensor):
        labels = input_ids.clone()
        input_lengths = attention_mask.sum(dim=1)
        labels[attention_mask == 0] = -100
        for idx in range(labels.size(0)):
            labels[idx, : prompt_lengths[idx]] = -100
            labels[idx, input_lengths[idx] :] = -100
        return labels

    def _build_alignment(self, prompts: List[str], student_inputs, teacher_inputs):
        student_prompt_lengths = self._build_prompt_lengths(self.student_tokenizer, prompts)
        teacher_prompt_lengths = self._build_prompt_lengths(self.teacher_tokenizer, prompts)

        student_offset_mapping = student_inputs.pop("offset_mapping")
        teacher_offset_mapping = teacher_inputs.pop("offset_mapping")
        student_pooler, teacher_pooler = prepare_pooler_v2(
            student_prompt_lengths,
            student_offset_mapping,
            teacher_prompt_lengths,
            teacher_offset_mapping,
        )
        return {
            "student_safe_idx": student_pooler["safe_idx"],
            "student_pooler_mask": student_pooler["mask"],
            "teacher_safe_idx": teacher_pooler["safe_idx"],
            "teacher_pooler_mask": teacher_pooler["mask"],
            "span_mask": student_pooler["mask"].any(dim=-1),
        }

    def build_teacher_batch_from_texts(self, prompts: List[str], responses: List[str]):
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]
        student_inputs = self._build_full_batch(self.student_tokenizer, full_texts, padding_side="right")
        teacher_inputs = self._build_full_batch(self.teacher_tokenizer, full_texts, padding_side="right")
        return self._build_alignment(prompts, student_inputs, teacher_inputs), teacher_inputs

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if hasattr(value, "to") and callable(value.to):
                try:
                    return value.to(device)
                except TypeError:
                    pass
            if isinstance(value, dict):
                return {key: move(inner) for key, inner in value.items()}
            if isinstance(value, Mapping):
                return {key: move(inner) for key, inner in value.items()}
            return value

        def move_mapping_in_place(mapping_obj):
            if hasattr(mapping_obj, "keys"):
                for key in list(mapping_obj.keys()):
                    mapping_obj[key] = move(mapping_obj[key])
            return mapping_obj

        model_data = move_mapping_in_place(model_data)
        no_model_data = move_mapping_in_place(no_model_data)
        gen_data = move_mapping_in_place(gen_data)
        return model_data, no_model_data, gen_data

    def collate(self, samples):
        prompts = [sample["prompt"] for sample in samples]
        outputs = [sample["output"] for sample in samples]
        full_texts = [prompt + output for prompt, output in zip(prompts, outputs)]

        student_prompt_batch = self._build_student_prompt_batch(prompts)
        student_inputs = self._build_full_batch(self.student_tokenizer, full_texts, padding_side="right")
        teacher_inputs = self._build_full_batch(self.teacher_tokenizer, full_texts, padding_side="right")

        prompt_lengths = self._build_prompt_lengths(self.student_tokenizer, prompts)
        labels = self._build_student_labels(
            student_inputs["input_ids"],
            student_inputs["attention_mask"],
            prompt_lengths,
        )
        alignment = self._build_alignment(prompts, student_inputs, teacher_inputs)

        model_batch = {
            "input_ids": student_inputs["input_ids"],
            "attention_mask": student_inputs["attention_mask"],
        }
        if self.model_type in ["gpt2"]:
            position_ids = torch.arange(model_batch["input_ids"].size(1), dtype=torch.long).unsqueeze(0).expand_as(model_batch["input_ids"])
            model_batch["position_ids"] = position_ids

        no_model_batch = {
            "label": labels,
            "loss_mask": (labels != -100).float(),
            "teacher_model_batch": teacher_inputs,
            "alignment": alignment,
            "prompt_texts": prompts,
        }

        gen_data = {
            "input_ids": student_prompt_batch["input_ids"],
            "attention_mask": student_prompt_batch["attention_mask"],
        }
        return model_batch, no_model_batch, gen_data
