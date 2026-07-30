import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from arguments import get_args
from data_utils.cross_tokenizer import CrossTokenizerDollyDataset, pool_hidden_states_sra
from data_utils.hf_data import find_data_file
from data_utils.lm_datasets import LMTrainDataset
from distillm.losses import frfd_distillation_loss, sra_geometric_loss, sra_soft_label_distill_loss, span_hidden_alignment_loss
from distillm.projector import Projector
from distillm.velocity_field import VelocityField
from rouge_metric import compute_metrics
from utils import (
    get_optimizer_params,
    get_shared_token_mappings,
    get_sra_distillation_schedule,
    get_sra_hidden_loss_weights,
    get_teacher_tokenizer,
    get_tokenizer,
    initialize,
    print_args,
    print_rank,
    save_rank,
)
from train_velocity_field import train as train_velocity_field


torch.set_num_threads(4)


def ensure_sra_attentions(attentions, model_config, attention_mask, device):
    batch_size, seq_len = attention_mask.size()
    num_heads = model_config.num_attention_heads
    if attentions is None or len(attentions) < model_config.num_hidden_layers or any(attn is None for attn in attentions):
        return tuple(
            torch.ones(
                (batch_size, num_heads, seq_len, seq_len),
                device=device,
                dtype=torch.float32,
            )
            for _ in range(model_config.num_hidden_layers)
        )
    return attentions


def get_student_model(args, tokenizer, device):
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.is_model_parallel = False
    student_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        device_map={"": device},
        torch_dtype=student_dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.resize_token_embeddings(len(tokenizer))
    model.sra_amp_dtype = student_dtype
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    proj_list = []
    for _ in range(args.num_distill_layers):
        weight_cpu = torch.empty(model.config.hidden_size, args.d_teacher, dtype=torch.float32)
        nn.init.orthogonal_(weight_cpu)
        proj_list.append(nn.Parameter(weight_cpu.to(device=model.device, dtype=torch.float32)))
    model.sra_proj_hidden_layers = nn.ParameterList(proj_list)
    return model


def get_teacher_model(args, teacher_tokenizer, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path, trust_remote_code=True)
    config.is_model_parallel = False
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        config=config,
        device_map={"": device},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.resize_token_embeddings(len(teacher_tokenizer))
    model.eval()
    return model


def get_velocity_field(args, device):
    device = f"cuda:{device}"
    velocity_field = VelocityField(
        d_input=args.d_teacher,
        d_model=args.velocity_d_model,
        num_distill_layers=args.num_distill_layers,
        n_layers=args.velocity_n_layers,
    )
    velocity_field.load_state_dict(torch.load(args.velocity_field_path, map_location=device, weights_only=True))
    velocity_field.to(device).eval()

    projector = Projector(d_student=args.d_student, d_teacher=args.d_teacher)
    projector.load_state_dict(torch.load(args.projector_path, map_location=device, weights_only=True))
    projector.to(device).eval()
    return velocity_field, projector


def get_optimizer(args, model):
    param_groups = get_optimizer_params(args, model)
    return AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_iters)
    if args.lr_decay_style == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=args.lr_min)
    if args.lr_decay_style == "noam":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5,
        )
    raise ValueError(f"Unsupported lr scheduler {args.lr_decay_style}")


def prepare_dataset(args, student_tokenizer, teacher_tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    eval_split = "dev" if find_data_file(args.gt_data_dir or args.data_dir, "dev.jsonl") else "valid"
    if args.do_train:
        data["train"] = CrossTokenizerDollyDataset(
            args, student_tokenizer, teacher_tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample
        )
        if args.do_valid:
            data["dev"] = CrossTokenizerDollyDataset(
                args, student_tokenizer, teacher_tokenizer, args.gt_data_dir or args.data_dir, eval_split, args.dev_num, args.dev_ratio, rng_sample
            )
    elif args.do_eval:
        data["test"] = CrossTokenizerDollyDataset(
            args, student_tokenizer, teacher_tokenizer, args.gt_data_dir or args.data_dir, eval_split, args.dev_num, args.dev_ratio, rng_sample
        )
    return data


def pt_loss(model, model_batch, no_model_batch):
    outputs = model(**model_batch, return_dict=True, use_cache=False)
    return nn.CrossEntropyLoss(ignore_index=-100)(outputs.logits.view(-1, outputs.logits.size(-1)), no_model_batch["label"].view(-1))


def evaluate(args, tokenizer, model, dataset, split, epoch, device):
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dist.get_rank(), num_replicas=dist.get_world_size())
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=dataset.collate)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )

    model.eval()
    all_loss = 0.0
    step = 0
    all_response_ids = []
    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():
        for model_batch, no_model_batch, gen_data in dataloader:
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            if args.eval_gen:
                gen_out = model.generate(**gen_data, generation_config=generation_config, max_new_tokens=args.max_length - gen_data["input_ids"].size(1))
                full_ids = gen_out.sequences
                full_ids = F.pad(full_ids, (0, args.max_length - full_ids.shape[1]), value=tokenizer.pad_token_id)
                all_response_ids.append(full_ids[:, gen_data["input_ids"].size(1):])
            all_loss += loss.item()
            step += 1

    res = {}
    if args.eval_gen and all_response_ids:
        responses = tokenizer.batch_decode(torch.cat(all_response_ids, dim=0), skip_special_tokens=True)
        references = dataset.answers
        responses = responses[: len(references)]
        res = compute_metrics(responses, references)

    avg_loss = all_loss / max(step, 1)
    if dist.get_rank() == 0:
        log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        
        metrics = {f"eval/{split}_loss": avg_loss}
        if res:
            for key, val in res.items():
                metrics[f"eval/{split}_{key}"] = val
        # log_metrics(metrics, step=epoch)
    model.train()
    return avg_loss, res


def finetune(args, tokenizer, teacher_tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model, velocity_field, projector, update_velocity_dict={}):
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(dataset["train"], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)

    if "pt_train" in dataset:
        pt_sampler = DistributedSampler(dataset["pt_train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
        pt_train_dataloader = DataLoader(dataset["pt_train"], sampler=pt_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["pt_train"].collate)
        pt_train_iter = iter(pt_train_dataloader)
    else:
        pt_train_iter = None

    teacher_schedule, student_schedule = get_sra_distillation_schedule(args)
    # hidden_loss_weights = get_sra_hidden_loss_weights(args, len(teacher_schedule))
    shared_student_ids, shared_teacher_ids = get_shared_token_mappings(tokenizer, teacher_tokenizer, device=f"cuda:{device}")
    amp_dtype = getattr(model, "sra_amp_dtype", torch.float16)
    use_grad_scaler = amp_dtype == torch.float16
    scaler = GradScaler(enabled=use_grad_scaler)
    loss_func = nn.CrossEntropyLoss()

    best_val_loss = 1e9
    best_val_iter = -1
    step, global_step = 1, 1
    total_loss = total_lm_loss = total_sra_kl_loss = total_contra_loss = total_time = 0.0

    if args.do_valid:
        best_val_loss, _ = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device)

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        cur_t = (epoch + 1) / args.epochs
        for model_batch, no_model_batch, gen_data in train_dataloader:
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            if pt_train_iter is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                except StopIteration:
                    pt_train_iter = iter(pt_train_dataloader)
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)

            st_time = time.time()
            with torch.no_grad():
                teacher_batch = no_model_batch["teacher_model_batch"]
                teacher_outputs = teacher_model(
                    **teacher_batch,
                    use_cache=False,
                    output_hidden_states=True,
                    output_attentions=True,
                )

            with autocast(dtype=amp_dtype):
                outputs = model(
                    **model_batch,
                    use_cache=False,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                logits = outputs.logits
                lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

                alignment = no_model_batch["alignment"]
                student_schedule_list = [int(x) for x in student_schedule]
                teacher_schedule_list = [int(x) for x in teacher_schedule]
                student_attentions = ensure_sra_attentions(outputs.attentions, model.config, model_batch["attention_mask"], model_batch["input_ids"].device)
                teacher_attentions = ensure_sra_attentions(teacher_outputs.attentions, teacher_model.config, teacher_batch["attention_mask"], teacher_batch["input_ids"].device)

                student_pooled_states, _ = pool_hidden_states_sra(
                    outputs.hidden_states,
                    student_attentions,
                    alignment["student_safe_idx"],
                    alignment["student_pooler_mask"],
                    model_batch["attention_mask"],
                    student_schedule_list + [len(outputs.hidden_states) - 1],
                )
                teacher_pooled_states, teacher_span_weights = pool_hidden_states_sra(
                    teacher_outputs.hidden_states,
                    teacher_attentions,
                    alignment["teacher_safe_idx"],
                    alignment["teacher_pooler_mask"],
                    teacher_batch["attention_mask"],
                    teacher_schedule_list + [len(teacher_outputs.hidden_states) - 1],
                )

                span_mask = alignment["span_mask"].float()
                student_span_logits = F.linear(
                    student_pooled_states[-1].float(),
                    model.get_output_embeddings().weight.float(),
                    model.get_output_embeddings().bias.float() if model.get_output_embeddings().bias is not None else None,
                )[:, :, shared_student_ids]
                teacher_span_logits = F.linear(
                    teacher_pooled_states[-1].float(),
                    teacher_model.get_output_embeddings().weight.float(),
                    teacher_model.get_output_embeddings().bias.float() if teacher_model.get_output_embeddings().bias is not None else None,
                )[:, :, shared_teacher_ids]
                invalid_span_mask = ~span_mask.bool()
                student_span_logits = student_span_logits.masked_fill(invalid_span_mask.unsqueeze(-1), 0.0)
                teacher_span_logits = teacher_span_logits.masked_fill(invalid_span_mask.unsqueeze(-1), 0.0)
                sra_kl_loss = sra_soft_label_distill_loss(student_span_logits, teacher_span_logits, distill_temperature=2.0)

                hidden_states_for_contra, _ = pool_hidden_states_sra(
                    outputs.hidden_states,
                    student_attentions,
                    alignment["student_safe_idx"],
                    alignment["student_pooler_mask"],
                    model_batch["attention_mask"],
                    student_schedule_list,
                )
                contra_loss = frfd_distillation_loss(
                    hidden_states_for_contra,
                    velocity_field,
                    projector,
                    list(range(len(hidden_states_for_contra))),
                    span_mask,
                    args.num_distill_layers,
                    cur_t=cur_t,
                    device=device,
                )
                distil_loss = sra_kl_loss #+ sra_hidden_loss + sra_geom_loss
                loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * (distil_loss + contra_loss)
                if pt_train_iter is not None:
                    loss = loss + args.lm_coef * pt_loss(model, pt_model_batch, pt_no_model_batch)

            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % args.gradient_accumulation_steps == 0:
                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            total_loss += loss.item() / args.log_interval
            total_lm_loss += lm_loss.item() / args.log_interval
            total_sra_kl_loss += sra_kl_loss.item() / args.log_interval
            # total_sra_hidden_loss += sra_hidden_loss.item() / args.log_interval
            # total_sra_geom_loss += sra_geom_loss.item() / args.log_interval
            total_contra_loss += contra_loss.item() / args.log_interval
            total_time += time.time() - st_time

            if global_step % args.log_interval == 0 and dist.get_rank() == 0:
                log_str = (
                    f"train | epoch {epoch:3d} | Iter: {step:6d}/{args.total_iters:6d} | "
                    f"global iter: {global_step:6d}/{args.total_iters:6d} | "
                    f"loss: {total_loss:.4f} | lm_loss: {total_lm_loss:.4f} | sra_kl_loss: {total_sra_kl_loss:.4f} | "
                    f"contra_loss: {total_contra_loss:.4f} | "
                    f"lr: {lr_scheduler.get_last_lr()[0]:.4e} | micro time: {total_time / args.log_interval:.3f}"
                )
                print_rank(log_str)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                # log_metrics(
                #     {
                #         "train/loss": total_loss,
                #         "train/lm_loss": total_lm_loss,
                #         "train/sra_kl_loss": total_sra_kl_loss,
                #         "train/sra_hidden_loss": total_sra_hidden_loss,
                #         "train/sra_geom_loss": total_sra_geom_loss,
                #         "train/contra_loss": total_contra_loss,
                #     },
                #     step=global_step,
                # )
                total_loss = total_lm_loss = total_sra_kl_loss = total_contra_loss = total_time = 0.0

            if args.do_valid and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                curr_avg_loss, _ = evaluate(args, tokenizer, model, dataset["dev"], "dev", global_step, device)
                if curr_avg_loss < best_val_loss and dist.get_rank() == 0:
                    best_val_loss = curr_avg_loss
                    best_val_iter = global_step
                    save_dir_path = os.path.join(args.save, str(global_step))
                    os.makedirs(save_dir_path, exist_ok=True)
                    tokenizer.save_pretrained(save_dir_path)
                    model.save_pretrained(save_dir_path, safe_serialization=False)
                model.train()

            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            step += 1
            if global_step > args.total_iters:
                break
            
        optimizer.zero_grad()
        if "contra" in args.type \
            and (epoch+1) % args.velocity_update_interval == 0 \
            and (epoch+1) != args.epochs:
            # update velocity field and projector
            assert velocity_field is not None and projector is not None
            model.eval()
            velocity_field.train()
            projector.train()
            os.makedirs(update_velocity_dict["args"].save, exist_ok=True)
            train_velocity_field(
                update_velocity_dict["args"], tokenizer, model,
                update_velocity_dict["optimizer"], update_velocity_dict["lr_scheduler"],
                dataset, 
                device, teacher_model,
                velocity_field, projector
            )
            model.train()
            velocity_field.eval()
            projector.eval()

    return model, best_val_iter


def main():
    args = get_args()
    assert args.cross_tokenizer_sra, "finetune_cross_tokenizer.py is only for the cross-tokenizer SRA path."
    args.deepspeed = False
    initialize(args)

    if dist.get_rank() == 0:
        print_args(args)
        os.makedirs(args.save, exist_ok=True)
        with open(os.path.join(args.save, "args.json"), "w") as handle:
            json.dump(vars(args), handle)

    device = torch.cuda.current_device()

    tokenizer = get_tokenizer(args)
    teacher_tokenizer = get_teacher_tokenizer(args)
    dataset = prepare_dataset(args, tokenizer, teacher_tokenizer)

    teacher_schedule, student_schedule = get_sra_distillation_schedule(args)
    args.num_distill_layers = len(teacher_schedule)
    args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dist.get_world_size() * args.gradient_accumulation_steps))
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.save_interval == -1:
        args.save_interval = args.train_iters_per_epoch
    if args.eval_interval == -1:
        args.eval_interval = args.train_iters_per_epoch

    model = get_student_model(args, tokenizer, device)
    teacher_model = get_teacher_model(args, teacher_tokenizer, device)
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    velocity_field, projector = get_velocity_field(args, device)
    
    update_velocity_dict = {}
    if "contra" in args.type:
        class UpdateVelocityFieldConfig:
            model_parallel = args.model_parallel
            type = "lm-distillm2" if "distillm2" in args.type else "lm"
            cross_tokenizer_sra = args.cross_tokenizer_sra
            batch_size = args.batch_size
            num_workers = args.num_workers
            model_type = args.model_type
            teacher_model_type = args.teacher_model_type
            model_path = args.model_path
            teacher_model_path = args.teacher_model_path
            teacher_tokenizer_path = args.teacher_tokenizer_path
            data_dir = args.data_dir
            gt_data_dir = args.gt_data_dir
            lm_data_dir = args.lm_data_dir
            train_num = args.train_num
            train_ratio = args.train_ratio
            dev_num = args.dev_num
            dev_ratio = args.dev_ratio
            max_length = args.max_length
            max_prompt_length = args.max_prompt_length
            num_teacher_layers = args.num_teacher_layers
            num_student_layers = args.num_student_layers
            num_distill_layers = args.num_distill_layers
            teacher_layers_mapping = args.teacher_layers_mapping
            student_encoder_layers_finetuned = args.student_encoder_layers_finetuned
            hidden_loss_weights = args.hidden_loss_weights
            sra_p = args.sra_p
            geom_loss_weight = args.geom_loss_weight
            epochs = args.velocity_epochs
            teacher_device = device
            student_device = device
            gradient_accumulation_steps = args.gradient_accumulation_steps
            total_iters = args.train_iters_per_epoch * args.velocity_epochs
            mid_log_num = args.mid_log_num
            log_interval = args.log_interval
            save = os.path.join(args.save, "velocity_field_update")
            lr = args.lr
            weight_decay = args.weight_decay
            lr_decay_style = args.lr_decay_style
            lr_min = args.lr_min
            peft = None  # Velocity field is not a PEFT model
            
        configs = UpdateVelocityFieldConfig()
        
        # add more to this dict, providing objs required for calling train_velocity_field function
        update_velocity_dict["args"] = configs
        optimizer_v = get_optimizer(configs, velocity_field)
        assert type(optimizer_v) is torch.optim.AdamW
        assert velocity_field is not None and projector is not None
        optimizer_v.add_param_group({'params': projector.parameters()})
        
        num_updates = args.epochs // args.velocity_update_interval
        if args.epochs % args.velocity_update_interval == 0 and num_updates != 1:
            num_updates -= 1
        configs.total_iters *= num_updates
        lr_scheduler_v = get_learning_rate_scheduler(configs, optimizer_v)
        configs.total_iters = configs.total_iters // num_updates
        update_velocity_dict["optimizer"] = optimizer_v
        update_velocity_dict["lr_scheduler"] = lr_scheduler_v
        # init dolly dataset for updating velocity field
        update_velocity_dict["dataset"] = {}

    if args.do_train:
        model, best_val_iter = finetune(args, tokenizer, teacher_tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model, velocity_field, projector, update_velocity_dict)
        if dist.get_rank() == 0:
            if best_val_iter == -1:
                model.save_pretrained(args.save, safe_serialization=False)
            else:
                best_ckpt_path = os.path.join(args.save, str(best_val_iter))
                import shutil
                for filename in os.listdir(best_ckpt_path):
                    src_path = os.path.join(best_ckpt_path, filename)
                    dst_path = os.path.join(args.save, filename)

                    if os.path.isfile(src_path):  # only copy files
                        shutil.copy2(src_path, dst_path)  # overwrite if exists
            tokenizer.save_pretrained(args.save)

if __name__ == "__main__":
    main()
