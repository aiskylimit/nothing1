import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from distillm.arguments import get_args

from distillm.lm_datasets import LMTrainDataset
from distillm.utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from distillm.utils import print_rank, get_rank
from distillm.utils import save_rank
from distillm.utils import all_gather
from distillm.utils import get_tokenizer, get_model

from distillm.losses import skewed_forward_kl, skewed_reverse_kl, forward_kl, reverse_kl, js_distance
from eval_math import compute_math_metric
from torch.func import functional_call
from logit_adapter import LogitAdapter
from torch.utils.checkpoint import checkpoint

from peft import PeftModel, LoraConfig, get_peft_model, TaskType

torch.set_num_threads(4)


def get_student_model(args, device):
    config = AutoConfig.from_pretrained(args.student_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try: model = AutoModelForCausalLM.from_pretrained(args.student_model_path, config=config, device_map={"": device}, torch_dtype=torch.bfloat16)
        except:
            model = AutoModelForCausalLM.from_pretrained(args.student_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()
        
        if args.peft is not None and args.student_peft_path is not None:
            if args.peft == "lora":
                model = PeftModel.from_pretrained(model, args.student_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)
    
    return model


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    # optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    elif args.lr_decay_style == "wrmup_cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_ratio * args.total_iters,
            num_training_steps=args.total_iters)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler

def setup_model_and_optimizer(args, ds_config, model, set_optim=True):
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler

def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    else:
        raise ValueError("Do train and do eval must set one")
        
    # pre-trained dataset
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data

def get_distil_loss(args, teacher_logits, no_model_batch, logits):
    if "sfkl" in args.type:
        distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    elif "srkl" in args.type:
        distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    elif "fkl" in args.type or args.type == "kd":
        distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
    elif "rkl" in args.type:
        distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
    else:
        raise NotImplementedError
    return distil_loss

def forward_kl_w(logits, teacher_logits, weight, t = 1.0):
    teacher_probs = F.softmax(teacher_logits / t, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits / t, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * weight.view(-1), dim=0) / torch.sum(weight.view(-1), dim=0)
    return distil_loss

def bilevel_optimization(model, sft_loss, delta_S, params):
    # grad_sft = torch.autograd.grad(sft_loss, params, retain_graph=True)
    # grad_delta_S = torch.autograd.grad(delta_S, params, retain_graph=True)

    model.backward(sft_loss, retain_graph=True) # DeepSpeed tự tính gradient và gán vào p.grad
    
    # Copy gradient ra biến riêng để lưu trữ
    grad_sft = []
    for p in params:
        if p.grad is not None:
            grad_sft.append(p.grad.clone().detach())
        else:
            grad_sft.append(torch.zeros_like(p))
            
    # Xóa grad để chuẩn bị tính cái tiếp theo
    model.zero_grad()

    # --- BƯỚC 2: Tính Gradient cho Loss 2 (Delta S) ---
    model.backward(delta_S, retain_graph=False)
    
    grad_delta_S = []
    for p in params:
        if p.grad is not None:
            grad_delta_S.append(p.grad.clone().detach())
        else:
            grad_delta_S.append(torch.zeros_like(p))
            
    model.zero_grad()

    with torch.no_grad():
        dot_sft_delta_S = sum((gs * gd).sum() for gs, gd in zip(grad_sft, grad_delta_S))
        norm_gd2 = sum((gd * gd).sum() for gd in grad_delta_S)

        lambda_t = torch.clamp(
            0.5 - dot_sft_delta_S / (norm_gd2 + 1e-6),
            min=0.0
        ).detach()

    with torch.no_grad():
        for p, gs, gd in zip(params, grad_sft, grad_delta_S):
            p.grad = gs + lambda_t * gd

class StudentMetaAccumulator:
    def __init__(self, student_model, args):
        self.student = student_model
        self.acc_steps = args.gradient_accumulation_steps
        # self.acc_steps = 1
        self.current_step = 0
        self.grad_buffer = {} # Lưu gradient tích lũy (detached)
        self.student_device = student_model.device
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")

    def accumulate_and_compute_anti_loss(self, model_batch, no_model_batch, args, logits):
        self.current_step += 1
        self.student.zero_grad()

        for k in model_batch:
            model_batch[k] = model_batch[k].to(self.student_device)
        no_model_batch["label"] = no_model_batch["label"].to(self.student_device)
        
        student_outputs = self.student(**model_batch, use_cache=False)
        student_logits = student_outputs.logits.float()

        no_model_batch["label"] = no_model_batch["label"]
        valid_mask = (no_model_batch["label"] != -100).float()
        probs = torch.softmax(student_logits.detach(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        weight_sft = entropy / math.log(self.student.config.vocab_size)   # [0,1]
        weight_sft = torch.clamp(weight_sft, min=0.0, max=1.0)
        weight_sft = weight_sft * valid_mask
        weight_sft = weight_sft / weight_sft[valid_mask.bool()].mean().clamp(min=1e-6)
        weight_sft = weight_sft.to(logits.device)

        k = 50
        s_topk_logits, s_topk_indices = torch.topk(student_logits, k, dim=-1)
        topk_logits_aligned = torch.gather(logits.float(), dim=-1, index=s_topk_indices.to(logits.device))

        anti_loss = forward_kl_w(topk_logits_aligned, s_topk_logits.to(logits.device), weight_sft, t=1.0)

        return anti_loss

def finetune(args, tokenizer: AutoTokenizer, model, logit_adapter: nn.Module, 
             optimizer: AdamW, lr_scheduler, dataset, device, student_model=None):
    print_rank("Start Fine-tuning")

    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)
    

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time, total_lm_loss, total_s_grad_norm, total_delta = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    student_accumulator = StudentMetaAccumulator(student_model, args)
        
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        model.train()
        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
                                    
            torch.cuda.synchronize()
            st_time = time.time()

            with torch.no_grad():
                outputs = model.model(**model_batch, use_cache=False, output_hidden_states=True)

            last_hidden_states = outputs.last_hidden_state
            base_logits = model.lm_head(last_hidden_states).detach().float().to(logit_adapter.device)
            h_base_logits = model.lm_head(outputs.hidden_states[25]).detach().float().to(logit_adapter.device)

            logits = logit_adapter(last_hidden_states.to(logit_adapter.device))
            logits = (base_logits + logits) / 1

            anti_loss = student_accumulator.accumulate_and_compute_anti_loss(model_batch, no_model_batch, args, logits)
            loss_kd = anti_loss
            
            if anti_loss is not None:
                tau = 1.0
                no_model_batch["label"] = no_model_batch["label"].to(logit_adapter.device)

                k = 50
                b_topk_logits, b_topk_indices = torch.topk(base_logits, k, dim=-1)
                topk_logits_aligned = torch.gather(logits.float(), dim=-1, index=b_topk_indices)
                base_loss = forward_kl(topk_logits_aligned, b_topk_logits, no_model_batch, t=tau)

                hb_topk_logits, hb_topk_indices = torch.topk(h_base_logits, k, dim=-1)
                topk_logits_aligned_hb = torch.gather(logits.float(), dim=-1, index=hb_topk_indices)
                h_base_loss = js_distance(topk_logits_aligned_hb, hb_topk_logits, no_model_batch, lam=0.5)

                loss = base_loss + h_base_loss
                trainable_named_params = {
                    n: p for n, p in logit_adapter.named_parameters() if p.requires_grad
                }
                bilevel_optimization(logit_adapter, loss, anti_loss, list(trainable_named_params.values()))
                base_loss = base_loss.detach()
                logit_adapter.step()
            else:
                pass

            if anti_loss is not None:
                dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
                global_loss = loss.item() / dp_world_size
                total_loss += global_loss

                dist.all_reduce(base_loss, dist.ReduceOp.SUM, group=dp_group)
                total_lm_loss += base_loss.item() / dp_world_size 

                dist.all_reduce(loss_kd, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = loss_kd.item() / dp_world_size 
                total_distil_loss += global_distil_loss

    
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_distil_loss, log_time, log_lm, log_delta, log_gr):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f} | lm_loss: {:.4f}, | S_delta: {:.4f}, | S_grad_norm: {:.4f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time, log_lm, log_delta, log_gr
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, 0, base_loss.item(), 0, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval),
                    total_distil_loss / (args.log_interval),
                    total_time / (args.log_interval), 
                    total_lm_loss / (args.log_interval),
                    total_delta / (args.log_interval),
                    total_s_grad_norm / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
                total_lm_loss, total_delta, total_s_grad_norm = 0.0, 0.0, 0.0

            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    raise NotImplementedError
                else:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        logit_adapter.module.save(save_dir_path + "/logit_adapter.pt")
                dist.barrier()

                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break

    if global_step % args.save_interval > 0:
        save_dir_path = os.path.join(args.save, str(global_step))                
        if dist.get_rank() == 0:
            os.makedirs(save_dir_path, exist_ok=True)
            print_rank(f"Model save to {save_dir_path}")
            tokenizer.save_pretrained(save_dir_path)
            logit_adapter.module.save(save_dir_path + "/logit_adapter.pt")
        dist.barrier()
     
    return model



def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    # ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["gradient_accumulation_steps"] = 1
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]  
    args.bf16 = "bf16" in ds_config and ds_config["bf16"]["enabled"]  
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch

    model = get_model(args, device)

    if args.student_model_type is None:
        args.student_model_type = args.model_type
    
    if args.student_model_path is not None:
        student_base_model = get_student_model(args, device)
        # student_base_model.resize_token_embeddings(model.config.vocab_size)
        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM, 
        #     inference_mode=False, 
        #     r=256, 
        #     lora_alpha=256,
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        #     lora_dropout=0.01
        # )
        # student_model = get_peft_model(student_base_model, peft_config)
        # student_model.print_trainable_parameters()
        student_model = student_base_model
        student_model.train()
    else:
        student_model = None
    
    logit_adapter = LogitAdapter(lm_head=model.lm_head)
    logit_adapter = logit_adapter.to(dtype=torch.bfloat16, device=device)
    print(' > number of logit adapter parameters: {}'.format(sum([p.nelement() for p in logit_adapter.parameters()])), flush=True)

    logit_adapter, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, logit_adapter, set_optim=args.do_train)
       
    model = finetune(args, tokenizer, model, logit_adapter, optimizer, lr_scheduler, dataset, device, student_model=student_model)
           
    
if __name__ == "__main__":
    main()