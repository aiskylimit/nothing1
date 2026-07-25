import multiprocessing
import os
import time
import torch
import json
import sys
import numpy as np
from indexed_dataset import make_builder
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse



def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--processed-data-dir", type=str, default=None)
    group.add_argument("--data-process-workers", type=int, default=-1)
    group.add_argument("--max-prompt-length", type=int, default=256)
    group.add_argument("--max_length", type=int, default=512)
    
    group.add_argument("--model-path", type=str)
    group.add_argument("--model-type", type=str)
    group.add_argument("--only-prompt", action="store_true")
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_data_args(parser)
    
    args, unknown = parser.parse_known_args()
    return args


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, padding_side="right")
        self.tokenizer = Encoder.tokenizer

    def encode(self, sample):
        full_tokens = Encoder.tokenizer.encode(sample["trace"])  
        prompt_tokens = sample["input_ids"]
        
        response_tokens = full_tokens[len(prompt_tokens):] + [Encoder.tokenizer.eos_token_id]
        
        # if len(prompt_tokens) > self.args.max_prompt_length:
        #     prompt_tokens = prompt_tokens[:self.args.max_prompt_length]
        
        return prompt_tokens, response_tokens, sample["problem"], sample['trace']



def main():
    print("OK")
    args = get_args()
        
    if 'generated' not in args.processed_data_dir:
        args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    dataset = load_dataset('VoCuc/self-metamath', split='train')
    
    
    encoder = Encoder(args)

    # 2. Mapping all datas with Encoder, with the help of multiprocessing
    pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
    encoded_docs = pool.imap_unordered(encoder.encode, dataset, chunksize=50)
    proc_start = time.time()
    
    bin_file = os.path.join(args.processed_data_dir, f"train_{0}.bin")
    idx_file = os.path.join(args.processed_data_dir, f"train_{0}.idx")

    if args.model_type!="qwen":
        binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
    else:
        binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint32)

    # put tokenized data into binary_builder
    inst_num = 0
    
    prompt_lens = []
    response_lens = []
    full_lens = []
    
    json_file = open(os.path.join(args.processed_data_dir, f"train.jsonl"), "w")
    
    for lid, (prompt, response, query_str, trace) in enumerate(encoded_docs):
        if prompt is None:
            continue

        if len(response) + len(prompt)> args.max_length:
            continue 
        if len(prompt) > args.max_prompt_length:
            continue
        
        if args.only_prompt:
            if len(prompt) < args.max_length:
                binary_builder.add_item(torch.IntTensor(prompt))
            else:
                continue
        else:
            binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))

        json_file.write(json.dumps({
            "query": query_str,
            "trace": trace,
        }) + "\n")

        prompt_lens.append(len(prompt))
        response_lens.append(len(response))
        full_lens.append(len(prompt) + len(response))

        inst_num += 1
        if lid % 1000 == 0:
            current = time.time()
            elapsed = current - proc_start
            print(f"Processed {lid} documents. {inst_num} instances.", f"({lid/elapsed} docs/s).", file=sys.stderr)

    # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
    binary_builder.finalize(idx_file)

    # close multiproceessing mapping
    pool.close()
    json_file.close()
            
    print("Data num", len(prompt_lens))
    print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
    print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))
    print("Full", "Mean:", np.mean(full_lens), "Max:", np.max(full_lens), "Min:", np.min(full_lens))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    print(tokenizer.eos_token, tokenizer.eos_token_id)


if __name__ == '__main__':
    main()