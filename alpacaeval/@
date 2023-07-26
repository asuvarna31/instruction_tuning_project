import csv
import json
import torch
import argparse
import transformers
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--device_id", type = int, default = 3)
parser.add_argument("--temp", type = float, default = 1.0)
parser.add_argument("--name", type = str, default = 'alpaca 7b')
parser.add_argument("--input_data", type = str, default = 'alpaca_eval_davinci003.json')
parser.add_argument("--save_generations", type = str, default = 'alpaca_eval_alpaca7b.json')
parser.add_argument("--model_path", type = str, default = "/local2/hbansal/recover_weights_alpaca_7b/")

args = parser.parse_args()

model_path = args.model_path

model = transformers.AutoModelForCausalLM.from_pretrained(model_path, 
                                                            device_map = {"": torch.device(f"cuda:{args.device_id}")},
                                                            torch_dtype = torch.float16,
                                                            low_cpu_mem_usage=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def main():

    datasets = {'helpful_base', 'vicuna', 'koala', 'oasst', 'selfinstruct'}
    
    with open(args.input_data, 'r') as f:
        data = json.load(f)

    result = []
    for example in tqdm(data):
        if example['dataset'] in datasets:
            response = {}
            response['dataset'] = example['dataset']
            response['generator'] = args.name
            instruction = example['instruction']
            response['instruction'] = instruction
            input_text = PROMPT_DICT['prompt_no_input'].format(instruction = instruction)
            inputs = tokenizer(input_text, return_tensors="pt")
            output_texts = model.generate(inputs=inputs.input_ids.to(f"cuda:{args.device_id}"), max_new_tokens = 300, num_return_sequences=1, temperature = args.temp, do_sample=True)
            output_texts = tokenizer.batch_decode(output_texts, skip_special_tokens=True)
            response['output'] = output_texts[0][len(input_text):]
            result.append(response)
            with open(args.save_generations, 'w') as f:
                json.dump(result, f)

if __name__ == '__main__':
    main()

'''
    S1: python eval_inference_alpaca.py --device_id 7 --input_data model_outputs/alpaca_eval_davinci003.json --save_generations model_outputs/alpaca_7b_5k.json --name alpaca_7b_5k --model_path /local2/hbansal/it/alpaca_5k/
S2: export OPENAI_API_KEY="afacac" (terminal)
S3: alpaca_eval --model_outputs model_outputs/alpaca_7b_5k.json --annotat
ors_config chatgpt_fn
'''
