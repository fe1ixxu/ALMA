from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import LlamaTokenizer
import torch
import argparse

def generate(args):
    prompt_no_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{args.prompt}\n\n### Response:"
    ),
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, trust_remote_code=True).cuda()
    model.eval()
    input_ids = tokenizer(prompt_no_input, return_tensors="pt", max_length=args.max_token_in_seq, truncation=True).input_ids.cuda()
    with torch.no_grad():
        generated_ids = model.generate(input_ids, num_beams=args.beam_size, max_length=args.max_token_in_seq)
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(outputs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="decapoda-research/llama-7b-hf")
    parser.add_argument("--prompt", type=str, default="Write something:")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_token_in_seq", type=int, default=512)

    args = parser.parse_args()
    set_seed(args.seed)
    generate(args)
if __name__ == "__main__":
    main()