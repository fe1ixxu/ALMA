from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
from tqdm import tqdm
import argparse
import logging

LANG_TABLE = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "cs": "Czech",
    "is": "Icelandic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ha": "Hausa",
}
def load_dataset(eval_path, prefix, suffix, batch_size):
    eval_data = []
    batch = []
    with open(eval_path, encoding="utf-8") as f:
        line = f.readline()
        while line:
            prompt = prefix + line.strip() + suffix
            if len(batch) < batch_size:
                batch.append(prompt)
            else:
                eval_data.append(batch)
                batch = [prompt]
            line = f.readline() 
    eval_data.append(batch)
    return eval_data

def clean_outputstring(output, key_word):
    try:
        out = output.split(key_word)[1].split("\n")
        if out[0].strip() != "":
            return out[0].strip()
        elif out[1].strip() != "":
            ## If there is an EOL directly after the suffix, ignore it
            logging.info(f"Detect empty output, we ignore it and move to next EOL: {out[1].strip()}")
            return out[1].strip()
        else:
            logging.info(f"Detect empty output AGAIN, we ignore it and move to next EOL: {out[2].strip()}")
            return out[2].strip()
    except:
        logging.info("Can not recover the translation by moving to the next EOL.. Trying move to the next suffix")
        
    try:
        return output.split(key_word)[2].split("\n")[0].strip()
    except:
        logging.info("Can not solve the edge case, recover the translation to empty string!")
        return ""

def generate(args):
    # Load model and tokenizer, note that the fast tokenizer currently does not work correctly
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, padding_side='left')

    src_fullname = LANG_TABLE[args.src_lang]
    tgt_fullname = LANG_TABLE[args.tgt_lang]

    # prompt template for zero-shot translation
    prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: "
    suffix = f"\n{tgt_fullname}:"
    logging.info(f"The prompt we use is {prefix + suffix}")

    # Load eval data
    eval_data = load_dataset(args.eval_path, prefix, suffix, args.batch_size)

    # Generate
    cleaned_outputs = []
    for eval_batch in tqdm(eval_data):
        input_ids = tokenizer(eval_batch, return_tensors="pt", padding=True).input_ids.cuda()
        max_length = min(int(input_ids.shape[-1] * 2.5), args.max_token_in_seq)
        generated_ids = model.generate(input_ids, num_beams=args.beam_size, length_penalty=1, max_length=max_length, do_sample=True, top_k=50)
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for out in outputs:
            cleaned_out = clean_outputstring(out, key_word=suffix)
            cleaned_outputs.append(cleaned_out)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for out in cleaned_outputs:
            f.writelines([out, "\n"])
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--src_lang", type=str, required=True)
    parser.add_argument("--tgt_lang", type=str, required=True)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_token_in_seq", type=int, default=512)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)
    generate(args)


if __name__ == "__main__":
    main()
