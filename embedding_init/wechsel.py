import torch
import numpy as np
from transformers import LlamaTokenizer, LlamaForCausalLM
from wechsel import WECHSEL, load_embeddings
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, help="Source model")
    parser.add_argument("--target_path", type=str, help="Target tokenizer (spm)")
    parser.add_argument("--target_lang", type=str, help="Target language code")
    parser.add_argument("--source_fasttext", default='en', type=str, help="Source fasttext embeddings")
    parser.add_argument("--target_fasttext", type=str, help="Target fasttext embeddings")
    parser.add_argument("--bilingual_dic", type=str, help="Bilingual dictionary")
    args = parser.parse_args()

    source_tokenizer = LlamaTokenizer.from_pretrained(args.source_path)
    target_tokenizer = LlamaTokenizer.from_pretrained(args.target_path)
    model = LlamaForCausalLM.from_pretrained(args.source_path)
    
    if not args.target_fasttext:
        args.target_fasttext = args.target_lang

    wechsel = WECHSEL(
        load_embeddings(args.source_fasttext),
        load_embeddings(args.target_fasttext),
        bilingual_dictionary=args.bilingual_dic,
    )

    input_embeddings, info = wechsel.apply(
        source_tokenizer,
        target_tokenizer,
        model.get_input_embeddings().weight.detach().numpy(),
    )

    output_embeddings, info = wechsel.apply(
        source_tokenizer,
        target_tokenizer,
        model.get_output_embeddings().weight.detach().numpy(),
    )

    # Filter out special tokens: <unk>, <s>, </s>
    target_embeddings = input_embeddings[3:]
    target_embeddings_output = output_embeddings[3:]

    # Save embeddings
    save_path = "embeddings"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    np.save(f"{save_path}/wechsel.{args.target_lang}.input.npy", input_embeddings)
    np.save(f"{save_path}/wechsel.{args.target_lang}.input.npy", output_embeddings)
    print(f"WECHSEL embeddings saved to {save_path}")