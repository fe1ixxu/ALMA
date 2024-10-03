#!/usr/bin/env python
# coding=utf-8

import copy
import os
from itertools import chain
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from peft import PeftModel
from collections import defaultdict
from transformers.trainer_callback import TrainerCallback
from datasets import concatenate_datasets
from tqdm import tqdm

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

        if os.path.isfile(pytorch_model_path) and torch.distributed.get_rank() == 0:
            os.remove(pytorch_model_path)
            # create an empty toy file to avoid error in deleting old checkpoints
            open(pytorch_model_path, 'w').close()
        return control

LANG_TABLE = {
    "en": "English",
    # Group 1:
    "da": "Danish",
    "nl": "Dutch",
    "de": "German",
    "is": "Icelandic",
    "no": "Norwegian",
    "sv": "Swedish",
    "af": "Afrikaans",
    # Group 2:
    "ca": "Catalan",
    "ro": "Romanian",
    "gl": "Galician",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
    # Group 3:
    "bg": "Bulgarian",
    "mk": "Macedonian",
    "sr": "Serbian",
    "uk": "Ukrainian",
    "ru": "Russian",
    # Group 4:
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
    "mg": "Malagasy",
    "fr": "French",
    # Group 5:
    "hu": "Hungarian",
    "el": "Greek",
    "cs": "Czech",
    "pl": "Polish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    # Group 6:
    "ka": "Georgian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fi": "Finnish",
    "et": "Estonian",
    # Group 7:
    "gu": "Gujarati",
    "hi": "Hindi",
    "mr": "Marathi",
    "ne": "Nepali",
    "ur": "Urdu",
    # Group 8:
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "tr": "Turkish",
    "uz": "Uzbek",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
}

NLLB_CODE = {
    "en": "eng_Latn",
    # Group 1:
    "da": "dan_Latn",
    "nl": "nld_Latn",
    "de": "deu_Latn",
    "is": "isl_Latn",
    "no": "nob_Latn",
    "sv": "swe_Latn",
    "af": "afr_Latn",
    # Group 2:
    "ca": "cat_Latn",
    "ro": "ron_Latn",
    "gl": "glg_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "es": "spa_Latn",
    # Group 3:
    "bg": "bul_Cyrl",
    "mk": "mkd_Cyrl",
    "sr": "srp_Cyrl",
    "uk": "ukr_Cyrl",
    "ru": "rus_Cyrl",
    # Group 4:
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "mg": "plt_Latn",
    "fr": "fra_Latn",
    # Group 5:
    "hu": "hun_Latn",
    "el": "ell_Grek",
    "cs": "ces_Latn",
    "pl": "pol_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    # Group 6:
    "ka": "kat_Geor",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fi": "fin_Latn",
    "et": "est_Latn",
    # Group 7:
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "ur": "urd_Arab",
    # Group 8:
    "az": "azj_Latn",
    "kk": "kaz_Cyrl",
    "ky": "kir_Cyrl",
    "tr": "tur_Latn",
    "uz": "uzn_Latn",
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
}

ISO1_ISO3_map = {
    "en": "eng",
    # Group 1:
    "da": "dan",
    "nl": "nld",
    "de": "deu",
    "is": "isl",
    "no": "nob",
    "sv": "swe",
    "af": "afr",
    # Group 2:
    "ca": "cat",
    "ro": "ron",
    "gl": "glg",
    "it": "ita",
    "pt": "por",
    "es": "spa",
    # Group 3:
    "bg": "bul",
    "mk": "mkd",
    "sr": "srp",
    "uk": "ukr",
    "ru": "rus",
    # Group 4:
    "id": "ind",
    "ms": "msa",
    "th": "tha",
    "vi": "vie",
    "mg": "mlg",
    "fr": "fra",
    # Group 5:
    "hu": "hun",
    "el": "ell",
    "cs": "ces",
    "pl": "pol",
    "lt": "lit",
    "lv": "lav",
    # Group 6:
    "ka": "kat",
    "zh": "zho",
    "ja": "jpn",
    "ko": "kor",
    "fi": "fin",
    "et": "est",
    # Group 7:
    "gu": "guj",
    "hi": "hin",
    "mr": "mar",
    "ne": "nep",
    "ur": "urd",
    # Group 8:
    "az": "aze",
    "kk": "kaz",
    "ky": "kir",
    "tr": "tur",
    "uz": "uzb",
    "ar": "arb",
    "he": "heb",
    "fa": "fas",
}

## Prefix and suffix for prompt in target language (only from English to target language if the target is non-English)
## Note that prefix and suffix for other languages are only used for zero-shot evaluation of other models.
## ALMA should only use English Prompt.
PREFIX = {
    "de": "Übersetzen Sie dies vom Englischen ins Deutsche:\nEnglisch: ",
    "fr": "Traduisez ceci de l'anglais vers le français :\nAnglais: ",
    "cs": "Přeložte toto z angličtiny do češtiny:\nanglicky: ",
    "is": "Þýddu þetta úr ensku yfir á íslensku:\nEnska: ",
    "zh": "将其从英文翻译成中文：\n英语：",
    "ja": "これを英語から日本語に翻訳してください:\n英語：",
    "ru": "Переведите это с английского на русский:\nАнглийский: ",
    "uk": "Перекладіть це з англійської на українську:\nАнглійська: ",
    "ha": "Fassara wannan daga Turanci zuwa Hausa:\nTuranci: ",
}

SUFFIX = {
    "en": "\nEnglish:",
    "de": "\nDeutsch:",
    "fr": "\nFrançais :",
    "cs": "\nčesky:",
    "is": "\nÍslenska:",
    "zh": "\n中文：",
    "ja": "\n日本語：",
    "ru": "\nРусский:",
    "uk": "\nУкраїнська:",
    "ha": "\nHausa:",
}



def load_mmt_dataset(pairs, data_args, model_args, training_args, logger):
    seen_files =set([])
    train_raw_data, valid_raw_data, test_raw_data = {}, {}, {}
    for pair in pairs:
        src_lang = pair.split("-")[0]
        tgt_lang = pair.split("-")[1]

        # The directory is always "xxen", e.g., deen
        first_lang = src_lang if src_lang != "en" else tgt_lang
        second_lang = "en"
        pair_dir = first_lang + second_lang
            
        h_suffix = f"-{data_args.suffix}" if data_args.suffix else ""
        train_file = os.path.join(data_args.mmt_data_path, pair_dir, f"train.{first_lang}-{second_lang}{h_suffix}.json")
        valid_file = os.path.join(data_args.mmt_data_path, pair_dir, f"valid.{first_lang}-{second_lang}.json")
        test_file = os.path.join(data_args.mmt_data_path, pair_dir, f"test.{src_lang}-{tgt_lang}.json")
        
        if not os.path.isfile(train_file):
            logger.info(f"Warning: training file {train_file} does not exist!")
        elif train_file not in seen_files and training_args.do_train:
            train_raw_data[f"{first_lang}-{second_lang}"] = load_dataset(
                "json",
                data_files={"train": train_file},
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
                )
        if not os.path.isfile(valid_file):
            logger.info(f"Warning: validation file {valid_file} does not exist!")
        elif valid_file not in seen_files and training_args.do_eval:
            valid_raw_data[f"{first_lang}-{second_lang}"] = load_dataset(
                "json",
                data_files={"validation": valid_file},
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                )

        if data_args.override_test_data_path and  training_args.do_predict:
            test_raw_data[f"{src_lang}-{tgt_lang}"] = load_dataset(
                data_args.override_test_data_path,
                f"{src_lang}-{tgt_lang}",
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                )
        elif not os.path.isfile(test_file):
            logger.info(f"Warning: test file {test_file} does not exist!")
        elif test_file not in seen_files and training_args.do_predict:
            test_raw_data[f"{src_lang}-{tgt_lang}"] = load_dataset(
                "json",
                data_files={"test": test_file},
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                )
            test_raw_data[f"{src_lang}-{tgt_lang}"] = test_raw_data[f"{src_lang}-{tgt_lang}"].rename_column("translation", f"{src_lang}-{tgt_lang}")

        seen_files.add(train_file)
        seen_files.add(valid_file)
        seen_files.add(test_file)

    return train_raw_data, valid_raw_data, test_raw_data

def load_a_single_text_file(pairs, data_args, model_args):
    assert len(pairs) == 1, "Specific translation text source file only needs one translation direction!"
    src_lang, tgt_lang = list(pairs)[0].split("-")
    test_raw_data = {}
    pair = f"{src_lang}-{tgt_lang}"
    test_raw_data[pair] = load_dataset(
        'text',
        data_files={"test": data_args.text_test_file},
        cache_dir=model_args.cache_dir,
        token=True if model_args.use_auth_token else None,
        )
    def format_features(example):
        return {pair: {src_lang: example["text"], tgt_lang: ""}}

    test_raw_data[pair] = test_raw_data[pair].map(format_features, remove_columns=["text"])

    return test_raw_data

def get_first_non_pad_index(input_tensor):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 1
    first_non_pad_index = (input_tensor != -100).nonzero(as_tuple=True)[0][0]
    return first_non_pad_index.item()

def get_first_special_index(input_tensor, special):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 1
    first_pad_index = (input_tensor == special).nonzero(as_tuple=True)[0]
    if len(first_pad_index) > 0:
        return first_pad_index[0].item()
    else:
        return -1

def get_first_special_index_batch(input_tensor, special):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 2
    matches = input_tensor.eq(special).long()
    indices = matches.argmax(dim=1)
    indices[matches.sum(dim=1) == 0] = -1
    return indices 

def get_first_non_specical_index(input_tensor, special):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 1
    first_non_pad_index = (input_tensor != special).nonzero(as_tuple=True)[0][0]
    return first_non_pad_index.item()

# Suffix for splitting and getting the generated sentences
def get_key_suffix(tgt_lang, data_args, additional_suffix):
    if data_args.use_target_lang_prompt_eval:
        return SUFFIX[tgt_lang]
    else:
        return f"\n{LANG_TABLE[tgt_lang]}:" + additional_suffix

def get_prompt_few_shot(source_lang, target_lang, ex, shots_eval_dict):
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    shots = shots_eval_dict[f"{source_lang}-{target_lang}"]
    prefix = f"Translate this from {src_fullname} to {tgt_fullname}:"
    shot_prompt = ""
    for shot in shots:
        shot_src = shot['source']
        shot_tgt = shot['target']
        shot_prompt += f"\n{src_fullname}: " + shot_src + f"\n{tgt_fullname}: " + shot_tgt
    suffix = f"\n{tgt_fullname}:"
    prompt = prefix + shot_prompt + f"\n{src_fullname}: " + ex[source_lang] + suffix
    return prompt

def get_prompt(source_lang, target_lang, ex, shots_eval_dict={}, use_target_lang_prompt_eval=False, encoder_decoder_type=False):
    if encoder_decoder_type == "aya101":
        return f"Translate to {LANG_TABLE[target_lang]}: " + ex[source_lang]
    elif encoder_decoder_type == "nllb":
        return ex[source_lang]

    if len(shots_eval_dict) != 0:
        return get_prompt_few_shot(source_lang, target_lang, ex, shots_eval_dict)
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    if use_target_lang_prompt_eval and target_lang != "en":
        prefix = PREFIX[target_lang]
        suffix = SUFFIX[target_lang]
    else:
        prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: "
        suffix = f"\n{tgt_fullname}:"
    prompt = prefix + ex[source_lang] + suffix
    return prompt

def check_add_eos(tokenized_inputs, tokenizer):
    if tokenized_inputs.input_ids[0][-1] != tokenizer.eos_token_id:
        for idx in range(len(tokenized_inputs.input_ids)):
            tokenized_inputs.input_ids[idx].append(tokenizer.eos_token_id)
            tokenized_inputs.attention_mask[idx].append(1)

def check_add_eos_right_pad(tokenized_inputs, tokenizer):
    for idx in range(len(tokenized_inputs.input_ids)):
        first_non_pad_idx = get_first_special_index(tokenized_inputs.input_ids[idx], tokenizer.pad_token_id)
        tokenized_inputs.input_ids[idx][first_non_pad_idx] = tokenizer.eos_token_id
        tokenized_inputs.attention_mask[idx][first_non_pad_idx] = 1

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def clean_outputstring(output, key_word, logger, split_idx):
    try:
        out = output.split(key_word)[split_idx].split("\n")
        if out[0].strip() != "":
            return out[0].strip()
        elif out[1].strip() != "":
            ## If there is an EOL directly after the suffix, ignore it
            logger.info(f"Detect empty output, we ignore it and move to next EOL: {out[1].strip()}")
            return out[1].strip()
        else:
            logger.info(f"Detect empty output AGAIN, we ignore it and move to next EOL: {out[2].strip()}")
            return out[2].strip()
    except:
        logger.info(f"Can not recover the translation by moving to the next EOL.. Trying move to the next suffix")
        
    try:
        return output.split(key_word)[2].split("\n")[0].strip()
    except:
        logger.info(f"Can not solve the edge case, recover the translation to empty string! The output is {output}")
        return ""

def load_model(data_args, model_args, training_args, tokenizer, logger):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and (training_args.do_train or training_args.do_predict ) and not training_args.overwrite_output_dir:
        last_checkpoint = training_args.output_dir
        # last_checkpoint = get_last_checkpoint(training_args.output_dir)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
        "max_length": data_args.max_source_length + data_args.max_new_tokens,
        # "norm_type": "low_precision_rmsnorm",
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    if "mosaicml/mpt" in model_args.model_name_or_path:
        config.attn_config["prefix_lm"] = data_args.use_prefix_lm

    ## Model Loading
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )

        
        AutoModelLoad = AutoModelForSeq2SeqLM if model_args.encoder_decoder_type else AutoModelForCausalLM

        if model_args.multi_gpu_one_model:
            model = AutoModelLoad.from_pretrained(
                model_args.model_name_or_path if last_checkpoint is None else last_checkpoint,
                device_map="auto",
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                use_flash_attention_2=model_args.use_flash_attention_2,
                torch_dtype=torch_dtype,
            )
        else:
            model = AutoModelLoad.from_pretrained(
                model_args.model_name_or_path if last_checkpoint is None else last_checkpoint,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                trust_remote_code=True,
                use_flash_attention_2=model_args.use_flash_attention_2,
            )
        model.generation_config.max_length = data_args.max_source_length + data_args.max_new_tokens
        model.generation_config.use_cache = True
    else:
        model = AutoModelLoad.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.use_peft:
        if model_args.peft_model_id:
            model = PeftModel.from_pretrained(model, model_args.peft_model_id, is_trainable=True)
            ## If still need to fine-tune
            # for name, param in model.named_parameters():
            #     if "lora_A" in name or "lora_B" in name:
            #         param.requires_grad = True
        else:
            config = LoraConfig(
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_rank * 2,
                target_modules="all-linear",
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM" if not model_args.encoder_decoder_type else "SEQ_2_SEQ_LM",
            )
            model = get_peft_model(model, config)
        print_trainable_parameters(model)
    return model

def load_tokenizer(data_args, model_args, training_args, logger):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": True if model_args.use_auth_token else None,
        "padding_side": 'left' if not data_args.right_pad else "right",
        "add_eos_token": False,
    }
        
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            **tokenizer_kwargs,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_preprocessed_data(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, shots_eval_dict, data_args, training_args, model_args):
    def tokenize_function_train_eval_left_pad(examples):
        inputs = []
        prompts = []
        for ex in examples["translation"]:
            source_lang, target_lang = list(ex.keys())
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt(source_lang, target_lang, ex)
                if model_args.chat_style:
                    chat_style_prompt = [{"role": "user", "content": prompt}]
                    chat_style_input = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": ex[target_lang]},
                        ]
                    prompt = tokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)
                    input_text = tokenizer.apply_chat_template(chat_style_input, tokenize=False, add_generation_prompt=False)
                else:
                    input_text = prompt + ex[target_lang]
                prompts.append(prompt)
                inputs.append(input_text)

            if f"{target_lang}-{source_lang}" in pairs:
                prompt = get_prompt(target_lang, source_lang, ex)
                if model_args.chat_style:
                    chat_style_prompt = [{"role": "user", "content": prompt}]
                    chat_style_input = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": ex[source_lang]},
                        ]

                    prompt = tokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)
                    input_text = tokenizer.apply_chat_template(chat_style_input, tokenize=False, add_generation_prompt=False)
                else:
                    input_text = prompt + ex[source_lang]
                prompts.append(prompt)
                inputs.append(input_text)
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length + data_args.max_new_tokens - 1, padding=padding, truncation=True, add_special_tokens=True if not model_args.chat_style else False)
        if not model_args.chat_style:
            check_add_eos(model_inputs, tokenizer)
        labels = copy.deepcopy(model_inputs)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            if data_args.ignore_prompt_token_for_loss:
                for idx, prompt in enumerate(prompts):
                    prompt = tokenizer(prompt, max_length=data_args.max_source_length, add_special_tokens=False).input_ids
                    first_non_pad_idx = get_first_non_pad_index(labels["input_ids"][idx])
                    labels["input_ids"][idx][first_non_pad_idx: first_non_pad_idx + len(prompt)] = [-100] * len(prompt) 
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_function_aya(examples):
        inputs = []
        prompts = []
        for inp, tgt in zip(examples["inputs"], examples["targets"]):
            assert model_args.chat_style, "It must be in a chat style with aya datasets"
            chat_style_prompt = [{"role": "user", "content": inp}]
            chat_style_input = [
                {"role": "user", "content": inp},
                {"role": "assistant", "content": tgt},
                ]

            prompt = tokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)
            input_text = tokenizer.apply_chat_template(chat_style_input, tokenize=False, add_generation_prompt=False)
            
            prompts.append(prompt)
            inputs.append(input_text)

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length + data_args.max_new_tokens - 1, padding=padding, truncation=True, add_special_tokens=False)
        # check_add_eos(model_inputs, tokenizer)
        labels = copy.deepcopy(model_inputs)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            if data_args.ignore_prompt_token_for_loss:
                for idx, prompt in enumerate(prompts):
                    prompt = tokenizer(prompt, max_length=data_args.max_source_length, add_special_tokens=False).input_ids
                    first_non_pad_idx = get_first_non_pad_index(labels["input_ids"][idx])
                    labels["input_ids"][idx][first_non_pad_idx: first_non_pad_idx + len(prompt)] = [-100] * len(prompt) 
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_function_train_eval_left_pad_enc_dec(examples):
        inputs = []
        targets = []
        for ex in examples["translation"]:
            source_lang, target_lang = list(ex.keys())
            if f"{source_lang}-{target_lang}" in pairs:
                inp = get_prompt(source_lang, target_lang, ex, encoder_decoder_type=model_args.encoder_decoder_type)
                inputs.append(inp)
                targets.append(ex[target_lang])
            if f"{target_lang}-{source_lang}" in pairs:
                inp = get_prompt(target_lang, source_lang, ex, encoder_decoder_type=model_args.encoder_decoder_type)
                inputs.append(inp)
                targets.append(ex[source_lang])
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True, add_special_tokens=True)
        labels = tokenizer(targets, max_length=data_args.max_new_tokens, padding=padding, truncation=True, add_special_tokens=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_function_train_eval_right_pad(examples):
        inputs = []
        prompts = []
        for ex in examples["translation"]:
            source_lang, target_lang = list(ex.keys())
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt(source_lang, target_lang, ex)
                prompts.append(prompt)
                inputs.append(prompt + ex[target_lang])
            if f"{target_lang}-{source_lang}" in pairs:
                prompt = get_prompt(target_lang, source_lang, ex)
                prompts.append(prompt)
                inputs.append(prompt + ex[source_lang])
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length + data_args.max_new_tokens, padding=padding, truncation=True, add_special_tokens=True)
        check_add_eos_right_pad(model_inputs, tokenizer)
        labels = copy.deepcopy(model_inputs)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if data_args.use_prefix_lm:
            assert data_args.ignore_prompt_token_for_loss
            model_inputs["prefix_mask"] = []

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            if data_args.ignore_prompt_token_for_loss:
                for idx, prompt in enumerate(prompts):
                    prompt = tokenizer(prompt, max_length=data_args.max_source_length, add_special_tokens=False).input_ids
                    labels["input_ids"][idx][: len(prompt)] = [-100] * len(prompt) 
                    if data_args.use_prefix_lm:
                        prefix_mask = [0] * len(model_inputs["attention_mask"][idx])
                        prefix_mask[: len(prompt)] = [1] * len(prompt)
                        model_inputs["prefix_mask"].append(prefix_mask)
                    
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_function_train_mono(examples):
        if data_args.use_prefix_lm:
            inputs = {"input_ids": [], "attention_mask": [], "prefix_mask": []}      
        else:
            inputs = {"input_ids": [], "attention_mask": []}
        block_size = data_args.max_source_length + data_args.max_new_tokens
        for ex in examples["translation"]:
            lang1, lang2 = list(ex.keys())
            lang = lang1 if lang1 != "en" else lang2

            for lang in [lang1, lang2]:
                if ex[lang] == "":
                    continue
                _input = tokenizer(ex[lang], max_length=4096, add_special_tokens=True)
                _input['input_ids'].append(tokenizer.eos_token_id)
                _input['attention_mask'].append(1)
                if data_args.use_prefix_lm:
                    _input['prefix_mask'] = [0] * len(_input['attention_mask'])
                    inputs["prefix_mask"].append(_input['prefix_mask'])
                inputs["input_ids"].append(_input['input_ids'])
                inputs['attention_mask'].append(_input['attention_mask'])
            
        
        concatenated_inputs = {k: list(chain(*inputs[k])) for k in inputs.keys()}
        total_length = len(concatenated_inputs[list(inputs.keys())[0]])
        total_length = (total_length // block_size) * block_size

        model_inputs = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_inputs.items()
        }

        model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
        return model_inputs

    def tokenize_function_train_oscar_mono(examples):
        if data_args.use_prefix_lm:
            inputs = {"input_ids": [], "attention_mask": [], "prefix_mask": []}      
        else:
            inputs = {"input_ids": [], "attention_mask": []}
        block_size = data_args.max_source_length + data_args.max_new_tokens
        for ex in examples["raw_text"]:
            _input = tokenizer(ex.strip(), max_length=4096, add_special_tokens=True)
            _input['input_ids'].append(tokenizer.eos_token_id)
            _input['attention_mask'].append(1)
            if data_args.use_prefix_lm:
                _input['prefix_mask'] = [0] * len(_input['attention_mask'])
                inputs["prefix_mask"].append(_input['prefix_mask'])
            inputs["input_ids"].append(_input['input_ids'])
            inputs['attention_mask'].append(_input['attention_mask'])
            
        
        concatenated_inputs = {k: list(chain(*inputs[k])) for k in inputs.keys()}
        total_length = len(concatenated_inputs[list(inputs.keys())[0]])
        total_length = (total_length // block_size) * block_size

        model_inputs = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_inputs.items()
        }

        model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
        return model_inputs

    def tokenize_function_train_nllb_pretrain(examples):
        inputs = {"input_ids": [], "attention_mask": []}
        block_size = data_args.max_source_length + data_args.max_new_tokens
        for ex in examples["raw_text"]:
            _input = tokenizer(ex.strip(), max_length=4096, add_special_tokens=True)
            _input['input_ids'].append(tokenizer.eos_token_id)
            _input['attention_mask'].append(1)
            inputs["input_ids"].append(_input['input_ids'])
            inputs['attention_mask'].append(_input['attention_mask'])
            
        concatenated_inputs = {k: list(chain(*inputs[k])) for k in inputs.keys()}
        total_length = len(concatenated_inputs[list(inputs.keys())[0]])
        total_length = (total_length // block_size) * block_size

        model_inputs = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_inputs.items()
        }
        model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
        return model_inputs

    def tokenize_function_test(examples):
        prompts = []
        feature_name = list(examples.keys())[0]
        source_lang, target_lang = feature_name.split("-")
        for ex in examples[feature_name]:
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt(source_lang, target_lang, ex, shots_eval_dict, data_args.use_target_lang_prompt_eval, model_args.encoder_decoder_type)
                if model_args.chat_style:
                    prompt = [{"role": "user", "content": prompt}]
                    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
        original_padding_side = tokenizer.padding_side
        if original_padding_side != "left":
            tokenizer.padding_side = "left"
        model_inputs = tokenizer(prompts, max_length=data_args.max_source_length, padding=padding, truncation=True, add_special_tokens=True if not model_args.chat_style else False)
        tokenizer.padding_side = original_padding_side
        if data_args.use_prefix_lm:
            model_inputs["prefix_mask"] = []
            for idx, prompt in enumerate(prompts):
                prefix_mask = model_inputs["attention_mask"][idx]
                model_inputs["prefix_mask"].append(prefix_mask)
        return model_inputs

    def tokenize_function_aya_test(examples):
        prompts = []
        for inp, tgt in zip(examples["inputs"], examples["targets"]):
            prompt = inp
            if model_args.chat_style:
                prompt = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        original_padding_side = tokenizer.padding_side
        if original_padding_side != "left":
            tokenizer.padding_side = "left"
        model_inputs = tokenizer(prompts, max_length=data_args.max_source_length, padding=padding, truncation=True, add_special_tokens=True if not model_args.chat_style else False)
        tokenizer.padding_side = original_padding_side
        return model_inputs
    
    # Preprocessing the datasets.
    if data_args.mmt_data_path or data_args.mono_data_path:
        column_names_mmt = ["translation"]
    if data_args.oscar_data_path:
        if "oscar" in data_args.oscar_data_path:
            column_names_oscar = ["id", "meta", "text", "raw_text"]
        elif data_args.oscar_data_path == "allenai/c4":
            column_names_oscar = ["text", "timestamp", "url", "raw_text"]
        elif data_args.oscar_data_path == "cc100":
            column_names_oscar = ["id", "text", "raw_text"]
    if data_args.aya_datasets:
        column_names_aya = ["inputs", "targets", "language", "language_code", "annotation_type", "user_id"]
    if data_args.nllb_pretrain_data_path:
        column_names_nllb = ["raw_text"]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    padding = "max_length"

    train_datasets, eval_datasets, test_datasets = None, None, None
    mmt_train_eval_tok_func = tokenize_function_train_eval_right_pad if data_args.right_pad else tokenize_function_train_eval_left_pad
    if model_args.encoder_decoder_type:
        mmt_train_eval_tok_func = tokenize_function_train_eval_left_pad_enc_dec
    
    if training_args.do_train:
        processed_datasets = []
        if data_args.mmt_data_path:
            for lg_pair, sub_raw_data in train_raw_data["mmt"].items():
                train_dataset = sub_raw_data["train"]
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    if not data_args.streaming:
                        train_dataset = train_dataset.map(
                            mmt_train_eval_tok_func,
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=column_names_mmt,
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc="Running tokenizer on MMT train dataset",
                        )
                    else:
                        train_dataset = train_dataset.map(
                            mmt_train_eval_tok_func,
                            batched=True,
                            remove_columns=column_names_mmt,
                        )    
                processed_datasets.append(train_dataset)

        if data_args.aya_datasets:
            train_dataset = train_raw_data["aya"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            with training_args.main_process_first(desc="Aya dataset map pre-processing"):
                assert not data_args.streaming, "Streaming is not supported for Aya datasets now"
                train_dataset = train_dataset.map(
                    tokenize_function_aya,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names_aya,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on Aya train dataset",
                )
            processed_datasets.append(train_dataset)
        

        if data_args.mono_data_path:
            train_dataset = train_raw_data["mono"]['train']
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                if not data_args.streaming:
                    train_dataset = train_dataset.map(
                        tokenize_function_train_mono,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names_mmt,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on monolingual train dataset",
                    )
                else:
                    train_dataset = train_dataset.map(
                        tokenize_function_train_mono,
                        batched=True,
                        remove_columns=column_names_mmt,
                    )
            processed_datasets.append(train_dataset)
            
        if "oscar" in train_raw_data:
            train_dataset = train_raw_data["oscar"]
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    tokenize_function_train_oscar_mono,
                    batched=True,
                    remove_columns=column_names_oscar,
                )
            processed_datasets.append(train_dataset)

        if "nllb_pretrain" in train_raw_data:
            train_dataset = train_raw_data["nllb_pretrain"]
            with training_args.main_process_first(desc="NLLB Pre-train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    tokenize_function_train_nllb_pretrain,
                    batched=True,
                    remove_columns=column_names_nllb,
                )
            processed_datasets.append(train_dataset)
             
        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets = train_datasets.shuffle(seed=training_args.seed)

    if training_args.do_eval:
        processed_datasets = []
        for lg_pair, sub_raw_data in valid_raw_data.items():
            eval_dataset = sub_raw_data["validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    mmt_train_eval_tok_func,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names_mmt,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer valid dataset",
                )
            processed_datasets.append(eval_dataset)
        eval_datasets = concatenate_datasets(processed_datasets)
        eval_datasets = eval_datasets.shuffle(seed=training_args.seed)

    if training_args.do_predict:
        test_datasets = {"mmt": {}, "aya": {}}
        if data_args.mmt_data_path:
            for lg_pair, sub_raw_data in test_raw_data["mmt"].items():
                test_dataset = sub_raw_data["test"]
                if data_args.max_test_samples is not None:
                    max_test_samples = min(len(test_dataset), data_args.max_test_samples)
                    test_dataset = test_dataset.select(range(max_test_samples))
                with training_args.main_process_first(desc="test dataset map pre-processing"):
                    test_dataset = test_dataset.map(
                        tokenize_function_test,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=[lg_pair],
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer test dataset",
                    )
                test_datasets["mmt"][lg_pair] = test_dataset
        if data_args.aya_datasets:
            for lg in test_raw_data["aya"].keys():
                test_dataset = test_raw_data["aya"][lg]
                if data_args.max_test_samples is not None:
                    max_test_samples = min(len(test_dataset), data_args.max_test_samples)
                    test_dataset = test_dataset.select(range(max_test_samples))
                with training_args.main_process_first(desc="test dataset map pre-processing"):
                    test_dataset = test_dataset.map(
                        tokenize_function_aya_test,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names_aya,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer Aya test dataset",
                    )
                test_datasets["aya"][lg] = test_dataset
        
    return train_datasets, eval_datasets, test_datasets

def preprocess_cpo_data(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, shots_eval_dict, data_args, training_args, model_args):

    def get_chosen_reject(example, target_lang):
        sys1_score_key = f"gpt4_{target_lang}_{data_args.cpo_scorer}"
        sys2_score_key = f"alma_{target_lang}_{data_args.cpo_scorer}"
        ref_score_key = f"ref_{target_lang}_{data_args.cpo_scorer}"

        sys1_output_key = f"gpt4_{target_lang}"
        sys2_output_key = f"alma_{target_lang}"
        ref_output_key = target_lang


        # Human eval
        if "Delta" in example and example["Delta"] != 0:
            if example["Delta"] > 0:
                return example[sys1_output_key], example[sys2_output_key]
            else:
                return example[sys2_output_key], example[sys1_output_key]

        # Defining the sentences and their scores
        sentences = [example[ref_output_key], example[sys1_output_key], example[sys2_output_key]]
        scores = [example[ref_score_key], example[sys1_score_key], example[sys2_score_key]]

        # Finding the indexes for the highest and lowest scores
        highest_score_index = scores.index(max(scores))
        lowest_score_index = scores.index(min(scores))

        # Assigning the corresponding sentences
        highest_score_sentence = sentences[highest_score_index]
        lowest_score_sentence = sentences[lowest_score_index]
        return highest_score_sentence, lowest_score_sentence
            
    def meet_requirements(prompt_tok, example, target_lang):
        # if prompt is too long
        if len(prompt_tok) > data_args.max_source_length:
            return False

        # if the order is fixed, e.g., it has to be en->de
        if "required_directions" in example and example["required_directions"] != "":
            tgt = example["required_directions"].split("-")[1]
            if tgt != target_lang:
                return False
        return True 

    def cpo_prompt_function(examples):
        new_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }
        for ex in examples["translation"]:
            source_lang, target_lang = ex["language_pair"].split("-")
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt(source_lang, target_lang, ex)
                if model_args.chat_style:
                    chat_style_prompt = [{"role": "user", "content": prompt}]
                    prompt = tokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)
                prompt_tok = tokenizer(prompt, max_length=data_args.max_source_length, padding=True, truncation=True, add_special_tokens=True if not model_args.chat_style else False).input_ids
                if meet_requirements(prompt_tok, ex, target_lang):
                    new_examples["prompt"].append(prompt)
                    chosen, rejected = get_chosen_reject(ex, target_lang)
                    new_examples["chosen"].append(chosen)
                    new_examples["rejected"].append(rejected)
            if f"{target_lang}-{source_lang}" in pairs:
                prompt = get_prompt(target_lang, source_lang, ex)
                if model_args.chat_style:
                    chat_style_prompt = [{"role": "user", "content": prompt}]
                    prompt = tokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)
                prompt_tok = tokenizer(prompt, max_length=data_args.max_source_length, padding=True, truncation=True, add_special_tokens=True if not model_args.chat_style else False).input_ids
                if meet_requirements(prompt_tok, ex, source_lang):
                    new_examples["prompt"].append(prompt)
                    chosen, rejected = get_chosen_reject(ex, source_lang)
                    new_examples["chosen"].append(chosen)
                    new_examples["rejected"].append(rejected)
        return new_examples

    def aya_cpo_function(examples):
        new_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            if model_args.chat_style:
                chat_style_prompt = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)
            prompt_tok = tokenizer(prompt, max_length=data_args.max_source_length, padding=True, truncation=True, add_special_tokens=True if not model_args.chat_style else False).input_ids
            if len(prompt_tok) < data_args.max_source_length:
                new_examples["prompt"].append(prompt)
                new_examples["chosen"].append(chosen)
                new_examples["rejected"].append(rejected)
        return new_examples

    # Preprocessing the datasets.
    train_datasets, eval_datasets, test_datasets = None, None, None
    if training_args.do_train:
        processed_datasets = []
        if data_args.cpo_data_path:
            for lg_pair, sub_raw_data in train_raw_data["mmt"].items():
                train_dataset = sub_raw_data["train"]
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="CPO train dataset map pre-processing"):
                    if not data_args.streaming:
                        train_dataset = train_dataset.map(
                            cpo_prompt_function,
                            batched=True,
                            batch_size=1,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=["translation"],
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc="Running CPO preprocessing",
                        )
                    else:
                        train_dataset = train_dataset.map(
                            cpo_prompt_function,
                            batched=True,
                            batch_size=1,
                            remove_columns=["translation"],
                        )    
                processed_datasets.append(train_dataset)
        
        if data_args.aya_datasets:
            for lg in train_raw_data["aya"].keys():
                train_dataset = train_raw_data["aya"][lg]["train"]
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="Aya CPO dataset map pre-processing"):
                    train_dataset = train_dataset.map(
                        aya_cpo_function,
                        batched=True,
                        batch_size=1,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on Aya CPO train dataset",
                    )
                processed_datasets.append(train_dataset)
            
        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets = train_datasets.shuffle(seed=training_args.seed)        

    return train_datasets, eval_datasets, test_datasets
