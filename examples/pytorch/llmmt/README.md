This repo is for fine-tuning large language model on Machine translation, instruction tuning, monolingual data fine-tuning, or combination of them.

## Environment Setup
Running the following command and it will install two virtual environments: `llmmt` and `comet`. `llmmt` is for model training and `comet` is for evaluation.
```
cd /home/aiscuser/; wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh; bash Anaconda3-2023.03-1-Linux-x86_64.sh; source ~/.bashrc; cd /home/aiscuser/LLMMT/examples/pytorch/llmmt/
bash install.sh
```
## Data Collection
Dataset is already copied at `/home/aiscuer/filtered_wmt22/` folder by running `install.sh` above. `filtered_wmt22` has 8 folders corresponding 8 language pairs. Each language pair folder contains
1K, 10K, 100K, 1M, 5M, 20M parallel data (5M only available for `deen` and `ruen`, and 20M available for `ruen`).
```
-deen
    -train.de-en-1000.json
    -train.de-en-10000.json
    -train.de-en-100000.json
    ....
-csen
-isen
-ruen
.....
```
The input data format in json files is (and must be):
```
{
  "translation":
  {
      "src(de)": "source sentence",
      "tgt(en)": "target sentence",
}
}
```

## Training
Activate training env:
```
conda activate llmmt
```
All LLM fine-tuning codes are written in a single file `run_clm.py`. I write different bash files for different training settings. Evaluation will automatically run after training finish.
### Fine-tuning LLM on parallel dataset (full weight fine-tune)
To fine-tuning llama-2-7b on multilingual translation:
```
bash runs/llama_2_7b.sh ${exp_name} ${NUM_OF_TRAINING_DATA} ${LANGUAGE_PAIRS}
```
Three variants mean:
```
${exp_name}: it indicates the checkpoint name and wandb name (if you enable it). It will store the checkpoint at `/home/aiscuser/checkpoints/llmmt-pre/${exp_name}`.
${NUM_OF_TRAINING_DATA}: number of training examples per pair, e.g., 1000, 10000, 100000....
${LANGUAGE_PAIRS}: Language pairs included in MMT.
```
For example, fine-tuning llama-2-7b on en->de,de->en,en->ru,ru->en (language pair split by comma), with 100K training data each pair, and experiment name is `tmp`. 
```
bash runs/llama_2_7b.sh tmp 100000 en->de,de->en,en->ru,ru->en
```
In `runs/llama_2_7b.sh`, you can change to your own settings. Some key fields:
```
--model_name_or_path: model name in huggingface, e.g.,  mosaicml/mpt-7b for MPT model
--gradient_accumulation_steps: gradient accumaution steps, make sure the number here is the same as the number in deepspeed_train_config.yaml
--max_source_length: max length of source sentences (default 256)
--max_new_tokens: max length of target sentences (default 256)
--num_beams: size for beam seach (5 for 7B model and 1 for 13B model)
--report_to: training details will report to wandb, but setting to none can disable it 
```
**Note that you need to pass `--right_pad` if you plan to run MPT model**

### Fine-tuning LLM on parallel dataset (LoRA fine-tune)
To run LoRA fine-tuning on MMT:
```
bash runs/llama_2_7b_peft.sh ${exp_name} ${NUM_OF_TRAINING_DATA} ${LANGUAGE_PAIRS}
```
It will create LoRA layers for every down projection layer with `rank=16`. The difference between `llama_2_7b.sh` and `llama_2_7b_peft.sh` is we pass one more field `--use_peft`.
After enable LoRA layers, the script will only store the LoRA parameters, not the whole model. To reload a pre-trained LoRA, by passing `--peft_model_id {MODEL_PATH}`.

### Fine-tuning LLM on monolingual dataset
**Fine-tune on Oscar dataset (remote dataset)**

To fine-tune llama-2 on Oscar monolingual dataset:
```
bash runs/llama_2_7b_mono_oscar.sh ${exp_name}
```
The key fields:
```
--oscar_data_path: The name of oscar dataset (default: oscar-corpus/OSCAR-2301)
--oscar_data_lang: The languages used for training (default: ru,en)
--interleave_probs: probabilities to sample the language (default: 0.5,0.5)
--streaming: it is required to enabled to use large and remote dataset
```

**Fine-tune on your own monolingual data**

One can also create own monolingual data to fine-tune. An example is finertuning `ha` monolingual data on MPT-7B model by passing `--mono_data_path`:
```
bash runs/run_mono_ha.sh
```
The monolingual data created by your own should be the same as parallel data format but leave the `en` value as empty:
```
{
  "translation":
  {
      "src(de)": "source sentence",
      "tgt(en)": "",
}
}
```

### Fine-tuning LLM on Instruction dataset (Alpaca dataset)
To fine-tune on the instruction dataset like Alpaca, first download their data:

```
wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```

An example of fine-tuning llama-1 model on the alpaca data:
```
bash runs/run_instruct_tune.sh
```
The key field is `--instruct_data_path` to make the script know you are passing the instruction data.

### Co-Training
The script supports co-training on MT, monolingual and instruction data together, or any two of them. Based on the MT script `runs/llama_2_7b.sh`,
You can pass `--oscar_data_path` or `--mono_data_path` to enable monolingual data fine-tuning, and `--instruct_data_path` to enable instruction fine-tuning.

### Other Loss Objectives
All training above is the simple CLM loss objective. One can also pass `--use_prefix_lm` to enable prefix language modeling and `use_ul2` to enable UL2 (mix of denoisers). You must enable prefix LM to enable UL2. Note that prefix LM and UL2 only supports MPT model so far.

## Evaluation
Activate eval env:
```
conda activate comet
```

`eval.sh` supports the multi-gpu multilingual translation evaluation.
In `eval.sh`, change `$OUTPUT_DIR$` to the location your checkpoint and remove `--overwrite_output_dir` the checkpoint is not empty (it is empty only if you want to do zero-shot evaluation)

Pass `--use_peft` and `--peft_model_id` if your model is LoRA.

Pass `--few_shot_eval_path` (/home/aiscuser/gpt-MT/data-shots/QR/5-shot/) if you want few-shot evaluation. You need to increase `--max_source_tokens` to 768 and `--num_beam` to 1 for 5-shot evaluation

Pass `--use_target_lang_prompt_eval` if you want to use prompt from the target language (only for evaluation now).

Then,
```
bash eval.sh ${LANGUAGE_PAIR}
```

e.g., `bash eval.sh en-de,de-en,en-ru,ru-en` to evaluate these four directions.






