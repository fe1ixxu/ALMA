<p align="center">
    <img alt="ALMA" src="alma_logo.png" width="500" height="203">
</p>

<div align="center">
    
# ALMA: Advanced Language Model-based translator
</div>

<p align="center">
<a href="LICENSE" alt="MIT License"><img src="https://img.shields.io/badge/license-MIT-FAD689.svg" /></a>
<a href="https://arxiv.org/abs/2309.11674" alt="ALMA paper"><img src="https://img.shields.io/badge/ALMA-Paper-D9AB42" /></a>
<a href="https://arxiv.org/abs/2401.08417" alt="ALMA-R paper"><img src="https://img.shields.io/badge/ALMA--R-Paper-F6C555" /></a>
<!-- <a href="https://notes.aimodels.fyi/alma-a-new-training-method-that-boosts-translation-performance-for-large-language-models/"><img alt="Summary Link" src="https://img.shields.io/badge/summary-link-F6C555" /></a> -->
<a href="https://www.clsp.jhu.edu/" alt="jhu"><img src="https://img.shields.io/badge/Johns_Hopkins_University-BEC23F" /></a>
<a href="https://www.microsoft.com/en-us/research/" alt="MSlogo"><img src="https://img.shields.io/badge/Microsoft-B1B479?logo=microsoft" /></a>
<a href="https://twitter.com/fe1ixxu">
  <img src="https://img.shields.io/twitter/follow/haoranxu?style=social&logo=twitter"
      alt="follow on Twitter"></a>
</p>

**ALMA** (**A**dvanced **L**anguage **M**odel-based Tr**A**nslator) is a many-to-many LLM-based translation model,  which adopts a new translation model paradigm: it begins with fine-tuning on monolingual data and is further optimized using high-quality parallel data. This two-step fine-tuning process ensures strong translation performance.

### **[ALMA-R](https://arxiv.org/pdf/2401.08417v2.pdf) (NEW!)** builds upon ALMA models, with further LoRA fine-tuning with our proposed **Contrastive Preference Optimization (CPO)** as opposed to the Supervised Fine-tuning used in ALMA. CPO fine-tuning requires our [triplet preference data](https://huggingface.co/datasets/haoranxu/ALMA-R-Preference) for preference learning. ALMA-R now can matches or even exceeds GPT-4 or WMT winners!

The original ALMA repository can be found [here](https://github.com/fe1ixxu/ALMA/tree/a3cc7877752779346312bb07798172eadc83d692).

# News 🌟
⭐ May.1 CPO paper has been accepted at ICML 2024!

⭐ Mar.22 2024 CPO method now is merged at [huggingface trl](https://github.com/huggingface/trl)! See details [here](https://github.com/huggingface/trl/pull/1382).

⭐ Jan.16 2024 **ALMA-R** is Released! Please check more details with our new paper: [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://arxiv.org/abs/2401.08417).

⭐ Jan.16 2024 The ALMA paper: [A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models](https://arxiv.org/abs/2309.11674) has been accepted at ICLR 2024! Check out more details [here](https://openreview.net/forum?id=farT6XXntP)!

# Contents 📄
- [Download ALMA(-R) Models and Dataset](#download-alma-r-models-and-dataset-)
- [Environment Setup](#environment-setup-)
- [Evaluation](#evaluation-)
- [Training](#training-)
- [Data Information](#data-information-)
- [FAQs](#faqs-)

:star: Supports :star:
  - AMD and Nvidia Cards
  - Data Parallel Evaluation
  - Also support LLaMA-1, LLaMA-2, OPT, Faclon, BLOOM, MPT
  - LoRA Fine-tuning
  - Monolingual data fine-tuning, parallel data fine-tuning

<p align="center">
<img src="almar.png" width="700" height="560">
</p>

# Download ALMA(-R) Models and Dataset 🚀

We release six translation models presented in the paper:
- ALMA-7B
- ALMA-7B-LoRA
- **ALMA-7B-R (NEW!)**: Further LoRA fine-tuning upon ALMA-7B-LoRA with contrastive preference optimization.
- ALMA-13B
- ALMA-13B-LoRA
- **ALMA-13B-R (NEW!)**: Further LoRA fine-tuning upon ALMA-13B-LoRA with contrastive preference optimization (BEST MODEL!). 
  
*We have also provided the WMT'22 and WMT'23 translation outputs from ALMA-13B-LoRA and ALMA-13B-R in the `outputs` directory. These outputs also includes our outputs of baselines and can be directly accessed and used for subsequent evaluations.*

Model checkpoints are released at huggingface:
|     Models    | Base Model Link | LoRA Link |
|:-------------:|:---------------:|:---------:|
|    ALMA-7B    |        [haoranxu/ALMA-7B](https://huggingface.co/haoranxu/ALMA-7B)        |     -     |
|  ALMA-7B-LoRA |        [haoranxu/ALMA-7B-Pretrain](https://huggingface.co/haoranxu/ALMA-7B-Pretrain)        |     [haoranxu/ALMA-7B-Pretrain-LoRA](https://huggingface.co/haoranxu/ALMA-7B-Pretrain-LoRA)     |
|  **ALMA-7B-R (NEW!)** |        [haoranxu/ALMA-7B-R (LoRA merged)](https://huggingface.co/haoranxu/ALMA-7B-R)        |     -    |
|    ALMA-13B   |        [haoranxu/ALMA-13B](https://huggingface.co/haoranxu/ALMA-13B)        |     -     |
| ALMA-13B-LoRA |        [haoranxu/ALMA-13B-Pretrain](https://huggingface.co/haoranxu/ALMA-13B-Pretrain)        |     [haoranxu/ALMA-13B-Pretrain-LoRA](https://huggingface.co/haoranxu/ALMA-13B-Pretrain-LoRA)     |
| **ALMA-13B-R (NEW!)** |        [haoranxu/ALMA-13B-R (LoRA merged)](https://huggingface.co/haoranxu/ALMA-13B-R)        |    -   |

**Note that `ALMA-7B-Pretrain` and `ALMA-13B-Pretrain` are NOT translation models. They only experience stage 1 monolingual fine-tuning (20B tokens for the 7B model and 12B tokens for the 13B model), and should be utilized in conjunction with their LoRA models.** 

Datasets used by ALMA and ALMA-R are also released at huggingface now (NEW!)
|     Datasets    | Train / Validation| Test |
|:-------------:|:---------------:|:---------:|
|    Human-Written Parallel Data (ALMA)    |        [train and validation](https://huggingface.co/datasets/haoranxu/ALMA-Human-Parallel)        |     [WMT'22](https://huggingface.co/datasets/haoranxu/WMT22-Test)    |
|  Triplet Preference Data |        [train](https://huggingface.co/datasets/haoranxu/ALMA-R-Preference)        |   [WMT'22](https://huggingface.co/datasets/haoranxu/WMT22-Test) and [WMT'23](https://huggingface.co/datasets/haoranxu/WMT23-Test)   |


A quick start to use our best system (ALMA-13B-R) for translation. An example of translating "我爱机器翻译。" into English:
```
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# Load base model and LoRA weights
model = AutoModelForCausalLM.from_pretrained("haoranxu/ALMA-13B-R", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("haoranxu/ALMA-13B-R", padding_side='left')

# Add the source sentence into the prompt template
prompt="Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"
input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.cuda()

# Translation
with torch.no_grad():
    generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=20, do_sample=True, temperature=0.6, top_p=0.9)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(outputs)
```

The general translation prompt is:
```
Translate this from <source language name> into <target language name>:
<source language name>: <source language sentence>
<target language name>:
```

# Environment Setup 🔧
```
conda create -n alma-r python=3.11
conda activate alma-r
```
If you use **Nvidia GPUs**, install torch with cuda 11.8
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
If you use **AMD GPUs**, install torch with ROCm 5.6
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```
Then install other dependencies:
```
bash install_alma.sh
```
# Evaluation 💻
### Evaluation on ALMA-13B-R
This is a quick start to evaluate our ALMA-13B-R model. To produce translation outputs for WMT'22 in both en→cs and cs→en directions (If you want to evaluate WMT'23 instead, simply pass `--override_test_data_path haoranxu/WMT23-Test`. Please look at `evals/alma_13b_r_wmt23.sh` as an example), run the following command:
```
accelerate launch --config_file configs/deepspeed_eval_config_bf16.yaml \
    run_llmmt.py \
    --model_name_or_path haoranxu/ALMA-13B-R \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs en-cs,cs-en \
    --mmt_data_path ./human_written_data/ \
    --per_device_eval_batch_size 1 \
    --output_dir ./your_output_dir/ \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir

```
The generated outputs will be saved in the `your_output_dir`. The translation file for the `en→cs` direction is named `test-en-cs`, and the file for the cs→en direction is `test-cs-en`.
We have prepared a bash file for the user to easily run the evaluation:
```
bash evals/alma_13b_r.sh ${your_output_dir} ${test_pairs}
```
The variable `${test_pairs}` denotes the translation directions you wish to evaluate. It supports testing multiple directions at once. For example, you can use `de-en,en-de,en-cs,cs-en`. Once the bash script completes its execution, both the BLEU scores and COMET results will be automatically displayed.

**Note that this will perform data-parallel evaluation supported by deepspeed: that is, placing a single full copy of your model onto each available GPU and splitting batches across GPUs to evaluate on K GPUs K times faster than on one**. For those with limited GPU memory, we offer an alternative method. The user can pass `--multi_gpu_one_model` to run the process by distributing a single model across multiple GPUs. Please see evaluation examples in `evals/alma_13b_r.sh` or  `evals/*no_parallel` files.

### Few-Shot In-Context Learning
To append examples in the prompt, you need to pass the `--few_shot_eval_path` flag and specify the location of their shot files. As a demonstration, you can execute the following command:
```
bash evals/llama-2-13b-5-shot.sh ${OUTPUT_DIR} ${test_pairs}
```

# Training 🔥
Here we show how to 
- **(NEW!) contrastive Preference Optmization Upon ALMA Models (ALMA→ALMA-R).**
- fine-tune LLaMA-2-7B on monolingual OSCAR data (stage 1)
- fine-tune human-written parallel data fine-tuning once stage 1 is completed, including full-weight and LoRA fine-tuning (stage 2)

## **CPO Fine-Tuning**
To run the CPO fine-tuning with our triplet preference data, run the following command:
```
bash runs/cpo_ft.sh ${your_output_dir}
```
### OSCAR Monolingual Fine-Tuning
To execute the OSCAR monolingual fine-tuning, use the following command:
```
bash runs/mono_ft.sh ${your_output_dir}
```
### Parallel Data Fine-Tuning (Full-Weight)
Once the monolingual data fine-tuning is complete, proceed to the parallel data fine-tuning using the full-weight approach. Execute the following command:
```
bash runs/parallel_ft.sh ${your_output_dir} $training_pairs$
```
where `training_pairs` is the translation directions you considered. The default is all 10 directions: `de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru`.

### Parallel Data Fine-Tuning (LoRA)
In Stage 2, there's also an option to employ LoRA for fine-tuning on the parallel data. To do so, execute the following command:
```
bash runs/parallel_ft_lora.sh ${your_output_dir} $training_pairs$
```

# Data Information 💾
Human-written training dataset, along with the WMT'22 test dataset, can be found in the `human_written_data` directory. Within this directory, there are five subfolders, each representing one of the five language pairs. Each of these subfolders contains the training, development, and test sets for its respective language pair.
```
-deen
    -train.de-en.json
    -valid.de-en.json
    -test.de-en.json
    -test.en-de.json
    ....
-csen
-isen
-ruen
.....

```
The data format in json files  must be:
```
{
  "translation":
  {
      "src(de)": "source sentence",
      "tgt(en)": "target sentence",
}
}
```
Within this directory, there are two additional subfolders specifically designed for few-shot in-context learning:
- `Filtered-5-shot`: This contains the "filtered shots" as referenced in the paper.
- `HW-5-shot`: This contains the "randomly extracted human-written data" mentioned in the paper.

# FAQs ❓
### What language directions do ALMA and ALMA-R support?
Currently, ALMA supports 10 directions: English↔German, Englishs↔Czech, Englishs↔Icelandic, Englishs↔Chinese, Englishs↔Russian. However, it may surprise us in other directions :)

### When should I stop fine-tuning at stage 1?
Our 7B and 13B models are trained on 20B and 12B tokens, respectively. However, as indicated in the paper, fine-tuning 1B tokens should boost the performance substantially. The steps required to fine-tune 1 billion tokens also vary based on your batch size. In our case, the batch size is calculated as follows: 16 GPUs * 4 (batch size per GPU) * 4 (gradient accumulation steps) = 256. With a sequence length of 512, we need approximately 8,000 steps to train on 1 billion tokens, calculated as 10^9 / (256*512) ≈8000 steps. However, you may choose to fine-tune more steps to get better performance.

### How to decide the interleave probability at stage 1?
Please find the reasons for interleave probability selection for stage 1 in Appendix D.1 in the [paper](https://arxiv.org/pdf/2309.11674.pdf)!

# Reference
Please find more details for ALMA models in our [paper](https://arxiv.org/abs/2309.11674) or the [summary](https://notes.aimodels.fyi/alma-a-new-training-method-that-boosts-translation-performance-for-large-language-models/) of the paper.
```
@misc{xu2023paradigm,
      title={A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models}, 
      author={Haoran Xu and Young Jin Kim and Amr Sharaf and Hany Hassan Awadalla},
      year={2023},
      eprint={2309.11674},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Please also find more detailed information for the ALMA-R model with Contrastive Preference Optimization in the [paper](https://arxiv.org/pdf/2401.08417v2.pdf).
```
@misc{xu2024contrastive,
      title={Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation}, 
      author={Haoran Xu and Amr Sharaf and Yunmo Chen and Weiting Tan and Lingfeng Shen and Benjamin Van Durme and Kenton Murray and Young Jin Kim},
      year={2024},
      eprint={2401.08417},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
