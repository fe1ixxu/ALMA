from dataclasses import dataclass, field
from typing import Optional
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether load model with int8"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether use PEFT (parameter efficient fine-tuning)"
            )
        },
    )
    lora_rank: int = field(
        default=16,
        metadata={
            "help": (
                "The rank for LoRA"
            )
        },
    )
    cpo_beta: float = field(
        default=0.1,
        metadata={
            "help": (
                "Beta for CPO training. Use --beta inseatd. This is deprecated and will be deleted in the future."
            )
        },
    )
    multi_gpu_one_model: bool = field(
        default=False,
        metadata={
            "help": "Use multiple GPUs to load one model."
        },
    )
    peft_model_id: str = field(
        default="",
        metadata={
            "help": (
                "PEFT model location"
            )
        },
    )
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    language_pairs: str = field(default="", metadata={"help": "training language pairs"})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    mmt_data_path: Optional[str] = field(default=None, metadata={"help": "The input MMT training data path."})
    override_test_data_path: Optional[str] = field(default=None, metadata={"help": "This will override the default test data in the mmt data"})
    cpo_data_path: Optional[str] = field(default=None, metadata={"help": "The input CPO training data path."})
    mono_data_path: Optional[str] = field(default=None, metadata={"help": "The input mono data training data path."})
    oscar_data_path: Optional[str] = field(default=None, metadata={"help": "The input Oscar mono data name."})
    oscar_data_lang: Optional[str] = field(default=None, metadata={"help": "The input Oscar mono data language."})
    text_test_file:  Optional[str] = field(default=None, metadata={"help": "A single test data file in text format, this will override the mmt_data_path and override_test_data_path"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes, truncate the number of test examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    ignore_prompt_token_for_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore the prompt tokens in the loss computation or not."
        },
    )
    use_ul2: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable mixture of denoisers from UL2 model."
        },
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum new tokens to generate except the prompt."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Beam size for generation"
            )
        }
    )

    display_num_translations: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "Number of translations will be displayed after translation."
            )
        }
    )

    right_pad: bool = field(
        default=False,
        metadata={
            "help": "Use right pad for training, especially for models like MPT."
        },
    )

    use_prefix_lm: bool = field(
        default=False,
        metadata={
            "help": "Use prefix language model, especially for models like MPT."
        },
    )
    few_shot_eval_path: str = field(
        default="",
        metadata={
            "help": "The path for few show evaluation"
        },
    )
    use_target_lang_prompt_eval: bool = field(
        default=False,
        metadata={
            "help": "Enable prompt from target language, e.g., in Chinese, the prompt is 将其从英语翻译成汉语：......"
        },
    )

    interleave_probs: str = field(
        default="",
        metadata={
            "help": "Usung interleave to concatenate datasets, with probabilities of p1,p2,p3,..., splited by commas"
        },
    )
    suffix_eval_file: str = field(
        default="",
        metadata={
            "help": "The suffix for the eval file: test-src-tgt'suffix_eval_file'"
        },
    )

    cpo_scorer: str = field(
        default="xcomet_kiwi",
        metadata={
            "help": "The scorer of CPO, e.g., using xcomet, kiwi, or both of them (xcomet-kiwi) for CPO training"
        },
    )


    # predict_source_lang: str = field(default="", metadata={"help": "The source language for testing"})
    # predict_target_lang: str = field(default="en", metadata={"help": "The target language for testing"})

    suffix: Optional[str] = field(default="", metadata={"help": "The suffix of the training file."})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
