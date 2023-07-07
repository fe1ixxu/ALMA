import random
from collections.abc import Mapping
import numpy as np
import torch
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)
import copy

from transformers import default_data_collator
from .utils import get_first_non_specical_index, get_first_special_index, get_first_special_index_batch

def random_spans_noise_mask(length, mean_noise_span_length, noise_density):
    """
    A copy from https://github.com/EleutherAI/oslo/blob/main/oslo/transformers/tasks/data_t5_pretraining.py#L230 (inception)
    This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens, num_noise_spans
    )

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]

@dataclass
class DataCollatorForUL2(DataCollatorMixin):
    """

    Data collator used for UL2

    """
    model: AutoModelForCausalLM
    tokenizer: PreTrainedTokenizerBase
    r_denoising: bool = True
    r_probability: float = 0.25
    r_denoising_config: Tuple[Tuple] = ((3, 0.15),)
    s_denoising: bool = True
    s_probability: float = 0.5
    x_denoising: bool = True
    x_probability: float = 0.25
    x_denoising_config: Tuple[Tuple] = ((32, 0.5, 0.5),)
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    label_pad_token_id: int = -100

    def __post_init__(self):
        self.total_task = [0, 1, 2]
        task_prob = []
        task_prob.append(self.r_probability if self.r_denoising else 0.0)
        task_prob.append(self.s_probability if self.s_denoising else 0.0)
        task_prob.append(self.x_probability if self.x_denoising else 0.0)
        self.task_prob = task_prob
        self.pad_token_id = self.tokenizer.pad_token_id
        self.decoder_start_token_id = self.tokenizer.bos_token_id

    def assign_task_type(self, batch_size: int):
        '''
            Randomly assign S,R,X to each sentence based on weighted prob
        '''
        return random.choices(self.total_task,weights=self.task_prob, k=batch_size)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        torch.set_printoptions(threshold=10_000)
        np.set_printoptions(threshold=10_000)
        if torch.rand(1) < -1 or not self.model.training:
            return default_data_collator(examples)

        # Handle dict or lists with proper padding and conversion to tensor.
        # print(examples)
        task_ids = self.assign_task_type(len(examples))
        task_type = torch.tensor(task_ids)
        lengths = torch.tensor([ len(e['input_ids']) for e in examples ], dtype=torch.long)
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer,
                    pad_to_multiple_of=self.pad_to_multiple_of)
            }
        max_length = batch['input_ids'].shape[-1]

        # new_batch = copy.deepcopy(batch)
        new_batch = {
            "input_ids": torch.zeros(batch['input_ids'].shape, dtype=torch.long),
            "labels": torch.zeros(batch['input_ids'].shape, dtype=torch.long),
            "attention_mask": torch.zeros(batch['input_ids'].shape, dtype=torch.long),
            "prefix_mask": torch.zeros(batch['input_ids'].shape, dtype=torch.long),
        }

        _, expanded_length = batch['input_ids'].shape
        input_ids = batch["input_ids"]
        r_denoising_idx = task_type == 0
        r_denoising_idx_num = torch.where(r_denoising_idx)[0]
        if r_denoising_idx.any():
            mask_indices = None
            sub_input_ids = input_ids[r_denoising_idx]
            # union of different denoising settings
            for (mean_span, noise) in self.r_denoising_config:
                _mask_indices = np.array([
                    random_spans_noise_mask(expanded_length, mean_span, noise) for _ in range(len(sub_input_ids))
                ])

                if mask_indices is None:
                    mask_indices = _mask_indices
                else:
                    mask_indices = mask_indices | _mask_indices

            valid_lengths = get_first_special_index_batch(sub_input_ids, self.pad_token_id)
            for idx, valid_len in enumerate(valid_lengths):
                mask_indices[idx, valid_len:] = False
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_mask = ~mask_indices
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
            _sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel)
            _labels = self.filter_input_ids(sub_input_ids, labels_sentinel)
            
            labels = []
            _input_ids = []
            for idx, _label in enumerate(_labels):
                label = _label[_label != self.pad_token_id]
                _sub_input_ids_idx = _sub_input_ids[idx][_sub_input_ids[idx] != self.pad_token_id]
                sub_input_len =  len(_sub_input_ids_idx)
                _sub_input_ids_idx = np.concatenate((_sub_input_ids_idx, label))
                label = np.concatenate(([self.label_pad_token_id] * sub_input_len, label))
                new_batch['attention_mask'][r_denoising_idx_num[idx]][:len(label)] = 1
                new_batch["prefix_mask"][r_denoising_idx_num[idx]][:sub_input_len] = 1
                if len(label) > max_length:
                    label = torch.from_numpy(label[: max_length])
                    _sub_input_ids_idx = torch.from_numpy(_sub_input_ids_idx[: max_length])
                else:
                    diff = max_length - len(label)
                    label = F.pad(torch.from_numpy(label), (0, diff), 'constant', self.label_pad_token_id)
                    _sub_input_ids_idx = F.pad(torch.from_numpy(_sub_input_ids_idx), (0, diff), 'constant', self.pad_token_id)
                labels.append(label)
                _input_ids.append(_sub_input_ids_idx)
            labels = torch.stack(labels)
            _input_ids = torch.stack(_input_ids)
            
            new_batch['input_ids'][r_denoising_idx] = _input_ids.long()
            new_batch['labels'][r_denoising_idx] = labels.long()

        s_denoising_idx = task_type == 1
        s_denoising_idx_num = torch.where(s_denoising_idx)[0]
        if s_denoising_idx.any():
            sub_input_ids = input_ids[s_denoising_idx]
            _labels = []
            _input_ids = []

            for idx, input_id in enumerate(sub_input_ids):
                valid_len = get_first_special_index(input_id, self.pad_token_id)
                split = max(valid_len//2, 2)
                new_batch["prefix_mask"][s_denoising_idx_num[idx]][:split] = 1

            # for input_id, len_ in zip(sub_input_ids, lengths[s_denoising_idx]):
            #     if self.tokenizer.padding_side == "left":
            #         idx = get_first_non_specical_index(input_id, self.pad_token_id)
            #         valid_len = len_ - idx - 1
            #         split = max(valid_len//2, 2) + idx
            #         diff = expanded_length - split
            #         _input_ids.append(F.pad(input_id[:split], (0, diff), 'constant', self.pad_token_id))
            #         past_seq = input_id[split:]
            #         if past_seq[-1] != self.tokenizer.eos_token_id:
            #             past_seq[-1] = self.tokenizer.eos_token_id
            #         # _labels.append(F.pad(past_seq, (split, 0), 'constant', self.pad_token_id))
            #     else:
            #         valid_len = get_first_special_index(input_id, self.pad_token_id)
            #         split = max(valid_len//2, 2)
            #         # diff = expanded_length - split
            #         # _input_ids.append(F.pad(input_id[:split], (0, diff), 'constant', self.pad_token_id))
            #         # past_seq = input_id[split:]
            #         # past_seq = torch.where(past_seq == self.pad_token_id, self.label_pad_token_id, past_seq)
            #         # _labels.append(F.pad(past_seq, (split, 0), 'constant', self.label_pad_token_id))

            new_batch['input_ids'][s_denoising_idx] = batch['input_ids'][s_denoising_idx]
            new_batch['labels'][s_denoising_idx] = batch['labels'][s_denoising_idx]
            new_batch['attention_mask'][s_denoising_idx] = batch['attention_mask'][s_denoising_idx]


        x_denoising_idx = task_type == 2
        x_denoising_idx_num = torch.where(x_denoising_idx)[0]
        if x_denoising_idx.any():
            sub_input_ids = input_ids[x_denoising_idx]
            mask_indices = []
            valid_lengths = get_first_special_index_batch(sub_input_ids, self.pad_token_id)
            for len_, valid_len in zip(lengths[x_denoising_idx], valid_lengths):
                mask_index = None
                # idx = get_first_non_specical_index(input_id, self.pad_token_id)
                # valid_len = len_ - idx - 1
                for (mean_span, noise, ratio) in self.x_denoising_config:
                    mean_span = min(mean_span, valid_len * ratio)
                    _mask_index = np.array(
                        random_spans_noise_mask(expanded_length, mean_span, noise)
                    )
                    if mask_index is None:
                        mask_index = _mask_index
                    else:
                        mask_index = mask_index | _mask_index
                mask_index[valid_len:] = False
                mask_indices.append(mask_index[np.newaxis,:])

            mask_indices = np.concatenate(mask_indices, axis=0)
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_mask = ~mask_indices
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
            _sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel)
            _labels = self.filter_input_ids(sub_input_ids, labels_sentinel)

            labels = []
            _input_ids = []
            for idx, _label in enumerate(_labels):
                label = _label[_label != self.pad_token_id]
                _sub_input_ids_idx = _sub_input_ids[idx][_sub_input_ids[idx] != self.pad_token_id]
                sub_input_len =  len(_sub_input_ids_idx)
                _sub_input_ids_idx = np.concatenate((_sub_input_ids_idx, label))
                label = np.concatenate(([self.label_pad_token_id] * sub_input_len, label))
                new_batch['attention_mask'][x_denoising_idx_num[idx]][:len(label)] = 1
                new_batch["prefix_mask"][x_denoising_idx_num[idx]][:sub_input_len] = 1
                if len(label) > max_length:
                    label = torch.from_numpy(label[: max_length])
                    _sub_input_ids_idx = torch.from_numpy(_sub_input_ids_idx[: max_length])
                else:
                    diff = max_length - len(label)
                    label = F.pad(torch.from_numpy(label), (0, diff), 'constant', self.label_pad_token_id)
                    _sub_input_ids_idx = F.pad(torch.from_numpy(_sub_input_ids_idx), (0, diff), 'constant', self.pad_token_id)
                labels.append(label)
                _input_ids.append(_sub_input_ids_idx)
            labels = torch.stack(labels)
            _input_ids = torch.stack(_input_ids)
            
            new_batch['input_ids'][x_denoising_idx] = _input_ids.long()
            new_batch['labels'][x_denoising_idx] = labels.long()

        # if torch.cuda.current_device() == 0:
        #     print(new_batch)
        # exit(0)
        ## Override labels
        # if "labels" in batch:
        #     new_batch["labels"] = batch["labels"]
        # new_batch["attention_mask"] = batch["attention_mask"]
        
        return new_batch 


    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = []
        for row in input_ids_full:
            collapsed_id = row[row >= 0]
            diff = len(row) - len(collapsed_id)
            collapsed_id = np.pad(collapsed_id, (0, diff), 'constant', constant_values=self.pad_token_id)
            input_ids.append(collapsed_id)
        return np.array(input_ids)

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids


    def prepare_decoder_inputs_from_labels(self, batch):
        # decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id.
        # See T5 docs for more information
        batch["labels"][ batch["labels"] == self.pad_token_id ] = self.label_pad_token_id
        shifted_labels = batch["labels"].new_zeros(batch["labels"].shape)
        shifted_labels[..., 1:] = batch["labels"][..., :-1].clone()
        shifted_labels[..., 0] = self.decoder_start_token_id  # decoder_start_token_id

        batch["decoder_input_ids"] = torch.masked_fill(
            shifted_labels,
            shifted_labels == self.label_pad_token_id,
            self.pad_token_id
        )
        batch["decoder_attention_mask"] = torch.where(
            shifted_labels == self.label_pad_token_id,
            0,
            torch.ones_like(shifted_labels),
        )
        return batch

    def np_prepare_decoder_inputs_from_labels(self, batch):
        batch["labels"][ batch["labels"] == self.pad_token_id ] = self.label_pad_token_id
        shifted_labels = np.zeros(batch["labels"].shape)
        shifted_labels[..., 1:] = batch["labels"][..., :-1].copy()
        shifted_labels[..., 0] = self.decoder_start_token_id

        batch["decoder_input_ids"] = np.where(
            shifted_labels == self.label_pad_token_id,
            self.pad_token_id,
            shifted_labels
        )
        batch["decoder_attention_mask"] = np.where(
            shifted_labels == self.label_pad_token_id,
            0,
            np.ones_like(shifted_labels)
        )
        return batch