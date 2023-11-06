from dataclasses import dataclass
from typing import Dict
import numpy as np
import logging
from tqdm.auto import tqdm
import math
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path
import fasttext
import tempfile
from datasets import load_dataset
import nltk
from functools import partial
import multiprocessing
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec

from wechsel.download_utils import download, gunzip

__version__ = "0.0.4"

CACHE_DIR = (
    (Path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "wechsel").expanduser().resolve()
)


def softmax(x, axis=-1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


class WordEmbedding:
    """
    Uniform interface to fastText models and gensim Word2Vec models.
    """

    def __init__(self, model):
        self.model = model

        if isinstance(model, fasttext.FastText._FastText):
            self.kind = "fasttext"
        elif isinstance(model, Word2Vec):
            self.kind = "word2vec"
        else:
            raise ValueError(
                f"{model} seems to be neither a fastText nor Word2Vec model."
            )

    def has_subword_info(self):
        return self.kind == "fasttext"

    def get_words_and_freqs(self):
        if self.kind == "fasttext":
            return self.model.get_words(include_freq=True, on_unicode_error="ignore")
        elif self.kind == "word2vec":
            return (self.model.wv.index_to_key, self.model.wv.expandos["count"])

    def get_dimension(self):
        if self.kind == "fasttext":
            return self.model.get_dimension()
        elif self.kind == "word2vec":
            return self.model.wv.vector_size

    def get_word_vector(self, word):
        if self.kind == "fasttext":
            return self.model.get_word_vector(word)
        elif self.kind == "word2vec":
            return self.model.wv[word]

    def get_word_id(self, word):
        if self.kind == "fasttext":
            return self.model.get_word_id(word)
        elif self.kind == "word2vec":
            return self.model.wv.key_to_index.get(word, -1)


def get_subword_embeddings_in_word_embedding_space(
    tokenizer, model, max_n_word_vectors=None, use_subword_info=True, verbose=True
):
    words, freqs = model.get_words_and_freqs()

    if max_n_word_vectors is None:
        max_n_word_vectors = len(words)

    sources = {}
    embs_matrix = np.zeros((len(tokenizer), model.get_dimension()))

    if use_subword_info:
        if not model.has_subword_info():
            raise ValueError("Can not use subword info of model without subword info!")

        for i in range(len(tokenizer)):
            token = tokenizer.decode(i).strip()

            # `get_word_vector` returns zeros if not able to decompose
            embs_matrix[i] = model.get_word_vector(token)
    else:
        embs = {value: [] for value in tokenizer.get_vocab().values()}

        for i, word in tqdm(
            enumerate(words[:max_n_word_vectors]),
            total=max_n_word_vectors,
            disable=not verbose,
        ):
            for tokenized in [
                tokenizer.encode(word, add_special_tokens=False),
                tokenizer.encode(" " + word, add_special_tokens=False),
            ]:
                for token_id in set(tokenized):
                    embs[token_id].append(i)

        for i in range(len(embs_matrix)):
            if len(embs[i]) == 0:
                continue

            weight = np.array([freqs[idx] for idx in embs[i]])
            weight = weight / weight.sum()

            vectors = [model.get_word_vector(words[idx]) for idx in embs[i]]

            sources[tokenizer.convert_ids_to_tokens([i])[0]] = embs[i]
            embs_matrix[i] = (np.stack(vectors) * weight[:, np.newaxis]).sum(axis=0)

    return embs_matrix, sources


def train_embeddings(
    text_path: str,
    language=None,
    tokenize_fn=None,
    encoding=None,
    epochs=20,
    **kwargs,
):
    """
    Utility function to train fastText embeddings.

    Args:
        text_path: path to a plaintext file to train on.
        language: language to use for Punkt tokenizer.
        tokenize_fn: function to tokenize the text (instead of using the Punkt tokenizer).
        encoding: file encoding.
        epochs: number of epochs to train for.
        kwargs: extra args to pass to `fasttext.train_unsupervised`.

    Returns:
        A fasttext model trained on text from the file.
    """
    if tokenize_fn is None:
        if language is None:
            raise ValueError(
                "`language` must not be `None` if no `tokenize_fn` is passed!"
            )

        tokenize_fn = partial(nltk.word_tokenize, language=language)

    if text_path.endswith(".txt"):
        dataset = load_dataset("text", data_files=text_path, split="train")
    if text_path.endswith(".json") or text_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=text_path, split="train")

    dataset = dataset.map(
        lambda row: {"text": " ".join(tokenize_fn(row["text"]))},
        num_proc=multiprocessing.cpu_count(),
    )

    out_file = tempfile.NamedTemporaryFile("w+")
    for text in dataset["text"]:
        out_file.write(text + "\n")

    return fasttext.train_unsupervised(
        out_file.name,
        dim=100,
        neg=10,
        model="cbow",
        minn=5,
        maxn=5,
        epoch=epochs,
        **kwargs,
    )


def load_embeddings(identifier: str, verbose=True):
    """
    Utility function to download and cache embeddings from https://fasttext.cc.

    Args:
        identifier: 2-letter language code.

    Returns:
        fastText model loaded from https://fasttext.cc/docs/en/crawl-vectors.html.
    """
    if os.path.exists(identifier):
        path = Path(identifier)
    else:
        logging.info(
            f"Identifier '{identifier}' does not seem to be a path (file does not exist). Interpreting as language code."
        )

        path = CACHE_DIR / f"cc.{identifier}.300.bin"

        if not path.exists():
            path = download(
                f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{identifier}.300.bin.gz",
                CACHE_DIR / f"cc.{identifier}.300.bin.gz",
                verbose=verbose,
            )
            path = gunzip(path)

    return fasttext.load_model(str(path))


def create_target_embeddings(
    source_subword_embeddings,
    target_subword_embeddings,
    source_tokenizer,
    target_tokenizer,
    source_matrix,
    neighbors=10,
    temperature=0.1,
    verbose=True,
):
    def get_n_closest(token_id, similarities, top_k):
        if (target_subword_embeddings[token_id] == 0).all():
            return None

        best_indices = np.argpartition(similarities, -top_k)[-top_k:]
        best_tokens = source_tokenizer.convert_ids_to_tokens(best_indices)

        best = sorted(
            [
                (token, similarities[idx])
                for token, idx in zip(best_tokens, best_indices)
            ],
            key=lambda x: -x[1],
        )

        return best

    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()

    target_matrix = np.zeros(
        (len(target_tokenizer), source_matrix.shape[1]), dtype=source_matrix.dtype
    )

    mean, std = (
        source_matrix.mean(0),
        source_matrix.std(0),
    )

    random_fallback_matrix = np.random.RandomState(1234).normal(
        mean, std, (len(target_vocab), source_matrix.shape[1])
    )

    batch_size = 1024
    n_matched = 0

    not_found = []
    sources = {}

    for i in tqdm(
        range(int(math.ceil(len(target_matrix) / batch_size))), disable=not verbose
    ):
        start, end = (
            i * batch_size,
            min((i + 1) * batch_size, len(target_matrix)),
        )

        similarities = cosine_similarity(
            target_subword_embeddings[start:end], source_subword_embeddings
        )
        for token_id in range(start, end):
            closest = get_n_closest(token_id, similarities[token_id - start], neighbors)

            if closest is not None:
                tokens, sims = zip(*closest)
                weights = softmax(np.array(sims) / temperature, 0)

                sources[target_tokenizer.convert_ids_to_tokens(token_id)] = (
                    tokens,
                    weights,
                    sims,
                )

                emb = np.zeros(target_matrix.shape[1])

                for i, close_token in enumerate(tokens):
                    emb += source_matrix[source_vocab[close_token]] * weights[i]

                target_matrix[token_id] = emb

                n_matched += 1
            else:
                target_matrix[token_id] = random_fallback_matrix[token_id]
                not_found.append(target_tokenizer.convert_ids_to_tokens([token_id])[0])

    for token in source_tokenizer.special_tokens_map.values():
        if isinstance(token, str):
            token = [token]

        for t in token:
            if t in target_vocab:
                target_matrix[target_vocab[t]] = source_matrix[
                    source_vocab[t]
                ]

    logging.info(
        f"Matching token found for {n_matched} of {len(target_matrix)} tokens."
    )
    return target_matrix, not_found, sources


@dataclass
class WECHSELInfo:
    source_subword_sources: Dict
    target_subword_sources: Dict
    sources: Dict
    not_found: Dict


class WECHSEL:
    def _compute_align_matrix_from_dictionary(
        self, source_embeddings, target_embeddings, dictionary
    ):
        correspondences = []

        for source_word, target_word in dictionary:
            for src_w in (source_word, source_word.lower(), source_word.title()):
                for trg_w in (target_word, target_word.lower(), target_word.title()):
                    src_id = source_embeddings.get_word_id(src_w)
                    trg_id = target_embeddings.get_word_id(trg_w)

                    if src_id != -1 and trg_id != -1:
                        correspondences.append(
                            [
                                source_embeddings.get_word_vector(src_w),
                                target_embeddings.get_word_vector(trg_w),
                            ]
                        )

        correspondences = np.array(correspondences)

        align_matrix, _ = orthogonal_procrustes(
            correspondences[:, 0], correspondences[:, 1]
        )

        return align_matrix

    def __init__(
        self,
        source_embeddings,
        target_embeddings,
        align_strategy="bilingual_dictionary",
        bilingual_dictionary=None,
    ):
        """
        Args:
            source_embeddings: fastText model or gensim Word2Vec model in the source language.

            target_embeddings: fastText model or gensim Word2Vec model in the source language.
            align_strategy: either of "bilingual_dictionary" or `None`.
                - If `None`, embeddings are treated as already aligned.
                - If "bilingual dictionary", a bilingual dictionary must be passed
                    which will be used to align the embeddings using the Orthogonal Procrustes method.
            bilingual_dictionary: path to a bilingual dictionary. The dictionary must be of the form
                ```
                english_word1 \t target_word1\n
                english_word2 \t target_word2\n
                ...
                english_wordn \t target_wordn\n
                ```
                alternatively, pass only the language name, e.g. "german", to use a bilingual dictionary
                stored as part of WECHSEL (https://github.com/CPJKU/wechsel/tree/main/dicts).
        """
        source_embeddings = WordEmbedding(source_embeddings)
        target_embeddings = WordEmbedding(target_embeddings)

        min_dim = min(
            source_embeddings.get_dimension(), target_embeddings.get_dimension()
        )
        if source_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(source_embeddings.model, min_dim)
        if target_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(source_embeddings.model, min_dim)

        if align_strategy == "bilingual_dictionary":
            if bilingual_dictionary is None:
                raise ValueError(
                    "`bilingual_dictionary` must not be `None` if `align_strategy` is 'bilingual_dictionary'."
                )

            if not os.path.exists(bilingual_dictionary):
                bilingual_dictionary = download(
                    f"https://raw.githubusercontent.com/CPJKU/wechsel/main/dicts/data/{bilingual_dictionary}.txt",
                    CACHE_DIR / f"{bilingual_dictionary}.txt",
                )

            dictionary = []

            for line in open(bilingual_dictionary):
                line = line.strip()
                try:
                    source_word, target_word = line.split("\t")
                except ValueError:
                    source_word, target_word = line.split()

                dictionary.append((source_word, target_word))

            align_matrix = self._compute_align_matrix_from_dictionary(
                source_embeddings, target_embeddings, dictionary
            )
            self.source_transform = lambda matrix: matrix @ align_matrix
            self.target_transform = lambda x: x
        elif align_strategy is None:
            self.source_transform = lambda x: x
            self.target_transform = lambda x: x
        else:
            raise ValueError(f"Unknown align strategy: {align_strategy}.")

        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings

    def apply(
        self,
        source_tokenizer,
        target_tokenizer,
        source_matrix,
        use_subword_info=True,
        max_n_word_vectors=None,
        neighbors=10,
        temperature=0.1,
    ):
        """
        Applies WECHSEL to initialize an embedding matrix.

        Args:
            source_tokenizer: T^s, the tokenizer in the source language.
            target_tokenizer: T^t, the tokenizer in the target language.
            source_matrix: E^s, the embeddings in the source language.
            use_subword_info: Whether to use fastText subword information. Default true.
            max_n_word_vectors: Maximum number of vectors to consider (only relevant if `use_subword_info` is False).

        Returns:
            target_matrix: The embedding matrix for the target tokenizer.
            info: Additional info about word sources, etc.
        """
        (
            source_subword_embeddings,
            source_subword_sources,
        ) = get_subword_embeddings_in_word_embedding_space(
            source_tokenizer,
            self.source_embeddings,
            use_subword_info=use_subword_info,
            max_n_word_vectors=max_n_word_vectors,
        )
        (
            target_subword_embeddings,
            target_subword_sources,
        ) = get_subword_embeddings_in_word_embedding_space(
            target_tokenizer,
            self.target_embeddings,
            use_subword_info=use_subword_info,
            max_n_word_vectors=max_n_word_vectors,
        )

        # align
        source_subword_embeddings = self.source_transform(source_subword_embeddings)
        target_subword_embeddings = self.target_transform(target_subword_embeddings)

        source_subword_embeddings /= (
            np.linalg.norm(source_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )
        target_subword_embeddings /= (
            np.linalg.norm(target_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )

        target_matrix, not_found, sources = create_target_embeddings(
            source_subword_embeddings,
            target_subword_embeddings,
            source_tokenizer,
            target_tokenizer,
            source_matrix.copy(),
            neighbors=neighbors,
            temperature=temperature,
        )

        return target_matrix, WECHSELInfo(
            source_subword_sources=source_subword_sources,
            target_subword_sources=target_subword_sources,
            sources=sources,
            not_found=not_found,
        )