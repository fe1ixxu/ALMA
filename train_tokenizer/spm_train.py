import sentencepiece as spm
import argparse
from transformers.convert_slow_tokenizer import import_protobuf
from transformers import LlamaTokenizer
import os
import shutil

def main(args):
    spm.SentencePieceTrainer.train(
        input=args.input
        model_prefix=args.model_prefix,
        model_type="bpe",
        vocab_size=args.vocab_size,
    )
    spm_folder = 'spm_models'
    path = spm_folder + '/' + args.model_prefix + '.model'
    if not os.path.exists(spm_folder):
        os.makedirs(spm_folder)
    shutil.move(args.model_prefix + '.model', spm_folder)

    llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_dir)
    llama_tokens = set(llama_tokenizer.get_vocab().keys())

    model_pb2 = import_protobuf()

    m = model_pb2.ModelProto()
    m.ParseFromString(open(path, 'rb').read())

    kept_pieces = m.pieces[:3]

    for p in m.pieces[3:]:
        if (
            not p.piece in llama_tokens and
            any(c.isalpha() for c in p.piece) and
            not p.piece.isnumeric()
        ):
            kept_pieces.append(p)

    print(f"Kept {len(kept_pieces)} pieces out of {len(m.pieces)}")

    kept_tokens = set([x.piece for x in kept_pieces])

    while i < len(m.pieces):
        if m.pieces[i].piece not in kept_tokens:
            m.pieces.pop(i)
        else:
            i += 1

    assert len(m.pieces) == len(kept_pieces)
    with open(path, 'wb') as f:
        f.write(m.SerializeToString())
    print(f"Model saved to {path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--model_prefix', type=str)
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--llama_dir', type=str)
    args = parser.parse_args()
    main(args)