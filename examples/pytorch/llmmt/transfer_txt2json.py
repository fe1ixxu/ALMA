import json
import argparse
import random

def detect_line_number(filename):
    with open(filename, encoding="utf-8") as f:
        count = 0
        line = f.readline()
        while line:
            count += 1
            line = f.readline()
    return count

def sample_txt_json(args):
    MAXNUM = detect_line_number(args.path + f"/{args.split}.{args.src}-{args.tgt}.{args.src}")
    print(MAXNUM)
    indices = list(range(MAXNUM))
    if args.sample_ratio <= 1:
        num_samples = int(MAXNUM * args.sample_ratio)
    else:
        num_samples = int(args.sample_ratio)
    sampled_indices = random.sample(indices, num_samples)
    sampled_indices = sorted(sampled_indices)
    print(sampled_indices[:10])
    samples = []
    with open(args.path + f"/{args.split}.{args.src}-{args.tgt}.{args.src}", encoding="utf-8") as f_src:
        with open(args.path + f"/{args.split}.{args.src}-{args.tgt}.{args.tgt}", encoding="utf-8") as f_tgt:
            src = f_src.readline()
            tgt = f_tgt.readline()
            cur_idx = 0
            while src and tgt:
                if cur_idx == sampled_indices[0]:
                    cur_sample = {
                       'translation':{
                        args.src: src.strip(),
                        args.tgt: tgt.strip()
                       } 
                    }
                    samples.append(cur_sample)
                    sampled_indices.pop(0)
                    if len(sampled_indices) == 0:
                        break
                cur_idx += 1
                src = f_src.readline()
                tgt = f_tgt.readline()
    suffix = "" if args.sample_ratio == 1 else "-" + str(num_samples)
    with open(args.path + f"/{args.split}.{args.src}-{args.tgt}{suffix}.json", 'w') as f_w:
        json.dump(samples, f_w, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--tgt", type=str, required=True)
    parser.add_argument("--sample_ratio", type=float, default=0.1)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    sample_txt_json(args)


if __name__ == "__main__":
    main()