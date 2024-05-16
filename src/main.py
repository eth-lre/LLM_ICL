import os
import argparse

from utils import fix_seed
from tasks import evaluations


def main_func(args):
    evaluations(args)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge-Pattern Graph")
    # general settings
    parser.add_argument("--tmp_dir", type=str, default="./tmp/",
                        help="the cache directory")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--device", type=int, default=0,
                        help="which GPU to use: set -1 to use CPU/prompt")
    parser.add_argument("--model_path", type=str, default="Llama-2-7b-hf",
                        help="the base model path / model name")

    # icl settings
    parser.add_argument("--icl_num", type=int, default=32,
                        help="the number of examples for in-context learning prompt")
    parser.add_argument("--tfidf_emb", type=bool, default=False,
                        help="whether use LM emb instead of TF-IDF emb. If not, use BOW")

    # task settings
    parser.add_argument("--dataset", type=str, default="SetFit/sst2",
                        help="the task for evaluation: [SetFit/sst2, SetFit/sst5, SetFit/CR, ag_news, dbpedia_14]")
    parser.add_argument("--task_pattern", type=str, default="none",
                        help="whether shuffle the input text: [none, token, label, memo_label_xxx (rand, kw, syn, ant)]. none: original ICL, token: SUI, label: SUL, memo_label_xxx: true label -> kw/syn/ant/rand label.")
    # synthetic task settings
    parser.add_argument("--test_num", type=int, default=256,
                        help="the size of test dataset for synthetic task")
    parser.add_argument("--repeat_num", type=int, default=20,
                        help="the size of test dataset for synthetic task")

    args = parser.parse_args()
    print(args)
    fix_seed(args.random_seed)
    if not os.path.exists(args.tmp_dir):
        os.mkdir(args.tmp_dir)

    main_func(args)
