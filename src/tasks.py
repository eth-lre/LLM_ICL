import numpy as np

from utils import fix_seed, data_loader
from tfidf import tfidf
from icl import icl_eval


def evaluations(args):
    fix_seed(args.random_seed)
    random_seeds = list(np.random.randint(1, 1e5, args.repeat_num))

    # prepare data
    data_list = []
    for rs in random_seeds:
        d = data_loader(args, random_seed=rs)
        data_list.append(d)
        del d
    if args.task_pattern in ["none", "token", "label"]:
        tfidf(args, data_list)
    icl_eval(args, data_list)

    return
