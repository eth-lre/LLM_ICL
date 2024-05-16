import os
import torch
import lorem
import random
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, LlamaConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

from utils import data_loader, avg, fix_seed


def prompt_construction(args, data, tokenizer, shuf_label_ids=[]):
    data_train = random.sample(data.train_data, k=args.icl_num)
    random.shuffle(data_train)
    shuffle_map_str = {}
    prompt = ""
    for d in data_train:
        x, y = d["text"], d["label"]
        prompt = prompt + data.prompt_x + x + data.prompt_y + y
    return prompt


def get_labels(args, tokenizer, data, shuffle_map):
    label_tokenids = []
    label_tokenids = tokenizer.encode(" ".join(data.prompt_label), add_special_tokens=False)
    return label_tokenids


def get_shufflemap(args, tokenizer, model, data):
    shuffle_map = {}
    if args.task_pattern == "token":
        tmp_ids = [i for i in range(tokenizer.vocab_size)]
        random.shuffle(tmp_ids)
        for i in range(len(tmp_ids)):
            shuffle_map[i] = tmp_ids[i]
        del tmp_ids
    # get label token ids
    label_tokenids = get_labels(args, tokenizer, data, shuffle_map)
    shuf_label_ids = []
    if False:
        labels = [l for l in data.prompt_label] + [" "]
        output_probs = []
        for l in labels:
            if l == " ": 
                d_text = " "
            else:
                d = random.choice(data.train_data)
                while d["label"] != l:
                    d = random.choice(data.train_data)
                d_text = d["text"]
            tmp_text = data.prompt_x + d_text + data.prompt_y
            tmp_inputs = tokenizer(tmp_text, return_tensors="pt", add_special_tokens=False).to(args.device)
            tmp_outputs = model(**tmp_inputs, labels=tmp_inputs["input_ids"]).logits.softmax(-1)[:, -1, :]
            output_probs.append(tmp_outputs)
        count = 0
        while count < 1:
            rand_text = lorem.text()
            rand_text = rand_text.replace(".", "")
            rand_text = rand_text.replace("\n", "")
            # rand_words = list(set(rand_text.split(" ")))
            rand_ids = tokenizer.encode(rand_text, add_special_tokens=False)
            rand_ids = list(set(rand_ids))
            shuf_label_ids = random.sample(rand_ids, k=len(label_tokenids))
            tmp_probs = [prob[:,shuf_label_ids] for prob in output_probs]
            # test if the label satisfies the constraints
            indicator = True
            for i in range(len(tmp_probs)):
                tmp_p = torch.squeeze(tmp_probs[i])
                if i == len(tmp_probs) -1:  # for empty sample sentence
                    if torch.max(tmp_p) / torch.min(tmp_p) > 2*len(label_tokenids):
                        continue
                        indicator = False
                    if torch.min(tmp_p) < 1e-8:
                        continue
                        indicator = False
                else:  # for labeled sample sentence
                    continue
                    tmp_idx = torch.argmax(tmp_p)
                    if tmp_idx != i:
                        indicator = False
            if indicator: 
                count = 1
                print(tmp_probs, tokenizer.decode(shuf_label_ids), [tokenizer.decode(i) for i in shuf_label_ids])
            del tmp_probs
    return shuffle_map, label_tokenids, shuf_label_ids


def icl_eval(args, data_list):
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if "scracth" in args.model_path:
        configuration = LlamaConfig(torch_dtype=torch.float16)
        model = LlamaForCausalLM(configuration).to(args.device)
    elif "70B" in args.model_path:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = load_checkpoint_and_dispatch(model, args.model_path,
                                             device_map='auto',
                                             offload_folder="offload",
                                             offload_state_dict=True,
                                             dtype = "float16",
                                             no_split_module_classes=["LlamaDecoderLayer"])
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(args.device)

    with torch.no_grad():
        if args.task_pattern == "perfect_none" or args.task_pattern == "perfect_label":
            data_list = get_perfect_set(args, tokenizer, model, data_list)

    # test with different random seeds
    f1_list = []
    rank_list = []
    with tqdm(total=len(data_list)) as t:
        # tmp_cm = None
        for data in data_list:
            shuffle_map, label_tokenids, shuf_label_ids = get_shufflemap(args, tokenizer, model, data)
            tmp_pred_list = []
            tmp_gt_list = []
            # memo_label
            if "memo_label" in args.task_pattern:
                new_data = data
                new_train_data, new_test_data = [], []
                for d in data.train_data:
                    x, y = d["text"], d["label"]
                    tmp_idx = data.prompt_label.index(y)
                    new_x = data.prompt_label_org[tmp_idx]
                    d["text"] = new_x
                    new_train_data.append(d)
                for d in data.test_data:
                    x, y = d["text"], d["label"]
                    tmp_idx = data.prompt_label.index(y)
                    new_x = data.prompt_label_org[tmp_idx]
                    d["text"] = new_x
                    new_test_data.append(d)
                data.train_data = new_train_data
                data.test_data = new_test_data
            print(data.test_data[0])

            # prompting
            with torch.no_grad():
                for d in tqdm(data.test_data):
                    x, y = d["text"], d["label"]

                    icl_prompt = prompt_construction(args, data, tokenizer, shuf_label_ids)
                    tmp_input = icl_prompt + data.prompt_x + x + data.prompt_y
                    # inference
                    inputs = tokenizer(tmp_input, return_tensors="pt", add_special_tokens=False)
                    inputs["input_ids"] = inputs["input_ids"].to(args.device)
                    # shuffle input / label
                    if args.task_pattern == "token":
                        shuf_ids = tokenizer.encode(tmp_input, add_special_tokens=False)
                        for i in range(len(shuf_ids)):
                            if args.task_pattern == "token" and shuf_ids[i] in label_tokenids: continue
                            if shuf_ids[i] in [13957, 29901, 29871, 13, 28048, 2073, 1134]: continue  # [10567, 13, 21099, 2463]
                            shuf_ids[i] = shuffle_map[shuf_ids[i]]
                        inputs["input_ids"] = torch.tensor([shuf_ids]).to(args.device)
                    outputs = model(**inputs, labels=inputs["input_ids"]).logits.softmax(-1)[:, -1, :][:,label_tokenids]
                    tmp_pred = int(torch.argmax(outputs))
                    outputs_prob = [float(outputs[0, i]) for i in range(outputs.shape[1])]
                    outputs_lidx = data.prompt_label.index(y)
                    tmp_rank = sorted(outputs_prob, reverse=True).index(outputs_prob[outputs_lidx]) + 1
                    rank_list.append(tmp_rank)
                    tmp_gt_list.append(data.prompt_label.index(y))
                    tmp_pred_list.append(tmp_pred)
                tmp_f1_macro = f1_score(tmp_gt_list, tmp_pred_list, average='macro')
                # tmp_f1_macro = accuracy_score(tmp_gt_list, tmp_pred_list)
                t.set_postfix(acc=tmp_f1_macro)
                t.update(1)
            f1_list.append(tmp_f1_macro)
            print(data.prompt_label, f1_list[-1])
    print("The ICL f1-macro score is: ", avg(f1_list), ": ", f1_list)
    print("The rank for positive examples is: ", avg(rank_list))
    return
