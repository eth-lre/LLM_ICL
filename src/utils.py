import os
import random
import json
import numpy as np
import torch
import lorem
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Transformers
    set_seed(seed)
    return


def avg(l):
    if len(l):
        return sum(l)/len(l)
    else:
        return 0


class data_loader():
    def __init__(self, args, random_seed=-1):
        if random_seed == -1:
            fix_seed(args.random_seed)
        else:
            fix_seed(random_seed)
        self.prompt_label = self.get_label(args)
        self.train_data, self.dev_data, self.test_data = self.get_data(args)
        self.dev_idx, self.test_idx = self.get_idx(args)
        self.prompt_x, self.prompt_y = self.get_prompt(args)

    def get_idx(self, args):
        dev_idx, test_idx = [], []
        idx_len = int(len(self.test_data)/2)
        test_str = set([self.test_data[i]["text"] for i in range(idx_len)])
        for i in range(len(self.test_data)):
            if i < idx_len:
                test_idx.append(i)
            else:
                if self.test_data[i]["text"] not in test_str:
                    dev_idx.append(i)
        return dev_idx, test_idx

    def get_prompt(self, args):
        if "sst" in args.dataset:
            prompt_x = "\nReview: "
            prompt_y = "\nSentiment: "
        elif "CR" in args.dataset:
            prompt_x = "\nReview: "
            prompt_y = "\nSentiment: "
        elif "ag" in args.dataset:
            prompt_x = "\nNews: "
            prompt_y = "\nNews type: "
        elif "dbpedia" in args.dataset:
            prompt_x = "\nArticle: "
            prompt_y = "\nArticle type: "
        else:  # undefined task
            prompt_x = ""
            prompt_y = ""
        return prompt_x, prompt_y

    def get_label(self, args):
        if "sst2" in args.dataset:
            prompt_label = ["negative", "positive"]
            if "label" in args.task_pattern:
                prompt_label = ["Ne", "por"]
                # prompt_label = ["bar", "foo"]
                self.prompt_label_org = ["negative", "positive"]

                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                map_tokenids = []
                while len(map_tokenids) < len(prompt_label):
                    map_tokenid = random.choice([i for i in range(tokenizer.vocab_size)])
                    if len(tokenizer.encode(tokenizer.decode(map_tokenid), add_special_tokens=False)) == 1:
                        map_tokenids.append(map_tokenid)
                map_tokens = [tokenizer.decode(mtid,  add_special_tokens=False) for mtid in map_tokenids]

                keyword_labels = [["poor", "terrible", "worse", "bad", "awful", "fault", "inferior", "problem", "regret", "disappoint"],
                                ["excellent", "perfect", "best", "satisfied", "loved", "favorite", "awesome", "superior", "pleasant", "recommend"],]
                antonym_labels = [["good", "bright", "happy", "cheer", "benefit", "fortune", "helpful", "joy", "help", "favorite"], 
                                ["bad", "dark", "dire", "sorrow", "harm", "down", "dim", "bitter", "sad", "blue"],]
                synonym_labels = [["bad", "dark", "dire", "sorrow", "harm", "down", "dim", "bitter", "sad", "blue"], 
                                ["good", "bright", "happy", "cheer", "benefit", "fortune", "helpful", "joy", "help", "favorite"]]
                new_prompt_label = []
                if "kw" in args.task_pattern:
                    all_labels = keyword_labels
                elif "syn" in args.task_pattern:
                    all_labels = synonym_labels
                elif "ant" in args.task_pattern:
                    all_labels = antonym_labels
                else:
                    all_labels = [[p] for p in prompt_label]
                while len(prompt_label) != len(new_prompt_label):
                    new_prompt_label = [random.choice(all_l) for all_l in all_labels]
                prompt_label = new_prompt_label
                if "rand" in args.task_pattern:
                    prompt_label = map_tokens
        elif "sst3" in args.dataset:
            prompt_label = ["negative", "natural", "positive"]
            if "label" in args.task_pattern:
                prompt_label = ["Ne", "tam", "por"]
                self.prompt_label_org = ["negative", "natural", "positive"]

                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                map_tokenids = []
                while len(map_tokenids) < len(prompt_label):
                    map_tokenid = random.choice([i for i in range(tokenizer.vocab_size)])
                    if len(tokenizer.encode(tokenizer.decode(map_tokenid), add_special_tokens=False)) == 1:
                        map_tokenids.append(map_tokenid)
                map_tokens = [tokenizer.decode(mtid,  add_special_tokens=False) for mtid in map_tokenids]
                prompt_label = map_tokens
        elif "sst5" in args.dataset:
            prompt_label = ["terrible", "bad", "okay", "good", "great"]
            if "label" in args.task_pattern:
                prompt_label = ["Ne", "Vol", "tam", "Mag", "por"]
                self.prompt_label_org = ["terrible", "bad", "okay", "good", "great"]

                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                map_tokenids = []
                while len(map_tokenids) < len(prompt_label):
                    map_tokenid = random.choice([i for i in range(tokenizer.vocab_size)])
                    if len(tokenizer.encode(tokenizer.decode(map_tokenid), add_special_tokens=False)) == 1:
                        map_tokenids.append(map_tokenid)
                map_tokens = [tokenizer.decode(mtid,  add_special_tokens=False) for mtid in map_tokenids]
                prompt_label = map_tokens
        elif "CR" in args.dataset:
            prompt_label = ["negative", "positive"]
            if "label" in args.task_pattern:
                # prompt_label = ["bar", "foo"]
                prompt_label = ["Ne", "por"]
                self.prompt_label_org = ["negative", "positive"]

                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                map_tokenids = []
                while len(map_tokenids) < len(prompt_label):
                    map_tokenid = random.choice([i for i in range(tokenizer.vocab_size)])
                    if len(tokenizer.encode(tokenizer.decode(map_tokenid), add_special_tokens=False)) == 1:
                        map_tokenids.append(map_tokenid)
                map_tokens = [tokenizer.decode(mtid,  add_special_tokens=False) for mtid in map_tokenids]
                prompt_label = map_tokens

                keyword_labels = [["poor", "terrible", "worse", "bad", "awful", "fault", "inferior", "problem", "regret", "disappoint"],
                                ["excellent", "perfect", "best", "satisfied", "loved", "favorite", "awesome", "superior", "pleasant", "recommend"],]
                antonym_labels = [["good", "bright", "happy", "cheer", "benefit", "fortune", "helpful", "joy", "help", "favorite"], 
                                ["bad", "dark", "dire", "sorrow", "harm", "down", "dim", "bitter", "sad", "blue"],]
                synonym_labels = [["bad", "dark", "dire", "sorrow", "harm", "down", "dim", "bitter", "sad", "blue"], 
                                ["good", "bright", "happy", "cheer", "benefit", "fortune", "helpful", "joy", "help", "favorite"]]
                new_prompt_label = []
                if "kw" in args.task_pattern:
                    all_labels = keyword_labels
                elif "syn" in args.task_pattern:
                    all_labels = synonym_labels
                elif "ant" in args.task_pattern:
                    all_labels = antonym_labels
                else:
                    all_labels = [[p] for p in prompt_label]
                while len(prompt_label) != len(new_prompt_label):
                    new_prompt_label = [random.choice(all_l) for all_l in all_labels]
                prompt_label = new_prompt_label
                if "rand" in args.task_pattern:
                    prompt_label = map_tokens
        elif "ag" in args.dataset:
            prompt_label = ["world", "sports", "business", "science"]
            if "label" in args.task_pattern:
                prompt_label = ["Mag", "Am", "Num", "Lab"]
                self.prompt_label_org = ["world", "sports", "business", "science"]
                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                map_tokenids = []
                while len(map_tokenids) < len(prompt_label):
                    map_tokenid = random.choice([i for i in range(tokenizer.vocab_size)])
                    if len(tokenizer.encode(tokenizer.decode(map_tokenid), add_special_tokens=False)) == 1:
                        map_tokenids.append(map_tokenid)
                map_tokens = [tokenizer.decode(mtid,  add_special_tokens=False) for mtid in map_tokenids]
                prompt_label = map_tokens

                keyword_labels = [["conflict", "election", "global", "crisis", "politics", "war", "peace", "regime", "protest", "climate"], 
                                ["win", "team", "game", "player", "score", "season", "championship", "coach", "injury", "fans"], 
                                ["market", "company", "stock", "profit", "loss", "growth", "trade", "startup", "shares", "industry"], 
                                ["data", "technology", "species", "experiment", "research", "discovery", "theory", "study", "energy", "physics"]]
                new_prompt_label = []
                if "kw" in args.task_pattern:
                    all_labels = keyword_labels
                elif "syn" in args.task_pattern:
                    all_labels = synonym_labels
                elif "ant" in args.task_pattern:
                    all_labels = antonym_labels
                else:
                    all_labels = [[p] for p in prompt_label]
                while len(prompt_label) != len(new_prompt_label):
                    new_prompt_label = [random.choice(all_l) for all_l in all_labels]
                prompt_label = new_prompt_label
                if "rand" in args.task_pattern:
                    prompt_label = map_tokens
        elif "dbpedia" in args.dataset:
            prompt_label = ["company", "school", "artist", "player", "politics", "transport", \
                            "building", "nature", "village", "animal", "plant", "album", "film", "book"]
            if "label" in args.task_pattern:
                prompt_label = ["qu", "dol", "Mag", "ii", "us", "sit", \
                            "un", "non", "sed", "up", "tam", "Vol", "Am", "Nu"]
                self.prompt_label_org = ["company", "school", "artist", "player", "politics", "transport", \
                            "building", "nature", "village", "animal", "plant", "album", "film", "book"]

                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                map_tokenids = []
                while len(map_tokenids) < len(prompt_label):
                    map_tokenid = random.choice([i for i in range(tokenizer.vocab_size)])
                    if len(tokenizer.encode(tokenizer.decode(map_tokenid), add_special_tokens=False)) == 1:
                        map_tokenids.append(map_tokenid)
                map_tokens = [tokenizer.decode(mtid,  add_special_tokens=False) for mtid in map_tokenids]
                prompt_label = map_tokens
                keyword_labels = [["business", "industry", "market", "employee", "product", "service", "global", "shares", "headquarters", "management"],
                                ["education", "students", "teachers", "campus", "classes", "degree", "sports", "research", "library", "principal"],
                                ["gallery", "style", "canvas", "commission", "portrait", "modern", "installation", "studio", "critique", "museum"],
                                ["sports", "team", "games", "coach", "championship", "training", "record", "league", "medal", "skills"],
                                ["government", "policy", "election", "party", "candidate", "debate", "law", "president", "campaign", "vote"],
                                ["vehicle", "traffic", "rail", "road", "air", "network", "fuel", "safety", "schedule", "port"],
                                ["architecture", "construction", "design", "foundation", "commercial", "material", "engineer", "planning", "urban", "space"],
                                ["environment", "forest", "conservation", "species", "habitat", "climate", "ocean", "mountains", "river", "park"],
                                ["rural", "community", "population", "tradition", "local", "settlement", "craft", "culture", "families", "homes"],
                                ["species", "habitat", "conservation", "bird", "fish", "behavior", "domestic", "wild", "zoo", "migration"],
                                ["native", "growth", "flower", "tree", "leaf", "seed", "garden", "conservation", "soil", "root"],
                                ["music", "tracks", "artist", "release", "genre", "recording", "label", "chart", "singles", "lyrics"],
                                ["movie", "director", "actor", "script", "genre", "cinema", "scene", "audience", "production", "cast"],
                                ["author", "novel", "pages", "chapter", "plot", "characters", "edition", "cover", "series", "review"],]
                new_prompt_label = []
                if "kw" in args.task_pattern:
                    all_labels = keyword_labels
                else:
                    all_labels = [[p] for p in prompt_label]
                while len(prompt_label) != len(new_prompt_label):
                    new_prompt_label = [random.choice(all_l) for all_l in all_labels]
                prompt_label = new_prompt_label
                if "rand" in args.task_pattern:
                    prompt_label = map_tokens
        else:  # undefined task
            prompt_label = []
        print(prompt_label)
        return prompt_label

    def get_data(self, args):
        train_data, dev_data, test_data = [], [], []
        dev_data = []
        if "sst3" in args.dataset:
            tmp_tn = args.dataset.replace("sst3", "sst5")
        else:
            tmp_tn = args.dataset
        train_data = list(load_dataset(tmp_tn, split="train"))
        test_data = list(load_dataset(tmp_tn, split="test"))
        train_data = random.sample(train_data, k=2*args.test_num)
        test_data = random.sample(test_data, k=args.test_num)
        # change label
        new_train, new_test = [], []
        for d in train_data + test_data:
            new_d = d
            if "sst3" in args.dataset:
                tmp_li = int(d["label"]/1.5)
            else:
                tmp_li = int(d["label"])
            new_d["label"] = self.prompt_label[tmp_li]
            if "text" not in new_d:
                if "content" in new_d:  # DBpedia
                    new_d["text"] = new_d["content"]
            if len(new_train) < len(train_data):
                new_train.append(new_d)
            else:
                new_test.append(new_d)
        train_data = new_train
        test_data = new_test
        return train_data, dev_data, test_data
