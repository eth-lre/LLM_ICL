import os
import torch
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from scipy.sparse import hstack
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import data_loader, avg


def tfidf(args, data_list):
    best_acc = []
    train_word_set, test_word_set = set(), set()
    train_char_set, test_char_set = set(), set()
    for data in tqdm(data_list):
        # load data
        train_data = list(data.train_data)
        train_data_new = train_data[:args.icl_num]
        random.shuffle(train_data_new)
        test_data = list(data.test_data)
        y_test = [data.prompt_label.index(d["label"]) for d in test_data]
        l_set = list(set(y_test))
        # check training dataset
        if args.icl_num >= len(l_set):
            while len(set([d["label"] for d in train_data_new])) < len(l_set):
                train_data_new = random.sample(train_data, k=args.icl_num)
        train_data = train_data_new
        train_text = [d["text"] for d in train_data]
        test_text = [d["text"] for d in test_data]
        # all_text = train_text + test_text
        all_text = train_text

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        all_text_llama = []
        for t in all_text:
            tokenids = tokenizer.encode(t, add_special_tokens=False)
            new_t = " ".join([tokenizer.decode(wid) for wid in tokenids])
            all_text_llama.append(new_t)
        # all_text = all_text_llama

        y = [data.prompt_label.index(d["label"]) for d in train_data]

        for d in train_text:
            for w in d.split(" "):
                train_word_set.add(w)
                for c in list(w):
                    train_char_set.add(c)
        for d in test_text:
            for w in d.split(" "):
                test_word_set.add(w)
                for c in list(w):
                    test_char_set.add(c)
        #print(train_word_set, test_word_set, train_char_set, test_char_set,)
        #print(len(train_word_set.intersection(test_word_set))/len(test_word_set), (len(train_char_set.intersection(test_char_set)))/len(test_char_set))
        results = []
        for cs in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]:#: 10 for sst2
            for solver in ['lbfgs', 'saga', 'sag']:#["sag"]: all for sst2
                for ngc_max in [1,2,3,4]:#[1,2,3,4]: 3 for sst2
                    for ngw_max in [1,2]:#[1,2]: 1 for sst2
                        '''
                        if args.tfidf_emb:
                            # acc
                            word_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', sublinear_tf=True, strip_accents='unicode', 
                                                              stop_words='english', ngram_range=(1, 1), max_features=10000)
                            word_vectorizer.fit(train_text)
                            train_word_features = word_vectorizer.transform(train_text)
                            char_vectorizer = TfidfVectorizer(analyzer='char', sublinear_tf=True, strip_accents='unicode', 
                                                              stop_words='english', ngram_range=(1, 3), max_features=50000)
                            char_vectorizer.fit(train_text)
                            train_char_features = char_vectorizer.transform(train_text)
                            train_features = hstack([train_word_features, train_char_features])
                        else:
                            # acc
                            word_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', strip_accents='unicode', 
                                                              stop_words='english', ngram_range=(1, 1), max_features=10000)
                            word_vectorizer.fit(train_text)
                            train_word_features = word_vectorizer.transform(train_text)
                            char_vectorizer = CountVectorizer(analyzer='char', strip_accents='unicode', 
                                                              stop_words='english', ngram_range=(1, 3), max_features=50000)
                            char_vectorizer.fit(train_text)
                            train_char_features = char_vectorizer.transform(train_text)
                            train_features = hstack([train_word_features, train_char_features])

                        X_train, X_test, y_train, y_test = train_test_split(train_features, y, test_size=0.3, random_state=args.random_seed)
                        lr_model = LogisticRegression(random_state=args.random_seed)
                        param_dict = {'C': [0.001, 0.01, 0.1, 1, 10],
                                     'solver': ['sag', 'lbfgs', 'saga']}

                        start = time()
                        grid_search = GridSearchCV(lr_model, param_dict)
                        grid_search.fit(X_train, y_train)
                        print("GridSearch took %.2f seconds to complete." % (time()-start))
                        print("Cross-Validated Score of the Best Estimator: %.3f" % grid_search.best_score_)

                        lr=LogisticRegression(C=1, solver ='saga')
                        lr.fit(X_train, y_train)
                        lr_preds=lr.predict(X_test)

                        print(confusion_matrix(y_test, lr_preds))
                        print(classification_report(y_test, lr_preds))
                        print("Accuracy Score: %.3f" % accuracy_score(y_test, lr_preds))

                        '''
                        if args.icl_num < len(l_set):
                            final_preds = [random.choices(l_set, k=1)[0] for _ in y_test]
                        else:
                            if args.tfidf_emb:
                                word_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', sublinear_tf=True, strip_accents='unicode', 
                                                                  stop_words='english', ngram_range=(1, ngw_max), max_features=10000)
                                word_vectorizer.fit(all_text)
                                train_word_features = word_vectorizer.transform(train_text)
                                test_word_features = word_vectorizer.transform(test_text)
                                char_vectorizer = TfidfVectorizer(analyzer='char', sublinear_tf=True, strip_accents='unicode', 
                                                                  ngram_range=(1, ngc_max), max_features=50000)
                                char_vectorizer.fit(all_text)
                                train_char_features = char_vectorizer.transform(train_text)
                                test_char_features = char_vectorizer.transform(test_text)
                                train_features = hstack([train_char_features, train_word_features])
                                test_features = hstack([test_char_features, test_word_features])
                            else:
                                word_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', strip_accents='unicode', 
                                                                  stop_words='english', ngram_range=(1, ngw_max), max_features=10000)
                                word_vectorizer.fit(all_text)
                                train_word_features = word_vectorizer.transform(train_text)
                                test_word_features = word_vectorizer.transform(test_text)
                                #'''
                                char_vectorizer = CountVectorizer(analyzer='char', strip_accents='unicode', 
                                                                  ngram_range=(1, ngc_max), max_features=50000)
                                char_vectorizer.fit(all_text)
                                train_char_features = char_vectorizer.transform(train_text)
                                test_char_features = char_vectorizer.transform(test_text)
                                train_features = hstack([train_char_features, train_word_features])
                                #'''
                                # train_features = train_word_features
                                test_features = hstack([test_char_features, test_word_features])
                                # test_features = test_word_features
                            lr=LogisticRegression(C=cs, solver=solver, max_iter=10000)
                            lr.fit(train_features, y)
                            final_preds=lr.predict(test_features)

                        results.append(f1_score(y_test, final_preds, average='macro'))
        '''
        if args.icl_num < len(l_set):
            best_acc.append(avg(results))
        else:
            best_acc.append(max(results))
        '''
        best_acc.append(max(results))
        # print(max(results), results.index(max(results)))
        # print(results)
    print("Linear model (best estimator):", avg(best_acc))

    return
