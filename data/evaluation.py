import argparse
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
from config.conf import CONFIG
import numpy as np
import json
from rouge import Rouge
from nltk import PorterStemmer
import argparse

stemmer = PorterStemmer()

baselines = ["single", "ppl", "gltr"]
splits = ["val", "test"]


def rouge_calculation(hypotheses, references):
    assert (len(hypotheses) == len(references))
    hypoth = [" ".join([stemmer.stem(i) for i in line.split()]) for line in hypotheses]
    max_scores = []
    for i, hyp in enumerate(hypoth):
        refs = [" ".join([stemmer.stem(i) for i in line.split()]) for line in references[i]]
        hyps = [hyp] * len(refs)
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=False)
        scores_sorted = sorted(scores, key=lambda kv: kv["rouge-l"]["f"], reverse=True)
        # print("#" * 8)
        # print(hyps)
        # print(refs)
        # print(scores)
        # print(f"{len(scores)}")
        # print(scores_sorted)
        # break
        max_scores.append(scores_sorted[0])
    return max_scores


def prepare_args():
    parser = argparse.ArgumentParser(description='Save ChatGPT QA data into mongoDB')
    parser.add_argument('--data_dir', help='Where to load', default='/data/tsq/CK/data')
    parser.add_argument('--source_type', help='open or api', default='open',
                        choices=['open', 'api'])
    parser.add_argument('--time', help='When is the chat', default='before0301')
    parser.add_argument('--language', help='en/zh', default='en',
                        choices=['en', 'zh'])
    parser.add_argument('--source_dataset', help='Which dataset', default='HC3')
    parser.add_argument('--file_name', help='Which dataset', default='HC3_en.jsonl')
    # json
    parser.add_argument('--times', help='For Changing', type=str, nargs='+', default=
    "2023-01-18"
                        # "all"
                        )
    parser.add_argument('--pp_suffixes', help='For Changing', type=str, nargs='+', default=
    ["base"]
                        # ["base", "para", "prompt", "prompt_para"]
                        # [""]
                        )
    # detection train
    parser.add_argument('--lgb_dir', type=str, default='/data/tsq/CK/model/lgb', )
    parser.add_argument('--train_time_setting', type=str, default='only01',
                        choices=['only01', '01to03', ])
    parser.add_argument('--train_feature_settings', help='For Changing features',
                        type=str, nargs='+', default=['bottom10_single', 'random10_single', 'single', 'ppl', 'gltr'])
    parser.add_argument('--seeds', help='For Changing trails', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    # task
    parser.add_argument('--start_id', help='start id', type=int, default=0)
    parser.add_argument('--end_id', help='end id', type=int, default=1000)
    parser.add_argument('--task', type=str, default='evaluate',
                        choices=['evaluate', 'calculate_detect_std'])
    args = parser.parse_args()
    return args


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.input_josnl_dir = os.path.join(args.data_dir, args.source_type, args.time, args.language,
                                            args.source_dataset)
        self.input_jsonl_path = os.path.join(self.input_josnl_dir, args.file_name)
        # where to load features
        self.eval_dir = os.path.join(self.args.data_dir, "api", "after0301",
                                     self.args.language,
                                     "HC3_eval")
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

    def load_question_and_refs(self):
        questions, refs_lst = [], []
        with open("/data/tsq/CK/data/open/before0301/en/HC3/HC3_en.jsonl", 'r') as fin:
            lines = fin.readlines()[self.args.start_id:self.args.end_id]
            for line in lines:
                json_obj = json.loads(line.strip())
                question = json_obj["question"]
                refs = json_obj["human_answers"]
                questions.append(question)
                refs_lst.append(refs)
        return questions, refs_lst

    def evaluate(self, time_qualifier, pp_suffix="base"):
        # date
        mm_dd = time_qualifier[-5:]
        if self.args.source_dataset == "HC3" and self.args.source_type == "open":
            input_jsonl_path = self.input_jsonl_path
        else:
            input_jsonl_path = os.path.join(self.input_josnl_dir, f"data{mm_dd}_{pp_suffix}.jsonl")
        # answers
        raw_answers = []
        res_lst = []
        with open(input_jsonl_path, 'r') as fin:
            lines = fin.readlines()[self.args.start_id:self.args.end_id]
            for line in tqdm(lines, total=len(lines)):
                json_obj = json.loads(line.strip())
                # get source_task, q and a
                if self.args.source_dataset == "HC3" and self.args.source_type == "open":
                    answer = json_obj["chatgpt_answers"][0]
                else:
                    answer = json_obj["a"]
                # answer
                raw_answers.append(answer)
                res_lst.append(json_obj)

        # questions and refs
        questions, refs_lst = self.load_question_and_refs()

        # calculate score
        scores = rouge_calculation(raw_answers, refs_lst)
        assert len(scores) == len(raw_answers)

        # save
        save_path = os.path.join(self.eval_dir, f"feature{mm_dd}_{pp_suffix}.json")
        with open(save_path, 'w') as fout:
            json.dump(scores, fout)
        print(f"scores is output at: {save_path}")

    def calculate_detect_std(self):
        detect_dir = os.path.join(self.args.lgb_dir, f"train{self.args.train_time_setting}")
        # init
        sp2model2acc_lst = {}
        for sp in splits:
            model2sp2acc_lst = {}
            for feature_setting in self.args.train_feature_settings:
                model2sp2acc_lst[feature_setting] = []
            sp2model2acc_lst[sp] = model2sp2acc_lst
        # calculate acc for single_ppl_gltr
        for seed in self.args.seeds:
            single_ppl_gltr_dir = os.path.join(detect_dir, f"single_ppl_gltr_trail{seed}")
            for sp in splits:
                if sp == "val":
                    file_name = "val223-01-18.csv"
                else:
                    file_name = "test2000-35days.csv"
                single_ppl_gltr_path = os.path.join(single_ppl_gltr_dir, file_name)
                # model and its acc
                spg_df = pd.read_csv(single_ppl_gltr_path)
                model2pred = {}
                for index, row in spg_df.iterrows():
                    for key in baselines:
                        prob = row[key]
                        if prob > 50:
                            pred = 1
                        else:
                            pred = 0
                        try:
                            model2pred[key].append(pred)
                        except KeyError:
                            model2pred[key] = [pred]
                # calculate acc
                for k, preds in model2pred.items():
                    acc = accuracy_score(spg_df[["label"]], preds)
                    tn, fp, fn, tp = confusion_matrix(spg_df[["label"]], preds).ravel()
                    sp2model2acc_lst[sp][k].append({
                        "acc": acc,
                        "tn": tn/len(preds),
                        "fp": fp/len(preds),
                        "fn": fn/len(preds),
                        "tp": tp/len(preds),
                    })
        # load acc for feature
        for seed in self.args.seeds:
            for key in self.args.train_feature_settings:
                if key in baselines:
                    continue
                result_dir = os.path.join(detect_dir, f"{key}_trail{seed}")
                for sp in splits:
                    res_file_csv = os.path.join(result_dir, f"offline_{sp}.csv")
                    # model's predicts and labels
                    res_df = pd.read_csv(res_file_csv)
                    acc = accuracy_score(res_df[["label"]], res_df[["preds"]])
                    tn, fp, fn, tp = confusion_matrix(res_df[["label"]], res_df[["preds"]]).ravel()
                    sp2model2acc_lst[sp][key].append({
                        "acc": acc,
                        "tn": tn/len(res_df[["preds"]]),
                        "fp": fp/len(res_df[["preds"]]),
                        "fn": fn/len(res_df[["preds"]]),
                        "tp": tp/len(res_df[["preds"]]),
                    })
        # calculate mean and std for each model
        keys = ["acc", "tn", "fp",  "fn", "tp"]
        for sp, model2acc_lst in sp2model2acc_lst.items():
            for model, acc_lst in model2acc_lst.items():
                for k in keys:
                    k_acc_lst = [i[k] for i in acc_lst]
                    if k == "acc":
                        print(f"split {sp}, metric {k}, model {model}, average {np.mean(k_acc_lst)}, std {np.std(k_acc_lst)}")
                        # s = """${_avg}_{{\pm{_std}}}$""".format(_avg=f"{np.mean(k_acc_lst):.3f}", _std=f"{np.std(k_acc_lst):.3f}")
                        # print(s)
                    else:
                        print(f"split {sp}, metric {k}, model {model}, average {np.mean(k_acc_lst)}, std {np.std(k_acc_lst)}")
                    s = """${_avg}_{{\pm{_std}}}$""".format(_avg=f"{np.mean(k_acc_lst)*100:.1f}", _std=f"{np.std(k_acc_lst)*100:.1f}")
                    print(s)

if __name__ == '__main__':
    args = prepare_args()
    evaluator = Evaluator(args)
    if args.task == 'evaluate':
        # set time_qualifier
        if args.times == ["all"] or args.times == "all":
            files = os.listdir("/data/tsq/CK/data/api/after0301/en/HC3")
            names = []
            for file in files:
                if file.startswith("data"):
                    names.append(file.split("_")[0])
            # filter complicate days
            final_names = set(names)
            time_qualifiers = list(final_names)
            time_qualifiers.sort()
        else:
            time_qualifiers = list(args.times)
        print(f"time_qualifiers {time_qualifiers}")
        print(f"pp_suffixes {args.pp_suffixes}")
        for _time_qualifier in time_qualifiers:
            for pp_suffix in args.pp_suffixes:
                evaluator.evaluate(_time_qualifier, pp_suffix)
    elif args.task == 'calculate_detect_std':
        evaluator.calculate_detect_std()
