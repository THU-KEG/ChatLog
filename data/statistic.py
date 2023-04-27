import argparse
import os
from tqdm import tqdm
from config.conf import CONFIG
from data.database import MongoDB
import time
import json

# 按论文发布时间算
dataset2date = {
    "HC3": "2023-01-18"
}
semantics = ['MathQA', 'WSD', 'SQuAD', 'ReAding', 'WNLI', 'Cola', 'WordContext', 'TextEntail']
pragmatics = ['Aggression', 'AggressionPer', 'Spam', 'Sarcasm''ColBERT', 'TweetSent', 'TweetEmoji', 'Unhealthy',
              'UnhealthyPer', 'TweetStance', 'GoEmoPer3', 'GoEmoPer0', 'GoEmoPer1', 'GoEmoPer2', 'GoEmo', ]
dialogues = ['medicine']
QAs = ['open_qa', 'finance', 'wiki_csai', 'reddit_eli5']


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
    parser.add_argument('--chatlog', help='Which dataset', default='monthly')
    args = parser.parse_args()
    args.input_jsonl_path = os.path.join(args.data_dir, args.source_type, args.time, args.language,
                                         args.source_dataset, args.file_name)
    return args


def get_class_len_and_num(class_keys, res_dict, class_name):
    avg_len = 0
    num = 0
    for k, answers in res_dict.items():
        if k not in class_keys:
            continue
        num += len(answers)
        for ans in answers:
            avg_len += len(str(ans).strip().split())
    print("class: ", class_name)
    print("num: ", num)
    print("avg_len: ", avg_len / num)


def get_len_and_num(args):
    task2ans = {}
    with open(args.input_jsonl_path, 'r') as fin:
        lines = fin.readlines()
        for line in tqdm(lines, total=len(lines)):
            json_obj = json.loads(line.strip())
            if args.source_dataset == "HC3":
                task = json_obj["source"]
            else:
                task = json_obj["source_task"]
            if args.source_dataset == "HC3" and args.source_type == "open":
                question = json_obj["question"]
                answer = json_obj["chatgpt_answers"]
            else:
                question = json_obj["q"]
                answer = json_obj["a"]
            if args.source_dataset == "HC3":
                try:
                    task2ans[task].extend(answer)
                except KeyError:
                    task2ans[task] = answer
            else:
                try:
                    task2ans[task].append(answer)
                except KeyError:
                    task2ans[task] = [answer]
    if args.source_dataset == "HC3":
        get_class_len_and_num(QAs, task2ans, "QA")
        get_class_len_and_num(dialogues, task2ans, "Dialogue")
    elif args.source_dataset == "jat" or args.source_dataset == 'Jack_of_all_trades':
        get_class_len_and_num(semantics, task2ans, "semantics")
        get_class_len_and_num(pragmatics, task2ans, "pragmatics")

    else:
        avg_len = 0
        num = 0
        for k, answers in task2ans.items():
            num += len(answers)
            for ans in answers:
                avg_len += len(ans.strip().split())
        print("num: ", num)
        print("avg_len: ", avg_len / num)


def get_len_and_num_days(args):
    feature_dir = "/data/tsq/CK/data/api/after0301/en/HC3"
    files = os.listdir(feature_dir)
    names = []
    for file in files:
        if file.endswith("base.jsonl"):
            names.append(file.split("_")[0])
    # filter complicate days
    final_names = set(names)
    time_qualifiers = list(final_names)
    time_qualifiers.sort()
    avg_len = 0
    num = 0

    for time_qualifier in time_qualifiers:
        mm_dd = time_qualifier[-5:]
        file_name = f"data{mm_dd}_base.jsonl"
        file_path = os.path.join(feature_dir, file_name)
        with open(file_path, 'r') as fin:
            lines = fin.readlines()
            for line in tqdm(lines, total=len(lines)):
                json_obj = json.loads(line.strip())
                answer = json_obj["a"]
                num += 1
                avg_len += len(answer.strip().split())
    print("num: ", num)
    print("avg_len: ", avg_len / num)


if __name__ == '__main__':
    args = prepare_args()
    if args.chatlog == 'monthly':
        get_len_and_num(args)
    else:
        get_len_and_num_days(args)
