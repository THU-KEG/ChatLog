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
    # task
    parser.add_argument('--task', type=str, default='save',
                        choices=['save', 'update'])
    parser.add_argument('--collection', type=str, default='ChatGPT_history_record',
                        choices=['ChatGPT_detection_result', 'ChatGPT_user_label',
                                 'ChatGPT_history_record', 'ChatGPT_history_record_from_user'])
    args = parser.parse_args()
    args.input_jsonl_path = os.path.join(args.data_dir, args.source_type, args.time, args.language,
                                         args.source_dataset, args.file_name)
    return args


class DataSaver:
    def __init__(self, args):
        self.args = args
        # database
        self.collection = f"{args.collection}_{args.language}"
        mongo_url = CONFIG.mongo_chatbot_uri
        mdb = MongoDB(collection_name=self.collection, url=mongo_url)
        self.mdb = mdb
        # input
        self.input_jsonl_path = args.input_jsonl_path

    def save(self):
        num_data = 0
        with open(self.input_jsonl_path, 'r') as fin:
            lines = fin.readlines()
            for line in tqdm(lines, total=len(lines)):
                json_obj = json.loads(line.strip())
                # get source_task, q and a
                if self.args.source_dataset == "HC3":
                    source_task = json_obj["source"]
                    question = json_obj["question"]
                    chatgpt_answers = json_obj["chatgpt_answers"]
                    # human_answers = json_obj["human_answers"]
                    # print(f"num of human: {len(human_answers)}, chatgpt: {len(chatgpt_answers)}")
                    for chatgpt_answer in chatgpt_answers:
                        id = self.mdb.get_size()
                        record = {
                            "id": id,  # 自增
                            "source_type": self.args.source_type,
                            "source_dataset": self.args.source_dataset,
                            "source_task": source_task,
                            "q": question,
                            "a": chatgpt_answer,
                            "language": self.args.language,
                            "chat_date": dataset2date[self.args.source_dataset],
                            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        }
                        # log
                        self.mdb.add_one(record)
                        num_data += 1
                else:
                    # lcy modified at 3/11
                    id = self.mdb.get_size()
                    record = {
                        "id": id,
                        "source_type": json_obj["source_type"],
                        "source_dataset": json_obj["source_dataset"],
                        "source_task": json_obj["source_task"],
                        "q": json_obj["q"],
                        "a": json_obj["a"],
                        "language": json_obj["language"],
                        "chat_date": json_obj["chat_date"],
                        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    }
                    # log
                    self.mdb.add_one(record)
                    num_data += 1

        print(f"Finish saving {num_data} data into {self.collection}")

    def update(self):
        pass


class DBLoader:
    def __init__(self, args):
        self.args = args
        # data
        self.input_josnl_dir = os.path.join(args.data_dir, args.source_type, args.time, args.language,
                                            args.source_dataset)
        self.input_jsonl_path = os.path.join(self.input_josnl_dir, args.file_name)
        # where to load features
        self.feature_dir = os.path.join(self.args.data_dir, "api", "after0301",
                                        self.args.language,
                                        "HC3_features")

    def load_questions(self):
        questions = []
        with open("/data/tsq/CK/data/open/before0301/en/HC3/HC3_en.jsonl", 'r') as fin:
            lines = fin.readlines()[self.args.start_id:self.args.end_id]
            for line in lines:
                json_obj = json.loads(line.strip())
                question = json_obj["question"]
                questions.append(question)
        return questions

    def load_by_questions(self, time_qualifier):
        raw_answers = []
        res_lst = []
        with open(self.input_jsonl_path, 'r') as fin:
            lines = fin.readlines()[self.args.start_id:self.args.end_id]
            for line in tqdm(lines, total=len(lines)):
                json_obj = json.loads(line.strip())
                # get source_task, q and a
                if self.args.source_dataset == "HC3" and self.args.source_type == "open":
                    question = json_obj["question"]
                else:
                    question = json_obj["q"]
                # search in DB by question
                query = {"q": question}
                # print("query")
                # print(query)
                # get from mdb
                _res = self.mdb.get_data(query)
                res_lst = list(_res)
                for res in res_lst:
                    # print("res")
                    # print(res)
                    # print("#" * 8)
                    if res["chat_date"] == time_qualifier:
                        raw_answers.append(res["a"])
                        res_lst.append(dict(res))
                        break

        return raw_answers, res_lst

    def load_by_json(self, time_qualifier, pp_suffix=""):
        mm_dd = time_qualifier[-5:]
        if pp_suffix:
            suffix = pp_suffix
        else:
            suffix = "base"
        if self.args.source_dataset == "HC3" and self.args.source_type == "open":
            input_jsonl_path = self.input_jsonl_path
        else:
            input_jsonl_path = os.path.join(self.input_josnl_dir, f"data{mm_dd}_{suffix}.jsonl")
        raw_answers = []
        res_lst = []
        with open(input_jsonl_path, 'r') as fin:
            lines = fin.readlines()[self.args.start_id:self.args.end_id]
            for line in tqdm(lines, total=len(lines)):
                json_obj = json.loads(line.strip())
                # get source_task, q and a
                if self.args.extract_source == "chatgpt_answers":
                    if self.args.source_dataset == "HC3" and self.args.source_type == "open":
                        answer = json_obj["chatgpt_answers"][0]
                    else:
                        answer = json_obj["a"]
                    # answer
                    raw_answers.append(answer)
                else:
                    raw_answers.extend(json_obj["human_answers"])
                res_lst.append(json_obj)
        if self.args.extract_source == "chatgpt_answers":
            save_json_path = os.path.join(self.feature_dir, f"feature{mm_dd}_{suffix}.json")
        else:
            print(f"raw_answers: {len(raw_answers)}")
            save_json_path = os.path.join(self.feature_dir, f"feature{mm_dd}_human.json")
        return raw_answers, res_lst, save_json_path

    def load_feature_by_json(self, times, pp_suffixes):
        # set time_qualifier
        print(f"times: {times}")
        if times == ["all"] or times == "all":
            print(self.feature_dir)
            files = os.listdir(self.feature_dir)
            names = []
            for file in files:
                if file.startswith("feature"):
                    names.append(file.split("_")[0])
            # filter complicate days
            final_names = set(names)
            time_qualifiers = list(final_names)
            time_qualifiers.sort()
        else:
            time_qualifiers = times

        # features dict, k1: pp_suffix, k2: time_qualifier, v: feature(also a dict load from json)
        features_dict = {}
        for pp_suffix in pp_suffixes:
            pp_features_dict = {}
            for time_qualifier in time_qualifiers:
                mm_dd = time_qualifier[-5:]
                file_name = f"feature{mm_dd}_{pp_suffix}.json"
                file_path = os.path.join(self.feature_dir, file_name)
                print("pp_suffixes: ", pp_suffixes)
                print("file_path: ", file_path)
                if len(pp_suffixes) == 1 and not os.path.exists(file_path):
                    if mm_dd in ['04-12', '06-14', '06-25', '07-05', '07-07', '09-15', '09-20', '09-22', '09-28']:
                        file_name = f"feature{mm_dd}_base1.json"
                    else:
                        file_name = f"feature{mm_dd}_base.json"
                    file_path = os.path.join(self.feature_dir, file_name)
                    
                with open(file_path, 'r') as fin:
                    feature_obj = json.load(fin)
                    pp_features_dict[mm_dd] = feature_obj

            features_dict[pp_suffix] = pp_features_dict
        return features_dict


if __name__ == '__main__':
    args = prepare_args()
    data_saver = DataSaver(args)
    if args.task == 'save':
        data_saver.save()
    elif args.task == 'update':
        data_saver.update()
