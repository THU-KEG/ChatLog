import json

from data.save_data import DBLoader
from model.extraction import lang2uie_schema, cogie_schema
import numpy as np
from collections import Counter
import pandas as pd
import os

uie_schema = lang2uie_schema['en']
knowledge_schemas = uie_schema + cogie_schema
hc3_group = ['qa', 'single', 'gltr', 'ppl']
prf_lst = ['p', 'r', 'f']


def calc_avg_var_cov(id_record):
    # id_record: List (id) of List (days)
    avg_lst, var_lst, coefficient_of_variances = [], [], []
    for lst in id_record:
        # calculate through days for each id
        arr = np.array(lst)
        avg = np.mean(arr)
        var = np.var(arr)
        coefficient_of_variance = var / avg
        avg_lst.append(avg)
        var_lst.append(var)
        coefficient_of_variances.append(coefficient_of_variance)
    # return average of ids
    # return np.mean(avg_lst), np.mean(var_lst), np.mean(coefficient_of_variances)
    mean = np.mean(avg_lst)
    variation = np.mean(var_lst)
    # np.mean(coefficient_of_variances)
    coefficient_of_variances = variation / abs(mean)
    return mean, variation, coefficient_of_variances


class Differentiation:
    def __init__(self, args, return_type="var_score"):
        self.args = args
        self.loader = DBLoader(args)
        self.return_type = return_type
        self.total = self.args.end_id - self.args.start_id
        self.conditions = []
        self.condition2features = {}

    def diff_features(self, diff_type):
        # load data
        feature2id_record = {}
        features_by_time = {}
        if diff_type == "time":
            #  diff the time
            feature2id_record, features_by_time = self.diff_through_time()

        # calculate return score
        if self.return_type == "var_score":
            df, counts = self.get_avg_scores(feature2id_record)
            return df
        elif self.return_type == "hc3_group":
            df = self.get_feature_group()
            return df
        elif self.return_type == "feature_and_eval":
            dfs = []
            for suffix, features in features_by_time.items():
                df = self.get_feature_and_eval(features, suffix)
                dfs.append(df)
            # average each df
            df_average = dfs[0].drop(columns=['time']) 
            for i in range(1, len(dfs)):
                next_df = dfs[i].drop(columns=['time'])
                print("Concatenate: ")
                print(df_average.shape, next_df.shape)
                assert df_average.shape == next_df.shape
                df_average += next_df
            df_average /= len(dfs)
            df_average = df_average.join(dfs[0]['time'])
            return df_average
        else:
            # return dict
            return feature2id_record, features_by_time

    def load_feature_by_type(self, feature_obj):
        final_features = {}
        if self.args.feature_type == "knowledge" or self.args.feature_type == "all":
            if self.args.knowledge_extractor == "uie" or self.args.knowledge_extractor == "all":
                # load uie
                for k in uie_schema:
                    final_features[k] = feature_obj[k]
            if self.args.knowledge_extractor == "cogie" or self.args.knowledge_extractor == "all":
                # load cogie
                for k in cogie_schema:
                    final_features[k] = feature_obj[k]
        if self.args.feature_type == "linguistic" or self.args.feature_type == "all":
            # load lingfeat
            for k, dic in feature_obj.items():
                if k not in knowledge_schemas and k not in hc3_group:
                    final_features[k] = dic

        if self.args.feature_type == "classify" or self.args.feature_type == "all":
            # load classify
            for k, dic in feature_obj.items():
                if k in hc3_group:
                    final_features[k] = dic

        return final_features

    def get_feature_and_eval(self, features_by_time, suffix):
        eval_dir = os.path.join(self.args.data_dir, "api", "after0301",
                                self.args.language,
                                "HC3_eval")
        new_feature_and_eval = {}
        for mm_dd, final_features in features_by_time.items():
            # load rouge scores from json
            eval_path = os.path.join(eval_dir, f"feature{mm_dd}_{suffix}.json")
            if not os.path.exists(eval_path):
                 if mm_dd == '04-12':
                    eval_path = os.path.join(eval_dir, f"feature{mm_dd}_base1.json")
                 else:
                    eval_path = os.path.join(eval_dir, f"feature{mm_dd}_base.json")
            with open(eval_path, 'r') as fin:
                rouge_scores = json.load(fin)
                rouge2scores = {}
                keys = []
                for rk in rouge_scores[0].keys():
                    for prf in prf_lst:
                        real_key = f"{rk}-{prf}"
                        keys.append(real_key)
                # init rouge-1-p, rouge-1-r, rouge-1-f, ...
                for k in keys:
                    rouge2scores[k] = []
                # load all rouge scores
                for rouge_score in rouge_scores:
                    for rk, score_dict in rouge_score.items():
                        for prf in prf_lst:
                            real_key = f"{rk}-{prf}"
                            rouge2scores[real_key].append(score_dict[prf])

                final_features.update(rouge2scores)
                time_lst = [mm_dd] * len(rouge_scores)
                if not new_feature_and_eval:
                    # init
                    new_feature_and_eval = final_features
                    new_feature_and_eval["time"] = time_lst
                else:
                    for k, lst in final_features.items():
                        new_feature_and_eval[k].extend(lst)
                    new_feature_and_eval["time"].extend(time_lst)
        # return df
        print('new_feature_and_eval keys:')
        print(new_feature_and_eval.keys())
        df = pd.DataFrame(new_feature_and_eval)
        print("get_feature_and_eval, return shape:")
        print(df.shape)
        # set hc3_group to human score
        for k in hc3_group:
            if k in new_feature_and_eval.keys():
                df[k] = 100 - df[k]
        return df

    def get_feature_group(self):
        conditions = []
        scores = []
        features = []
        # for feature in feature_group:
        #     id_record = feature2id_record[feature]
        #     for lst in id_record:
        #         for i, score in enumerate(lst):
        #             condition = self.conditions[i]
        #             conditions.append(condition)
        #             scores.append(score)
        for condition, feature_obj in self.condition2features.items():
            for feature, lst in feature_obj.items():
                # count average of 1000 questions
                if self.args.feature_group_type == "avg_prob":
                    avg = sum(lst) / len(lst)
                else:
                    acc_lst = [100 if prob > 50 else 0 for prob in lst]
                    avg = sum(acc_lst) / len(acc_lst)
                conditions.append(condition)
                features.append(feature)
                scores.append(avg)

        final_dict = {
            "condition": conditions,
            "score": scores,
            "feature": features,
        }
        return pd.DataFrame(final_dict)

    def get_avg_scores(self, feature2id_record):
        # turn features into counts and max frequency for knowledge features
        feature = []
        frequency = []
        label = []
        for k, lst in feature2id_record.items():
            if k not in knowledge_schemas:
                # linguistic features are all numeric
                avg, var, coefficient_of_variance = calc_avg_var_cov(lst)
                if var == 0 and avg == 0:
                    print("zero")
                    print(k)
                    print(lst[0])
                    continue
                # avg
                feature.append(k)
                frequency.append(avg)
                label.append("avg score")
                # var
                feature.append(k)
                frequency.append(var)
                label.append("var score")
                # coefficient of variance
                feature.append(k)
                frequency.append(coefficient_of_variance)
                label.append("coefficient of variance")
            else:
                # knowledge features
                # first
                feature.append(k[:3])  # 缩写一下
                avg_freq = (len(lst) / self.total)
                frequency.append(avg_freq)
                label.append("avg frequency")
                # second
                feature.append(k[:3])
                print("len:", len(lst))
                print("lst 0:", lst[0])
                count = Counter(lst)
                ans_dict = count.most_common(1)[0]
                max_freq = ans_dict[1] / self.total
                frequency.append(max_freq)
                # counts[f"#{k[0]}_{ans_dict[0]}"] =
                label.append("max frequency")
        # get features count
        counts = {"feature": feature, "frequency": frequency, "label": label}
        # print(counts)
        df = pd.DataFrame(counts)
        # print(df)
        return df, counts

    def diff_through_time(self):
        # fix the prompt and parameter
        # pp_suffix = self.args.pp_suffixes[0]
        pp_sufffix2features = {}
        for pp_suffix in self.args.pp_suffixes:
            # load feature
            features_dict = self.loader.load_feature_by_json(self.args.times, [pp_suffix])
            _features_by_time = features_dict[pp_suffix]
            features_by_time = {}
            for mm_dd, feature_obj in _features_by_time.items():
                final_features = self.load_feature_by_type(feature_obj)
                features_by_time[mm_dd] = final_features
            # 250 feature for 1000 Id for 20 day
            # we should calculate the variation on each id with 20 days
            feature2id_record = {}
            print(features_by_time.keys())
            feature_keys = list(features_by_time[list(features_by_time.keys())[0]].keys())
            print(feature_keys)
            print(len(feature_keys))
            for k in feature_keys:
                # init
                feature2id_record[k] = [[] for i in range(self.total)]
            # transform into feature2id_record
            for mm_dd, feature_obj in features_by_time.items():
                for k, lst in feature_obj.items():
                    for i, value in enumerate(lst):
                        # feature2id_record[k][i] will finally reach a length of days
                        feature2id_record[k][i].append(value)
            self.conditions = [f"{mm_dd}_base" for mm_dd in features_by_time.keys()]
            self.condition2features = features_by_time
            pp_sufffix2features[pp_suffix] = features_by_time
        return feature2id_record, pp_sufffix2features
