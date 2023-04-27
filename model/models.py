import json
from model.analyse import prepare_args
from model.diff import Differentiation
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import os
from sklearn.metrics import accuracy_score, auc, confusion_matrix

version2api = {
    "cpm2": "http://localhost:5452/cpm",
    "glm_base": "http://localhost:9546/glm",
    "glm130b_base": "http://103.238.162.37:9622/general",
    "cdail_gpt": "http://0.0.0.0:9600/cdail",
    "eva": "http://0.0.0.0:9601/eva",
    "gpt3": "http://0.0.0.0:9602/gpt",
    "bm25": "http://0.0.0.0:9200",
    "ChatGPT": "http://0.0.0.0:9200",
}

linguistic_features = ['WRich05_S', 'WRich10_S', 'WRich15_S', 'WRich20_S', 'WClar05_S', 'WClar10_S', 'WClar15_S',
                       'WClar20_S', 'WNois05_S', 'WNois10_S', 'WNois15_S', 'WNois20_S', 'WTopc05_S', 'WTopc10_S',
                       'WTopc15_S', 'WTopc20_S', 'BRich05_S', 'BRich10_S', 'BRich15_S', 'BRich20_S', 'BClar05_S',
                       'BClar10_S', 'BClar15_S', 'BClar20_S', 'BNois05_S', 'BNois10_S', 'BNois15_S', 'BNois20_S',
                       'BTopc05_S', 'BTopc10_S', 'BTopc15_S', 'BTopc20_S', 'to_EntiM_C', 'as_EntiM_C', 'at_EntiM_C',
                       'to_UEnti_C', 'as_UEnti_C', 'at_UEnti_C', 'ra_SSTo_C', 'ra_SOTo_C', 'ra_SXTo_C', 'ra_SNTo_C',
                       'ra_OSTo_C', 'ra_OOTo_C', 'ra_OXTo_C', 'ra_ONTo_C', 'ra_XSTo_C', 'ra_XOTo_C', 'ra_XXTo_C',
                       'ra_XNTo_C', 'ra_NSTo_C', 'ra_NOTo_C', 'ra_NXTo_C', 'ra_NNTo_C', 'LoCohPA_S', 'LoCohPW_S',
                       'LoCohPU_S', 'LoCoDPA_S', 'LoCoDPW_S', 'LoCoDPU_S', 'to_NoPhr_C', 'as_NoPhr_C', 'at_NoPhr_C',
                       'ra_NoVeP_C', 'ra_NoSuP_C', 'ra_NoPrP_C', 'ra_NoAjP_C', 'ra_NoAvP_C', 'to_VePhr_C', 'as_VePhr_C',
                       'at_VePhr_C', 'ra_VeNoP_C', 'ra_VeSuP_C', 'ra_VePrP_C', 'ra_VeAjP_C', 'ra_VeAvP_C', 'to_SuPhr_C',
                       'as_SuPhr_C', 'at_SuPhr_C', 'ra_SuNoP_C', 'ra_SuVeP_C', 'ra_SuPrP_C', 'ra_SuAjP_C', 'ra_SuAvP_C',
                       'to_PrPhr_C', 'as_PrPhr_C', 'at_PrPhr_C', 'ra_PrNoP_C', 'ra_PrVeP_C', 'ra_PrSuP_C', 'ra_PrAjP_C',
                       'ra_PrAvP_C', 'to_AjPhr_C', 'as_AjPhr_C', 'at_AjPhr_C', 'ra_AjNoP_C', 'ra_AjVeP_C', 'ra_AjSuP_C',
                       'ra_AjPrP_C', 'ra_AjAvP_C', 'to_AvPhr_C', 'as_AvPhr_C', 'at_AvPhr_C', 'ra_AvNoP_C', 'ra_AvVeP_C',
                       'ra_AvSuP_C', 'ra_AvPrP_C', 'ra_AvAjP_C', 'to_TreeH_C', 'as_TreeH_C', 'at_TreeH_C', 'to_FTree_C',
                       'as_FTree_C', 'at_FTree_C', 'to_NoTag_C', 'as_NoTag_C', 'at_NoTag_C', 'ra_NoAjT_C', 'ra_NoVeT_C',
                       'ra_NoAvT_C', 'ra_NoSuT_C', 'ra_NoCoT_C', 'to_VeTag_C', 'as_VeTag_C', 'at_VeTag_C', 'ra_VeAjT_C',
                       'ra_VeNoT_C', 'ra_VeAvT_C', 'ra_VeSuT_C', 'ra_VeCoT_C', 'to_AjTag_C', 'as_AjTag_C', 'at_AjTag_C',
                       'ra_AjNoT_C', 'ra_AjVeT_C', 'ra_AjAvT_C', 'ra_AjSuT_C', 'ra_AjCoT_C', 'to_AvTag_C', 'as_AvTag_C',
                       'at_AvTag_C', 'ra_AvAjT_C', 'ra_AvNoT_C', 'ra_AvVeT_C', 'ra_AvSuT_C', 'ra_AvCoT_C', 'to_SuTag_C',
                       'as_SuTag_C', 'at_SuTag_C', 'ra_SuAjT_C', 'ra_SuNoT_C', 'ra_SuVeT_C', 'ra_SuAvT_C', 'ra_SuCoT_C',
                       'to_CoTag_C', 'as_CoTag_C', 'at_CoTag_C', 'ra_CoAjT_C', 'ra_CoNoT_C', 'ra_CoVeT_C', 'ra_CoAvT_C',
                       'ra_CoSuT_C', 'to_ContW_C', 'as_ContW_C', 'at_ContW_C', 'to_FuncW_C', 'as_FuncW_C', 'at_FuncW_C',
                       'ra_CoFuW_C', 'SimpTTR_S', 'CorrTTR_S', 'BiLoTTR_S', 'UberTTR_S', 'MTLDTTR_S', 'SimpNoV_S',
                       'SquaNoV_S', 'CorrNoV_S', 'SimpVeV_S', 'SquaVeV_S', 'CorrVeV_S', 'SimpAjV_S', 'SquaAjV_S',
                       'CorrAjV_S', 'SimpAvV_S', 'SquaAvV_S', 'CorrAvV_S', 'to_AAKuW_C', 'as_AAKuW_C', 'at_AAKuW_C',
                       'to_AAKuL_C', 'as_AAKuL_C', 'at_AAKuL_C', 'to_AABiL_C', 'as_AABiL_C', 'at_AABiL_C', 'to_AABrL_C',
                       'as_AABrL_C', 'at_AABrL_C', 'to_AACoL_C', 'as_AACoL_C', 'at_AACoL_C', 'to_SbFrQ_C', 'as_SbFrQ_C',
                       'at_SbFrQ_C', 'to_SbCDC_C', 'as_SbCDC_C', 'at_SbCDC_C', 'to_SbFrL_C', 'as_SbFrL_C', 'at_SbFrL_C',
                       'to_SbCDL_C', 'as_SbCDL_C', 'at_SbCDL_C', 'to_SbSBW_C', 'as_SbSBW_C', 'at_SbSBW_C', 'to_SbL1W_C',
                       'as_SbL1W_C', 'at_SbL1W_C', 'to_SbSBC_C', 'as_SbSBC_C', 'at_SbSBC_C', 'to_SbL1C_C', 'as_SbL1C_C',
                       'at_SbL1C_C', 'TokSenM_S', 'TokSenS_S', 'TokSenL_S', 'as_Token_C', 'as_Sylla_C', 'at_Sylla_C',
                       'as_Chara_C', 'at_Chara_C', 'FleschG_S', 'AutoRea_S', 'ColeLia_S', 'SmogInd_S', 'Gunning_S',
                       'LinseaW_S']
top_10_features = ['to_SbFrQ_C', 'to_SbFrL_C', 'as_SbFrQ_C', 'as_SbFrL_C', 'to_SbSBW_C', 'to_SbCDC_C', 'to_SbCDL_C',
                   'at_SbFrQ_C', 'at_SbFrL_C', 'as_SbSBW_C']
bottom_10_features = ['at_VeTag_C', 'ra_ONTo_C', 'at_SbL1C_C', 'at_ContW_C', 'at_FTree_C', 'BiLoTTR_S', 'BClar15_S',
                      'BClar20_S', 'ra_NNTo_C', 'ColeLia_S']
# ensure random won't choose the top/bottom features
for feature in bottom_10_features:
    linguistic_features.remove(feature)
for feature in top_10_features:
    linguistic_features.remove(feature)
PLM_4_features = ['qa', 'single', 'gltr', 'ppl']

diff_found_features = {
    "top10": top_10_features,
    "top5": top_10_features[:5],
    "bottom10": bottom_10_features,
    "bottom5": bottom_10_features[-5:],
    "plm4": PLM_4_features,
    'qa': ['qa'],
    'single': ['single'],
    'gltr': ['gltr'],
    'ppl': ['ppl']
}


class Classifier:
    def __init__(self, args):
        self.args = args
        random.seed(self.args.seed)
        # qualify the feature we use
        self.limited_features = []

    def set_limited_features(self, keys):
        if self.args.train_feature_setting != "no_limit":
            print(self.args.train_feature_setting)
            limited_feat_keys = self.args.train_feature_setting.split("_")
            for k in limited_feat_keys:
                if k.startswith("random"):
                    # random choose
                    num = int(k[6:])
                    # random.seed(self.args.seed)
                    feat_keys = random.sample(linguistic_features, num)
                    # random.seed(1453)
                    self.limited_features.extend(feat_keys)
                else:
                    # if self.args.train_feature_setting in PLM_4_features:
                    #     self.limited_features.append(self.args.train_feature_setting)
                    # else:
                    self.limited_features.extend(diff_found_features[k])

    def load_data(self):
        # First, prepare the data
        differ = Differentiation(self.args, return_type=self.args.return_type)
        # load human features
        human_path = "/data/tsq/CK/data/api/after0301/en/HC3_features/feature01-18_human.json"
        with open(human_path, 'r') as fin:
            # k: feature_name, v: [feature values for 3000 raw answers]
            human_feature_obj = json.load(fin)
            keys = list(human_feature_obj.keys())
            total_human_num = len(human_feature_obj[keys[0]])
            id2human_features = [{} for i in range(total_human_num)]
            self.set_limited_features(keys)
            for k, lst in human_feature_obj.items():
                if self.limited_features:
                    # have limits
                    if k not in self.limited_features:
                        continue
                for i, score in enumerate(lst):
                    id2human_features[i][k] = score

        # 250 feature for 1000 Id for 20 day
        _, features_by_condition = differ.diff_features(self.args.diff_type)
        return features_by_condition, id2human_features

    def split_data(self, bot_features_by_condition, id2human_features, save_dir):
        # sample
        half_num = len(id2human_features) // 2
        print(f"len(id2human_features): {len(id2human_features)}")
        print(f"half_num: {half_num}")
        train_features = random.sample(id2human_features[:half_num], self.args.train_human_num)
        print(f"id2human_features[half_num:]: {len(id2human_features[half_num:])}")
        print(f"self.args.test_num: {self.args.test_num}")
        test_human_features = random.sample(id2human_features[half_num:], self.args.test_num)
        print(f"train_features: {len(train_features)}")
        print(f"test_human_features: {len(test_human_features)}")
        feature_keys = list(train_features[0].keys())
        # random sampling some bot features for everyday
        test_bot_features = []
        day_num = len(bot_features_by_condition.keys()) - 1  # 1 for 01-18
        test_every_day_num = self.args.test_num // day_num
        if test_every_day_num * day_num < self.args.test_num:
            test_every_day_num += 1
        print(f"day_num: {day_num}")
        print(f"test_every_day_num: {test_every_day_num}")
        for mm_dd, feature_obj in bot_features_by_condition.items():
            total_bot_num = len(feature_obj[feature_keys[0]])
            id2bot_features = [{} for i in range(total_bot_num)]
            for k in feature_keys:
                if self.limited_features:
                    # have limits
                    if k not in self.limited_features:
                        continue
                lst = feature_obj[k]
                for i, score in enumerate(lst):
                    id2bot_features[i][k] = score
            print(mm_dd)
            if mm_dd == "01-18":
                # training
                print("########### 01-18")
                train_features.extend(id2bot_features)
            else:
                # testing
                test_lst = random.sample(id2bot_features, test_every_day_num)
                test_bot_features.extend(test_lst)
        # clip
        test_bot_features = test_bot_features[:self.args.test_num]
        # add labels
        for i in range(self.args.train_human_num):
            # train_features[i]["label"] = "human"
            train_features[i]["label"] = 0
        for i in range(self.args.train_human_num, len(train_features)):
            # train_features[i]["label"] = "ChatGPT"
            train_features[i]["label"] = 1
        print("len(test_human_features)")
        print(len(test_human_features))
        print("len(test_bot_features)")
        print(len(test_bot_features))
        for i in range(len(test_bot_features)):
            # test_bot_features[i]["label"] = "ChatGPT"
            test_bot_features[i]["label"] = 1
            # test_human_features[i]["label"] = "human"
            test_human_features[i]["label"] = 0
        test_bot_features.extend(test_human_features)
        # for dic in test_bot_features:
        #     print(dic["label"])
        # shuffle
        random.shuffle(train_features)
        random.shuffle(test_bot_features)
        trains = pd.DataFrame(train_features)
        offline_test = pd.DataFrame(test_bot_features)

        # to csv
        offline_test_path = os.path.join(save_dir, f"test{len(offline_test)}-{day_num}days.csv")
        offline_test.to_csv(offline_test_path)
        return trains, offline_test

    def train(self):
        features_by_condition, id2human_features = self.load_data()
        print("[1] Finish loading data for Bot and Human")
        # save to save_dir
        # if self.args.train_feature_setting.startswith('random'):
        last_name = f"{self.args.train_feature_setting}_trail{self.args.seed}"
        # else:
        #     last_name = self.args.train_feature_setting
        save_dir = os.path.join(self.args.lgb_dir, f"train{self.args.train_time_setting}",
                                last_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        trains, offline_test = self.split_data(features_by_condition, id2human_features, save_dir)
        # training with lgb
        # if self.args.train_feature_setting in PLM_4_features:
        #     # no need to train lgb, just calculate acc
        #     calculate_plm_acc(trains, offline_test, lgb_dir, )
        #     print(f"[2] Finish calculating acc for {self.args.train_feature_setting}")
        # else:
        self.run_single_rgb(trains, offline_test, save_dir)
        print("[2] Finish training with lgb")

    def run_single_rgb(self, trains, offline_test, lgb_dir, num_class=2, max_depth=4):
        # split the data, train:valid = 8:1
        print("split the data")
        # train_xy, offline_test = train_test_split(trains, test_size=0.1, random_state=21)
        train, val = train_test_split(trains, test_size=1 / 9, random_state=21)
        # save
        train_path = os.path.join(lgb_dir, f"train{len(trains)}-01-18.csv")
        trains.to_csv(train_path)
        val_path = os.path.join(lgb_dir, f"val{len(val)}-01-18.csv")
        val.to_csv(val_path)
        # drop
        label_name = "label"
        drop = [label_name]
        print("train set")
        print(trains.shape)
        y = train.label  # train set label
        val_y = val.label  # valid set label
        X = train.drop(drop, axis=1)  # train set Characteristic matrix
        print("valid set")
        val_X = val.drop(drop, axis=1)  # valid set Characteristic matrix

        print("test set")
        offline_test_X = offline_test.drop(drop, axis=1)

        # data process
        # check Series.dtypes
        # for row in X.iterrows():
        #     print(row.dtypes)
        print(X.dtypes)
        print(y.dtypes)
        print(val_X.dtypes)
        print("val_X.shape")
        print(val_X.shape)
        print(val_y.dtypes)
        lgb_train = lgb.Dataset(X, y, free_raw_data=False)
        lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train, free_raw_data=False)

        # start training
        seed = 1453
        print('set parameter')
        # params = {
        #     'lambda_l1': 0.1,
        #     'lambda_l2': 0.2,
        #     'max_depth': max_depth,
        #     'learning_rate': 0.1,
        #     'random_state': seed,
        #     'device': 'cpu',
        #     'objective': 'multiclass',
        #     'num_class': num_class,
        # }
        print("start training")
        # gbm = lgb.train(params,  # parameter dict
        #                 lgb_train,  # train set
        #                 num_boost_round=2000,  # iterator round
        #                 valid_sets=[lgb_eval],  # valid set
        #                 early_stopping_rounds=30)  # param of early_stopping

        boost_round = 50
        early_stop_rounds = 10

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': ['auc'],
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'random_state': seed,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        results = {}
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=boost_round,
                        valid_sets=(lgb_eval, lgb_train),
                        valid_names=('validate', 'train'),
                        early_stopping_rounds=early_stop_rounds,
                        evals_result=results)
        self.test(lgb_dir, X, val_X, val_y, label_name, gbm, max_depth, split="val")
        self.test(lgb_dir, X, offline_test_X, offline_test, label_name, gbm, max_depth, split="test")

    def test(self, lgb_dir, X, offline_test_X, offline_test, label_name, gbm, max_depth, split):
        # offline predict
        print("offline predict")
        print("offline_test_X.shape")
        print(offline_test_X.shape)
        print("offline_test.shape")
        print(offline_test.shape)
        preds_offline_probs = gbm.predict(offline_test_X, num_iteration=gbm.best_iteration)  # output probability
        print("preds_offline_probs")
        print(preds_offline_probs)
        # preds_offline = preds_offline_probs.argmax()
        preds_offline = []
        for pred in preds_offline_probs:
            if pred > 0.5:
                preds_offline.append(1)
            else:
                preds_offline.append(0)

        print(offline_test)
        print(split)
        if split == "test":
            offline = offline_test[[label_name]]
        else:
            offline = offline_test
        offline = offline.values
        offline = pd.DataFrame(offline, columns=['label'])
        print("preds_offline")
        print(preds_offline)

        preds_offline_df = pd.DataFrame(list(preds_offline), columns=['preds'])
        print("offline.shape", offline.shape)
        print("preds_offline_df.shape", preds_offline_df.shape)
        offline = pd.concat([offline, preds_offline_df], axis=1, ignore_index=True)
        offline.columns = ['label', 'preds']
        # offline.loc[:,'preds'] = preds_offline_df['prob']
        # offline.label = offline['label'].astype(np.float64)
        offline.label = offline['label']
        offline.to_csv(os.path.join(lgb_dir, f"offline_{split}.csv"), index=None, encoding='gbk')
        score_path = os.path.join(lgb_dir, "scores.txt")
        score_fout = open(score_path, "w")
        self.count_scores(offline.label, offline.preds, score_fout)
        # choose character
        print("choose character")
        df = pd.DataFrame(X.columns.tolist(), columns=['feature'])
        df['importance'] = list(gbm.feature_importance())
        df = df.sort_values(by='importance', ascending=False)
        if split == 'test':
            df.to_csv(os.path.join(lgb_dir, "feature_score.csv"), index=None, encoding='gbk')
        # write result
        # accuracy
        tmp_fout = open("tmp.txt", "a")
        tmp_fout.write(
            'max_depth: %d,  accuracy: %.3f\n' % (max_depth, float(accuracy_score(offline.label, offline.preds))))

    def count_scores(self, label, preds, score_fout):
        print("label")
        print(label)
        print("preds")
        print(preds)
        score = accuracy_score(label, preds)
        score_fout.write("Accuracy:\n")
        score_fout.write(f"{score}\n")
        array = confusion_matrix(label, preds)
        score_fout.write("Confusion matrix:\n")
        score_fout.write("TP & TN & FP & FN \n")
        tp = array[1][1]
        tn = array[0][0]
        fp = array[0][1]
        fn = array[1][0]
        score_fout.write(f"{tp} & {tn} & {fp} & {fn} \\\\ \n")
        total = len(label)
        score_fout.write(
            f"{round(tp / total, 2)} & {round(tn / total, 2)} & {round(fp / total, 2)} & {round(fn / total, 2)} \\\\\n")


if __name__ == '__main__':
    args = prepare_args()
    classifier = Classifier(args)
    if args.task == "train":
        classifier.train()
