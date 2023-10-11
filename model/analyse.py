import argparse
import json
import os
import pandas as pd
from model.visualization import Visualizer
from model.diff import Differentiation
from data.save_data import DBLoader
from model.extraction import Extractor


def prepare_args():
    parser = argparse.ArgumentParser(description='Analyse inherent patterns')
    # input data
    parser.add_argument('--data_dir', help='Where to load', default='/data/tsq/CK/data')
    parser.add_argument('--source_type', help='open or api', default='open',
                        choices=['open', 'api'])
    parser.add_argument('--time', help='When is the chat', default='before0301')
    parser.add_argument('--language', help='en/zh', default='en',
                        choices=['en', 'zh'])
    parser.add_argument('--source_dataset', help='Which dataset', default='HC3')
    # jsonl file
    parser.add_argument('--file_name', help='Which dataset', default='HC3_en.jsonl')
    parser.add_argument('--start_id', help='start id', type=int, default=0)
    parser.add_argument('--end_id', help='end id', type=int, default=1000)
    parser.add_argument('--times', help='For Changing', type=str, nargs='+', default=
    # ["2023-01-18"]
    # ["2023-03-05", "2023-03-06", "2023-03-07", "2023-03-08", "2023-03-09", "2023-03-10", ]
    # ["2023-03-11", "2023-03-12", "2023-03-13", "2023-03-14", "2023-03-15"]
    # ["2023-03-17", "2023-03-18", "2023-03-19", "2023-03-20", "2023-03-21", "2023-03-22", ]
    # ["2023-03-05", "2023-03-06", "2023-03-07", "2023-03-08", "2023-03-09", "2023-03-10",
    #  "2023-03-11", "2023-03-12", "2023-03-13", "2023-03-14", "2023-03-15",
    #  "2023-03-17", "2023-03-18", "2023-03-19", "2023-03-20", "2023-03-21", "2023-03-22", ]
    # ["2023-03-23", "2023-03-24", "2023-03-25", "2023-03-26", "2023-03-27", "2023-03-28", ]
    # 10 days for training
    # ["2023-01-18", "2023-03-05", "2023-03-07", "2023-03-09", "2023-03-13", "2023-03-15",
    #  "2023-03-17", "2023-03-19", "2023-03-22", "2023-03-25", "2023-03-28", ],
    ["all"]
                        )
    parser.add_argument('--pp_suffixes', help='For Changing', type=str, nargs='+', default=
    # ["base1", "base2", "base3"]
    ["base1"]
                        # ["base", "para", "prompt", "prompt_para"]
                        # [""]
                        )
    # visualization parameter
    parser.add_argument('--batch_size', help='For UIE', type=int, default=1)
    parser.add_argument('--pic_dir', help='Where to save', default='/data/tsq/CK/pic')
    parser.add_argument('--pic_name', help='Where to save', default='avg_HC3')
    parser.add_argument('--pic_type', help='picture type', default='bar', choices=['bar', 'radar', 'line', 'corr'])
    parser.add_argument('--corr_type', help='corr type', default='pearson',
                        choices=['pearson', 'kendall', 'spearman', 'tcc'])
    parser.add_argument('--pic_topk_feat', help='How many first feature be selected',
                        type=int, default=0)
    parser.add_argument('--vis_task', type=str, default='diff_snapshots_by_var',
                        choices=['visualize_one_snapshot', 'visualize_one_feature', 'diff_snapshots_by_var'])
    parser.add_argument('--diff_type', type=str, default='time',
                        choices=['time', 'prompt', 'para'])
    # for visualize_one_feature
    parser.add_argument('--return_type', help='Which feature group to visualize', default='dict',
                        choices=['hc3_group', 'var_score', 'dict', 'feature_and_eval'])
    parser.add_argument('--feature_group_type', help='For feature group, how to calculate score', default='avg_prob',
                        choices=['avg_prob', 'avg_acc'])
    # for visualize_one_snapshot
    parser.add_argument('--knowledge_category', help='knowledge category', default=['Person'])
    # feature extraction parameter
    parser.add_argument('--feature_type', help='feature type', default='all',
                        choices=['knowledge', 'linguistic', 'classify', 'all'])
    parser.add_argument('--extract_source', type=str, default='chatgpt_answers',
                        choices=['human_answers', 'chatgpt_answers'])
    parser.add_argument('--knowledge_extractor', help='knowledge extractor', default='all',
                        choices=['uie', 'cogie', 'all', 'no'])
    parser.add_argument('--linguistic_features', help='linguistic features to extract',
                        type=str, nargs='+',
                        default=["AdSem", "Disco", "Synta", "LxSem", "ShTra"])
    # training parameters
    parser.add_argument('--lgb_dir', type=str, default='/data/tsq/CK/model/lgb', )
    parser.add_argument('--train_time_setting', type=str, default='only01',
                        choices=['only01', '01to03', ])
    parser.add_argument('--train_feature_setting', type=str, default='no_limit',
                        choices=['bottom10', 'top10', 'top10_bottom10',
                                 'bottom5', 'top5', 'top5_bottom5',
                                 'bottom10_single', 'random10_single', 'single_ppl_gltr',
                                 'random10', 'random5', 'no_limit'])
    parser.add_argument('--train_human_num', help='data num for human labels', type=int, default=1000)
    parser.add_argument('--test_num', help='data num for test human labels', type=int, default=1000)
    parser.add_argument('--seed', help='random seed', type=int, default=1)
    # task
    parser.add_argument('--task', type=str, default='extract_feature',
                        choices=['extract_feature', 'visualize', 'train'])

    args = parser.parse_args()

    return args


def visualize_one_snapshot(args, save_path):
    # first, prepare the data
    loader = DBLoader(args)
    _time_qualifier = args.times
    df, counts = loader.load_feature_by_json(_time_qualifier, args.pp_suffixes)
    # df, counts = extractor.analyse_features(raw_answers)
    visualizer = Visualizer(args, df, save_path)
    visualizer.draw_pic(args.pic_type)  # maybe multiple draws


def visualize_one_feature(args, save_path):
    # first, prepare the data
    differ = Differentiation(args, return_type=args.return_type)
    df = differ.diff_features(args.diff_type)
    # df has 3 columns, one is conditions, two is feature scores, three is hue(feature names)
    visualizer = Visualizer(args, df, save_path)
    visualizer.draw_pic(args.pic_type)


def diff_snapshots_by_var(args, save_path):
    # first, prepare the data
    differ = Differentiation(args, return_type=args.return_type)
    df = differ.diff_features(args.diff_type)
    # draw most frequent and least frequent features
    if args.pic_topk_feat > 0:
        # multiple pic draws
        _df = df[df["label"] == "coefficient of variance"]
        df_sort = _df.sort_values(by=['frequency'], ascending=[False])
        save_file = save_path[:-4]
        # pic1
        topk = args.pic_topk_feat
        save_path1 = f"{save_file}_top{topk}.pdf"
        topk_features = list(df_sort.iloc[:topk, :]["feature"])
        print("topk_features for df 1")
        print(topk_features)
        frames1 = []
        for topk_feature in topk_features:
            frames1.append(df[df['feature'] == topk_feature])
        df1 = pd.concat(frames1)
        print(df1)
        visualizer = Visualizer(args, df1, save_path1)
        visualizer.draw_pic(args.pic_type)
        # pic2
        save_path2 = f"{save_file}_bottom{topk}.pdf"
        bottomk_features = list(df_sort.iloc[-topk:, :]["feature"])
        print("bottom k_features for df 1")
        print(bottomk_features)
        frames2 = []
        for bottomk_feature in bottomk_features:
            frames2.append(df[df['feature'] == bottomk_feature])
        df2 = pd.concat(frames2)
        print(df2)
        visualizer = Visualizer(args, df2, save_path2)
        visualizer.draw_pic(args.pic_type)
        print(f"Actually save at {save_path1} and {save_path2}")
    else:
        # draw all features in one pic
        visualizer = Visualizer(args, df, save_path)
        visualizer.draw_pic(args.pic_type)


def visualize(args):
    # set path
    if args.feature_type == 'linguistic':
        abbreviation = ""
        for linguistic_feature in args.linguistic_features:
            abbreviation += linguistic_feature[:2]
        args.pic_name = f"{args.pic_name}_{abbreviation}"
    last_name = f"{args.pic_type}"
    middle_name = args.feature_type
    # special names
    if args.vis_task == "diff_snapshots_by_var":
        middle_name = f"diff{args.diff_type}"
    elif args.vis_task == "hc3_group":
        middle_name = f"group_{args.feature_group_type}"
    elif args.return_type == "feature_and_eval":
        last_name = f"{args.corr_type}_{args.pic_type}"

    save_path = os.path.join(args.pic_dir, f"{args.pic_name}_{middle_name}_{last_name}.pdf")
    # visualize for each demand
    if args.vis_task == "visualize_one_snapshot":
        visualize_one_snapshot(args, save_path)
    elif args.vis_task == "visualize_one_feature":
        visualize_one_feature(args, save_path)
    elif args.vis_task == 'diff_snapshots_by_var':
        diff_snapshots_by_var(args, save_path)


def extract_feature(_args, _time_qualifier, _pp_suffix):
    # First, prepare the data
    loader = DBLoader(_args)
    raw_answers, res_lst, save_json_path = loader.load_by_json(_time_qualifier, _pp_suffix)
    print("save_json_path")
    print(save_json_path)
    # Second, extract features
    extractor = Extractor(_args)
    aggregated_features = extractor.extract_features(raw_answers)
    # Finally, save
    if os.path.exists(save_json_path):
        old_aggregated_features = json.load(open(save_json_path, "r"))
        old_aggregated_features.update(aggregated_features)
        aggregated_features = old_aggregated_features
    with open(save_json_path, "w") as fout:
        json.dump(aggregated_features, fout)


if __name__ == '__main__':
    args = prepare_args()
    if args.task == "extract_feature":
        for time_qualifier in args.times:
            print(f"time_qualifier {time_qualifier}")
            for pp_suffix in args.pp_suffixes:
                extract_feature(args, time_qualifier, pp_suffix)
    elif args.task == "visualize":
        visualize(args)
