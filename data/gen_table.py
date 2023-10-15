import argparse
import json
semantics = ['GSM8K', 'WSD', 'SQuAD', 'ReAding', 'WNLI', 'Cola', 'WordContext', 'TextEntail']
pragmatics = ['Aggression', 'AggressionPer', 'Spam', 'Sarcasm''ColBERT', 'TweetSent', 'TweetEmoji', 'Unhealthy',
              'UnhealthyPer', 'TweetStance', 'GoEmoPer3', 'GoEmoPer0', 'GoEmoPer1', 'GoEmoPer2', 'GoEmo', ]
accs = [ k.lower() for k in ['WordContext', 'WNLI', 'SQuAD', 'GSM8K']]
category2task = {
    "Sentiment":["ColBERT", "TweetEmoji", "TweetSent", "TweetStance", "GoEmo", 'GoEmoPer3', 'GoEmoPer0', 'GoEmoPer1', 'GoEmoPer2'],
    "Classification":["CoLa", "Aggression", "AggressionPer", "WordContext", "Spam", "Sarcasm"],
    "NLI":["TextEntail", "WNLI"],
    "MRC":["SQuAD", "ReAding"],
    "Reasoning": ["GSM8K"],
}
task2category = {}
for category, lst in category2task.items():
    for task in lst:
        task2category[task.lower()] = category

def prepare_args():
    parser = argparse.ArgumentParser(description='generate table with t-test')
    parser.add_argument('--add_metric', default=False, action="store_true", help="add metric to the table")
    parser.add_argument('--data_file', help='Where to load', default='/home/tsq/ChatLog/data/monthly_res.json')
    args = parser.parse_args()
    return args


def gen_table(args):
    # load data
    with open(args.data_file, 'r') as fin:
        data_dict = json.load(fin)
        print(data_dict)
        
    final_table_str = ""
    for level, lst in enumerate(category2task.values()):
        print(f"level {level}")
        for i, task_name in enumerate(lst):
            cate = task2category[task_name.lower()]
            color_type = "deep" if i % 2 == 0 else "shallow"
            if args.add_metric:         
                metric = "Accuracy" if task_name.lower() in accs else "F1"
                task_str = "\\rowcolor{'" + color_type + str(level+1) +"'}" + f" {task_name} &" +  f" {cate} &" +  f" {metric} &"
            else:
                task_str = "\\rowcolor{'" + color_type + str(level+1) +"'}" + f" {task_name} &" +  f" {cate} &"
            prev_month = ""
            first_month = list(data_dict.keys())[0]
            for month, month_dict in data_dict.items():
                # print(f"month {month}")
                # get data
                # print(month_dict)
                task_value = month_dict[task_name.lower()] * 100
                v_str = f" {task_value:.2f} &"
                # judge the improvement
                if month not in [first_month, "SOTA"]:
                    prev_value = data_dict[prev_month][task_name.lower()] * 100
                    improve = (task_value - prev_value) / prev_value
                    if improve > 0.1:
                        v_str = f" {task_value:.2f}" + " $\\uparrow$ &"
                    elif improve < -0.1:
                        v_str = f" {task_value:.2f}" + " $\\downarrow$ &"
                    else:
                        v_str = f" {task_value:.2f}" + " $\\sim$ &"
                task_str += v_str
                        
                prev_month = month
            
            task_str = task_str.strip("&") + "\\\\"
            final_table_str += task_str + "\n"
    print(final_table_str)


if __name__ == '__main__':
    args = prepare_args()
    gen_table(args)