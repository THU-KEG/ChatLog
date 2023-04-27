import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def temporal_correlation_coefficient(a, b):
    """
    :param a: ordered Temporal sequence
    :param b: ordered Temporal sequence
    :return: The First Order Temporal Correlation Coefficient
    """
    diff1 = np.diff(np.array(a))
    diff2 = np.diff(np.array(b))
    TCC = np.sum(diff1 * diff2) / (np.sqrt(np.sum(pow(diff1, 2))) * np.sqrt(np.sum(pow(diff2, 2))))
    return TCC


def get_tcc(args, df):
    pass


class Visualizer:
    def __init__(self, args, df, save_path):
        self.args = args
        self.df = df
        self.save_path = save_path

    def draw_bar(self):
        # 　设置字体
        sns.set_theme(style="ticks", font='Times New Roman')
        f, ax = plt.subplots(figsize=(7, 5))
        sns.despine(f)
        # ax = sns.barplot(x="model_names", y="times", hue="Type", data=df, dodge=False)
        sns.histplot(
            self.df,
            x="feature", hue="label", weights='frequency',
            # multiple="stack",
            # bins=9,
            # edgecolor=".3",
            # linewidth=.5,
        )
        # label
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Feature")
        print(list(set(self.df["feature"])))
        ax.set_xticklabels(list(set(self.df["feature"])), rotation=30)
        ax.grid(False)

    def draw_radar(self):
        # 在下面放入你得到的数据！！！！
        values = []
        feature = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            if row["label"][:3] == "avg":
                feature.append(row['feature'])
                values.append(row['frequency'])

        # 设置每个数据点的显示位置，在雷达图上用角度表示
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)

        # 拼接数据首尾，使图形中线条封闭
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        feature = np.concatenate((feature, [feature[0]]))

        plt.style.use('ggplot')
        # 绘图
        fig = plt.figure()
        # 设置为极坐标格式
        ax = fig.add_subplot(111, polar=True)
        # 绘制折线图
        ax.plot(angles, values, 'o-', linewidth=2)
        # 填充颜色
        ax.fill(angles, values, alpha=0.25)

    def draw_line(self):
        # 　设置字体
        sns.set_theme(style="ticks", font='Times New Roman')
        f, ax = plt.subplots(figsize=(12, 7))
        sns.despine(f)
        print("self.df")
        if self.args.return_type == "hc3_group":
            new_name = "Classifier"
        else:
            new_name = "feature"
        self.df.rename(columns={'feature': new_name}, inplace=True)
        print(self.df)
        sns.lineplot(
            x="condition",
            y="score",
            hue=new_name,
            data=self.df,
            # multiple="stack",
            # bins=9,
            # edgecolor=".3",
            # linewidth=.5,
            marker="o")
        # label
        if self.args.feature_group_type == "avg_acc":
            ax.set_ylabel("Accuracy (%)")
            ax.set_xlabel("Time (month-day)")
            # plt.legend(['qa', 'single', 'gltr', 'ppl'], title=new_name, ncol=2)
            plt.legend(['qa', 'single', 'gltr', 'ppl'], title=new_name, bbox_to_anchor=(0.97, 0.28), loc='lower right',
                       ncol=2)
        elif self.args.feature_group_type == "avg_prob":
            ax.set_ylabel("Average predicted probability (%)")
            ax.set_xlabel("Time (month-day)")
            plt.legend(['qa', 'single', 'gltr', 'ppl'], title=new_name, bbox_to_anchor=(0.97, 0.28), loc='lower right',
                       ncol=2)
        else:
            ax.set_ylabel("score")
            ax.set_xlabel("condition")
        a = list(set(self.df["condition"]))
        a.sort()
        print(a)
        ax.set_xticklabels(a, rotation=38)
        ax.grid(False)

    def draw_pic(self, pic_type):
        # draw distribution
        if pic_type == "bar":
            self.draw_bar()

        elif pic_type == "radar":
            self.draw_radar()

        elif pic_type == "line":
            self.draw_line()

        elif pic_type == "corr":
            self.draw_corr()
        # save
        plt.savefig(self.save_path, bbox_inches='tight')
        plt.close('all')
        print(f"finish at {self.save_path}")

    def draw_corr(self):
        print(self.df)
        print(f"##### shape #####")
        print(self.df.columns)
        print(self.df.shape)
        sns.set_theme(style="ticks", font='Times New Roman')
        if self.args.feature_type == "linguistic":
            self.df = self.df.drop(['ra_SSTo_C', 'ra_SOTo_C', 'ra_SXTo_C',
                                    'ra_OSTo_C', 'ra_OOTo_C', 'ra_OXTo_C',
                                    'ra_XSTo_C', 'ra_XOTo_C', 'ra_XXTo_C'], axis=1)
        print(f"##### new shape #####")
        print(self.df.columns)
        print(self.df.shape)
        # save all
        self.df.to_csv(f"{self.save_path[:-4]}_feats.csv")

        # for tcc, use different corr
        if self.args.corr_type == "tcc":
            corrmat = get_tcc(self.args, self.df)
        else:
            corrmat = self.df.corr(method=self.args.corr_type)
        f, ax = plt.subplots(figsize=(12, 9))
        # sns.heatmap(corrmat, vmax=.8, square=True)
        print(corrmat.shape)
        print(corrmat)
        corrmat.to_csv(f"{self.save_path[:-4]}.csv")
        sns.clustermap(corrmat, annot=True, square=True)
