import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

tps = pd.read_csv(
    "/data/tsq/CK/pic/avg_HC3_all_pearson_corr_feats.csv", parse_dates=["time"]
)
print(tps.shape)
print(tps.columns)
print(len(tps))

if len(tps) > 90000:
    gps = tps.groupby("time").agg('mean').drop("Unnamed: 0", axis=1).drop(['04-21'],axis=0)
    # gps = tps.groupby("time").agg('mean').drop("Unnamed: 0", axis=1)
else:
    gps = tps.groupby("time").agg('mean').drop("Unnamed: 0", axis=1)
# gps = tps.groupby("time").agg('mean').drop("Unnamed: 0", axis=1).iloc[46:50,:]
# gps = tps.groupby("time").agg('mean').drop("Unnamed: 0", axis=1)
# print(gps)
print(gps.shape)
print(gps.columns)
rouges =  gps["rouge-1-p"]
# print(rouges['03-05':'03-23'])
# print(rouges['03-24':'05-14'])
# print(rouges['05-14':'06-27'])
# print(rouges['06-28':'08-14'])
# print(rouges['08-14':'10-02'])
# print(rouges[i:i+1]) for i in ['03-05','03-24','05-14','06-28','08-14']
starts = ['03-05','03-24','05-10','06-28','08-15']
ends = ['03-23','05-09','06-27','08-14','10-02']
sigma_mean_len = []
for i in range(5):
    rouge_i = rouges[starts[i]:ends[i]]
    # sigma_and_mean
    sigma_mean_len.append([rouge_i.std(),rouge_i.mean(),len(rouge_i)])
    
for i in range(5):
    print(sigma_mean_len[i])
    if i == 0:
        continue
    # stastitical test
    z_score = (sigma_mean_len[i][1] - sigma_mean_len[i-1][1]) / np.sqrt(sigma_mean_len[i][0]**2/sigma_mean_len[i][2] + sigma_mean_len[i-1][0]**2/sigma_mean_len[i-1][2])
    print(z_score)