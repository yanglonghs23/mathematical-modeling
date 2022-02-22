from datetime import date
import pandas as pd
import quandl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras import layers, Sequential,models
import tensorflow as tf
import pickle

df = pd.read_csv("../data/input.csv")  # 删除前30天的 黄金比特币预测结果.csv
xt = list(df["usd"])
xt_1 = list(df["usd_nextday_predict"])
yt = list(df["Value"])
yt_1 = list(df["BCHAIN-MKPRU_nextday_predict"])
beta_c = 8  # 1-100
beta_b =15 #1-100

for i in range(len(xt)-1):
    if xt[i+1] == 0:
        xt[i+1] = xt[i]
        if xt_1[i+1] == 0:
            xt_1[i+1] = xt[i+1]

def run(erfa_b = 2,erfa_c = 1):
    result = [1000,0,0] # crash gold bitcion
    for i in range(len(xt)-1):
        if xt[i+1]==xt_1[i+1] and yt_1[i+1] > (1+beta_b*0.01)*yt[i+1]:
            result[2]=result[2]*(1+(yt[i+1]-yt[i])/yt[i]) + \
                         result[0]*(1-erfa_b*0.01)
            result[0]=0
#             print(f"hold 黄金,美->比")
        elif xt[i+1]==xt_1[i+1] and yt_1[i+1] < yt[i+1]*(1-beta_b*0.01):
            result[0]=result[0]+result[2]*(1+(yt[i+1]-yt[i])/yt[i])*(1-erfa_b*0.01)
            result[2]=0
#             print(f"hold 黄金，比->美")
        elif xt[i+1]*(1-beta_c*0.01) <= xt_1[i+1] <= (1+beta_c*0.01)*xt[i+1] and yt[i+1]*(1-beta_b*0.01) <= yt_1[i+1] <= (1+beta_b*0.01)*yt[i+1]:
            result[1] = result[1]*(1+(xt[i+1]-xt[i])/xt[i])
            result[2] = result[2]*(1+(yt[i+1]-yt[i])/yt[i])
#             print(f"全hold:{result}")

        elif xt_1[i+1] > (1+beta_c*0.01)*xt[i+1] and (xt_1[i+1]-xt[i+1])*yt[i+1] >= (yt_1[i+1]-yt[i+1])*xt[i+1]:
            result[1] = result[1]*(1+(xt[i+1]-xt[i])/xt[i]) + \
                        result[2]*(1+(yt[i+1]-yt[i])/yt[i])*(1-erfa_b*0.01-erfa_c*0.01) +\
                        result[0]*(1-erfa_c*0.01)
            result[0] = 0
            result[2] = 0
#             print(f"全黄金:{result}")
        elif yt_1[i+1] > (1+beta_b*0.01)*yt[i+1] and (yt_1[i+1]-yt[i+1])*xt[i+1] >= (xt_1[i+1]-xt[i+1])*yt[i+1]:
            result[2] = result[2]*(1+(yt[i+1]-yt[i])/yt[i]) + \
                        result[0]*(1-erfa_b*0.01) + \
                        result[1]*(1+(xt[i+1]-xt[i])/xt[i])*(1-erfa_b*0.01-erfa_c*0.01)
            result[0] = 0
            result[1] = 0
#             print(f"全换成比特币:{result}")
        elif xt_1[i+1] < xt[i+1]*(1-beta_c*0.01) and yt_1[i+1] < yt[i+1]*(1-beta_b*0.01):
            result[0] = result[0] + \
                        result[1]*(1+(xt[i+1]-xt[i])/xt[i])*(1-erfa_b*0.01) +\
                        result[2]*(1+(yt[i+1]-yt[i])/yt[i])*(1-erfa_c*0.01)
            result[1] = 0
            result[2] = 0
#             print(f"全美金:{result}")

    return result

def main():
    global beta_b
    global beta_c
    test = dict()
    for i in range(6, 100+1):
        beta_b = i
        for j in range(4, 100+1):
            beta_c = j
            test[str((i,j))] = sum(run())
            print(test[str((i,j))])
    with open("../data/result.pkl","wb") as f:
        pickle.dump(test,f)


if __name__ == '__main__':
    main()