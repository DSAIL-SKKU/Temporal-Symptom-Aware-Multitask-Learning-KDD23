import torch
import torch.nn as nn


def class_FScore(op, t, expt_type):
    FScores = []
    for i in range(expt_type):
        opc = op[t==i]
        tc = t[t==i]
        TP = (opc==tc).sum()
        FN = (tc>opc).sum()
        FP = (tc<opc).sum()

        GP = TP/(TP + FP  + 1e-8)
        GR = TP/(TP + FN + 1e-8)

        FS = 2 * GP * GR / (GP + GR + 1e-8)
        FScores.append(FS)
    return FScores

def gr_metrics(op, t):
    TP = (op==t).sum()
    FN = (t>op).sum()
    FP = (t<op).sum()

    GP = TP/(TP + FP)
    GR = TP/(TP + FN)

    FS = 2 * GP * GR / (GP + GR)

    OE = (t-op > 1).sum()
    OE = OE / op.shape[0]

    return GP, GR, FS, OE

def splits(df, dist_values):
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sort_values(by='label').reset_index(drop=True)
    df_test = df[df['label']==0][0:dist_values[0]].reset_index(drop=True)
    for i in range(1,5):
        df_test = df_test.append(df[df['label']==i][0:dist_values[i]], ignore_index=True)

    for i in range(5):
        df.drop(df[df['label']==i].index[0:dist_values[i]], inplace=True)

    df = df.reset_index(drop=True)
    return df, df_test

def make_31(five_class):
    if five_class!=0:
        five_class=five_class-1
    return five_class


