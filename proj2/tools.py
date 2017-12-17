import numpy as np
import pandas as pd
from datetime import datetime


def to_csv(y_pred, ids):
    import csv
    with open('my_answer.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['id', 'target'])
        for i in range(len(y_pred)):
            if len(y_pred) % 10000 == 0:
                print(str(100*float(i)/len(y_pred))+"% of the data copied in csv file")
            spamwriter.writerow([ids[i], y_pred[i]])


def log_loss(y, p):
    print("hello")
    p_2 = np.minimum(p, np.ones(len(p))-np.power(10.0, -8))
    p_2 = np.maximum(p_2, np.zeros(len(p))+np.power(10.0, -8))
    return -np.mean(y*np.log(p_2)+(1-y)*np.log(1-p_2))


def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)


def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)


def memory_usage(df):
    mem = df.memory_usage(index=True).sum()
    return mem / 1024**2, " MB"


def to_int(x):
    return int(x) if pd.notnull(x) else -1


def to_date(x):
    return datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN"
