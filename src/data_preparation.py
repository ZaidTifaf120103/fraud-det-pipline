#As we saw in the eda, we will apply the necessery transformation on our data so that we can fit it in the model

import pandas as pd
import numpy as np

import datetime as dt
import category_encoders as ce


def apply_woe(train, columns, target_col):
    woe = ce.WOEEncoder()

    for col in columns:
        X = train[col]
        y = train[target_col]

        new_col_name = f"{col}_WOE"
        train[new_col_name] = woe.fit_transform(X, y)

    return train

df = pd.read_csv(r"C:\Users\zaid2\Desktop\fraud-det-pipline\data\raw\fraudTrain.csv", index_col=0)

df['age'] = dt.date.today().year-pd.to_datetime(df['dob']).dt.year
df['hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
df['month'] = pd.to_datetime(df['trans_date_trans_time']).dt.month

df.drop(columns=["merchant", "first", "last", "street",
                   "unix_time", "trans_num"], inplace=True)

df["amt_log"] = np.log1p(df["amt"])

columns_to_encode = ["category", "state", "city", "job"]
target_column = "is_fraud"

df = apply_woe(df, columns_to_encode, target_column)

gender_mapping = {"F": 0, "M": 1}

df["gender_binary"] = df["gender"].map(gender_mapping)

intervals = [600, 1200, 1800, 2400, 3000, 3600]


def classify_frequency(freq):
    for i, c in enumerate(intervals):
        if freq <= c:
            return i

freq_enc_df = (df.groupby("cc_num").size())
freq_enc_df.sort_values(ascending=True)
df["cc_num_frequency"] = df["cc_num"].apply(lambda x: freq_enc_df[x])
df["cc_num_frequency_classification"] = df["cc_num_frequency"].apply(
    classify_frequency)

df.to_csv(r'C:\Users\zaid2\Desktop\fraud-det-pipline\data\processed\fraudTrain_processed.csv')

