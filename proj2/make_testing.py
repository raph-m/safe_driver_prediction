from datetime import datetime, timedelta
import time

from collections import Counter
import pandas as pd
import numpy as np

print("hey thats a new version")


def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if np.max(df[col]) <= 127 and np.min(df[col] >= -128):
            df[col] = df[col].astype(np.int8)
        elif np.max(df[col]) <= 32767 and np.min(df[col] >= -32768):
            df[col] = df[col].astype(np.int16)
        elif np.max(df[col]) <= 2147483647 and np.min(df[col] >= -2147483648):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)


def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)


def memory_usage(df):
    mem = df.memory_usage(index=True).sum()
    return mem / 1024 ** 2, " MB"


path_to_data = "../../churn/"
transactions_chunk_size = 10000000
todo = "test"
date_zero = datetime.strptime("20000101", "%Y%m%d").date()

if todo == "train":
    max_date = datetime.strptime("20170201", "%Y%m%d").date()
else:
    max_date = datetime.strptime("20170301", "%Y%m%d").date()

if todo == "train":
    train = pd.read_csv(path_to_data + todo + "_v2.csv")
else:
    train = pd.read_csv(path_to_data + "test_v2.csv")

members = pd.read_csv(path_to_data + "members_v3.csv")

training = pd.merge(left=train, right=members, how='left', on=['msno'])
print("merge train and members")
print(training.describe())
del train
del members

# changing type to int and putting -1 for missing values
training['city'] = training.city.apply(lambda x: int(x) if pd.notnull(x) else -1)
training['registered_via'] = training.registered_via.apply(lambda x: int(x) if pd.notnull(x) else -1)
training['bd'] = training.bd.apply(lambda x: int(x) if pd.notnull(x) else -1)
training['bd'] = training.bd.apply(lambda x: x if (10 < x < 100) else -1)

# encode gender
genders_encoding = {'male': 0, 'female': 1}
training['gender'] = training.gender.apply(lambda x: genders_encoding[x] if pd.notnull(x) else -1)

# changing date formats
training['registration_init_time'] = training.registration_init_time.apply(
    lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else date_zero)

training.set_index('msno', inplace=True)

# reducing memory usage:
change_datatype(training)
change_datatype_float(training)


def reformat_transactions(df):
    df['transaction_date'] = df.transaction_date.apply(
        lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else date_zero)
    df['membership_expire_date'] = df.membership_expire_date.apply(
        lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else date_zero)
    df['payment_method_id'] = df.payment_method_id.apply(lambda x: int(x) if pd.notnull(x) else -1)
    boolean_indexes = df["transaction_date"] > max_date
    indexes_to_drop = [e[0] if e[1] else None for e in boolean_indexes.iteritems()]
    indexes_to_keep = set(range(df.shape[0])) - set(indexes_to_drop)
    df = df.take(list(indexes_to_keep))
    return df


def iterate_on_transactions_1(t, version=1):
    if version == 1:
        print("start iteration on transactions...")
        path_to_csv = path_to_data + 'transactions.csv'
    else:
        print("start iterate on transactions_v2...")
        path_to_csv = path_to_data + 'transactions_v2.csv'

    i = 0
    df_iter = pd.read_csv(path_to_csv, low_memory=False, iterator=True,
                          chunksize=transactions_chunk_size)

    print("computing total number of transactions...")
    for transactions in df_iter:
        print("i=" + str(i))
        i += 1
        transactions = reformat_transactions(transactions)
        user_count = Counter(transactions['msno']).most_common()
        user_count = pd.DataFrame(user_count)
        user_count.columns = ['msno', 'current_number_of_transactions']
        user_count.set_index('msno', inplace=True)
        t = pd.merge(left=t, right=user_count, how='left', left_index=True, right_index=True)
        t['current_number_of_transactions'] = t.current_number_of_transactions.apply(
            lambda x: int(x) if pd.notnull(x) else 0)
        t["total_number_of_transactions"] += t["current_number_of_transactions"]
        t.drop(['current_number_of_transactions'], axis=1, inplace=True)

    print("end of iteration...")
    return t

training["total_number_of_transactions"] = 0
u = training.copy()
u = iterate_on_transactions_1(u, version=1)
u = iterate_on_transactions_1(u, version=2)
print("after first iteration: ")
print(u.describe())


training = u
del u


def iterate_on_transactions_2(t):
    print("start iterate on transactions...")
    path_to_csv = path_to_data + 'transactions.csv'

    i = 0
    t_copy = t.copy()
    t_copy.reset_index(inplace=True)

    df_iter = pd.read_csv(path_to_csv, low_memory=False, iterator=True,
                          chunksize=transactions_chunk_size)
    for transactions in df_iter:
        print("i=" + str(i))
        t.reset_index(inplace=True)
        reformat_transactions(transactions)
        recent_transactions = transactions.sort_values(['transaction_date']).groupby('msno').first()
        recent_transactions.reset_index(inplace=True)
        temp_t = pd.merge(left=t_copy, right=recent_transactions, how='inner', on=['msno'])
        t = pd.concat((t, temp_t))
        t = t.sort_values(['transaction_date'], ascending=False).groupby('msno').first()
        i += 1

    print("end of iteration...")

    print("start iterate on transactions_v2...")
    path_to_csv = path_to_data + 'transactions_v2.csv'

    df_iter = pd.read_csv(path_to_csv, low_memory=False, iterator=True,
                          chunksize=transactions_chunk_size)
    i = 0

    for transactions in df_iter:
        print("i=" + str(i))
        t.reset_index(inplace=True)
        reformat_transactions(transactions)
        recent_transactions = transactions.sort_values(['transaction_date']).groupby('msno').first()
        recent_transactions.reset_index(inplace=True)
        temp_t = pd.merge(left=t_copy, right=recent_transactions, how='inner', on=['msno'])
        t = pd.concat((t, temp_t))
        t = t.sort_values(['transaction_date'], ascending=False).groupby('msno').first()
        i += 1

    return t

u = training.copy()
u = iterate_on_transactions_2(u)
print("after second iteration: ")
print(u.describe())

training = u


def iterate_on_transactions_3(t, version=1):
    if version == 1:
        print("start iterate on transactions...")
        path_to_csv = path_to_data + 'transactions.csv'
    else:
        print("start iterate on transactions_v2...")
        path_to_csv = path_to_data + 'transactions_v2.csv'
    i = 0

    df_iter = pd.read_csv(path_to_csv, low_memory=False, iterator=True,
                          chunksize=transactions_chunk_size)

    print("starting iteration, looking for usual price per day...")
    for transactions in df_iter:
        print("i=" + str(i))
        i += 1
        transactions = reformat_transactions(transactions)
        transactions["current_price_per_day"] = transactions["actual_amount_paid"] / (
            transactions["payment_plan_days"] + 0.01)
        transactions = transactions.groupby("msno").sum()
        columns_to_keep = ["current_price_per_day"]
        transactions = transactions[columns_to_keep]

        t = pd.merge(left=t, right=transactions, how='left', left_index=True, right_index=True)

        t["current_price_per_day"] = t.current_price_per_day.apply(lambda x: int(x) if pd.notnull(x) else 0)
        t["usual_price_per_day"] += t["current_price_per_day"]
        t.drop(['current_price_per_day'], axis=1, inplace=True)

    return t

u = training.copy()
u["usual_price_per_day"] = 0

u["price_per_day"] = u["actual_amount_paid"] / (u["payment_plan_days"] + 0.01)
u = iterate_on_transactions_3(u, version=1)
u = iterate_on_transactions_3(u, version=2)
print("after second iteration: ")
print(u.describe())

training = u

u = training.copy()
u["price_per_day"] = u.price_per_day.apply(lambda x: x if pd.notnull(x) else 0.0)
u["usual_price_per_day"] /= (u["total_number_of_transactions"] + 0.01)
u["price_per_day_diff"] = u["price_per_day"] - u["usual_price_per_day"]

training = u

u = training.copy()
if todo == "test":
    time_delta = timedelta(days=-31)
else:
    time_delta = timedelta(days=0)

u["membership_expire_date"] = u.membership_expire_date.apply(lambda x: x + time_delta if not pd.isnull(x) else x)
u["transaction_date"] = u.transaction_date.apply(lambda x: x + time_delta if not pd.isnull(x) else x)

u['membership_expire_date'] = u.membership_expire_date.apply(
    lambda x: time.mktime(x.timetuple()) if not (pd.isnull(x) or type(x) == type(0.1)) else 0.0)
u['transaction_date'] = u.membership_expire_date.apply(
    lambda x: time.mktime(x.timetuple()) if not (pd.isnull(x) or type(x) == type(0.1)) else 0.0)

print("just before to csv: ")
print(u.describe())

training.to_csv(path_or_buf=todo + "ing.csv")
