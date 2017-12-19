# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime
from collections import Counter
from datetime import timedelta
import time
import numpy as np


# In[2]:


def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and (np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and (np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and (np.min(df[col] >= -2147483648))):
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


# In[3]:


path_to_data = "../../churn/"
transactions_chunk_size = 3000000
todo = "test"
date_zero = datetime.strptime("20000101", "%Y%m%d").date()

# In[4]:


if todo == "train":
    max_date = datetime.strptime("20170201", "%Y%m%d").date()
else:
    max_date = datetime.strptime("20170301", "%Y%m%d").date()

if todo == "train":
    train = pd.read_csv(path_to_data + todo + "_v2.csv")
    train = pd.concat((train, pd.read_csv(path_to_data + todo + ".csv")), axis=0,
                      ignore_index=True).reset_index(drop=True)
else:
    train = pd.read_csv(path_to_data + "test_v2.csv")
    train = pd.concat((train, pd.read_csv(path_to_data + "test.csv")), axis=0,
                      ignore_index=True).reset_index(drop=True)

members = pd.read_csv(path_to_data + "members_v3.csv")

training = pd.merge(left=train, right=members, how='left', on=['msno'])
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

# In[5]:


training.head()


# In[6]:


def reformat_transactions(df):
    start_time = time.time()
    df['transaction_date'] = df.transaction_date.apply(
        lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else date_zero)
    df['membership_expire_date'] = df.membership_expire_date.apply(
        lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else date_zero)
    df['payment_method_id'] = df.payment_method_id.apply(lambda x: int(x) if pd.notnull(x) else -1)
    boolean_indexes = df["transaction_date"] > max_date
    start_loop = time.time()
    indexes_to_drop = [e[0] if e[1] else None for e in boolean_indexes.iteritems()]
    end_loop = time.time()
    indexes_to_keep = set(range(df.shape[0])) - set(indexes_to_drop)
    df = df.take(list(indexes_to_keep))
    end_time = time.time()
    return df


# In[7]:


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

        if i > 5:
            break

    print("end of iteration...")
    return t


# In[8]:


training["total_number_of_transactions"] = 0
t = training.copy()
t = iterate_on_transactions_1(t, version=1)
t = iterate_on_transactions_1(t, version=2)

# In[9]:


t.head()

# In[10]:


training = t
del t


# In[11]:


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
        print("recent_transactions.columns")
        print(recent_transactions.columns)
        temp_t = pd.merge(left=t_copy, right=recent_transactions, how='inner', on=['msno'])
        print("temp_t.columns")
        print(temp_t.columns)
        t = pd.concat((t, temp_t))
        print("t.columns")
        print(t.columns)
        t = t.sort_values(['transaction_date'], ascending=False).groupby('msno').first()
        i += 1

        if i > 3:
            break

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
        print("recent_transactions.columns")
        print(recent_transactions.columns)
        temp_t = pd.merge(left=t_copy, right=recent_transactions, how='inner', on=['msno'])
        print("temp_t.columns")
        print(temp_t.columns)
        t = pd.concat((t, temp_t))
        print("t.columns")
        print(t.columns)
        t = t.sort_values(['transaction_date'], ascending=False).groupby('msno').first()
        i += 1

        if i > 3:
            break

    return t


# In[12]:


t = training.copy()
t = iterate_on_transactions_2(t)

# In[13]:


t.head()

# In[14]:


training = t


# In[15]:


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

        if i > 3:
            break

    return t


# In[16]:


t = training.copy()
t["usual_price_per_day"] = 0

t["price_per_day"] = t["actual_amount_paid"] / (t["payment_plan_days"] + 0.01)
t = iterate_on_transactions_3(t, version=1)
t = iterate_on_transactions_3(t, version=2)

# In[17]:


t.head()

# In[18]:


training = t

# In[19]:


t = training.copy()
t["price_per_day"] = t.price_per_day.apply(lambda x: x if pd.notnull(x) else 0.0)
t["usual_price_per_day"] /= (t["total_number_of_transactions"] + 0.01)
t["price_per_day_diff"] = t["price_per_day"] - t["usual_price_per_day"]
t.head()

# In[20]:


training = t

# In[21]:


t = training.copy()
if todo == "test":
    time_delta = timedelta(days=-31)
else:
    time_delta = timedelta(days=0)

t["membership_expire_date"] = t.membership_expire_date.apply(lambda x: x + time_delta if not pd.isnull(x) else x)
t["transaction_date"] = t.transaction_date.apply(lambda x: x + time_delta if not pd.isnull(x) else x)

t['membership_expire_date'] = t.membership_expire_date.apply(
    lambda x: time.mktime(x.timetuple()) if not (pd.isnull(x) or type(x) == type(0.1)) else 0.0)
t['transaction_date'] = t.membership_expire_date.apply(
    lambda x: time.mktime(x.timetuple()) if not (pd.isnull(x) or type(x) == type(0.1)) else 0.0)

# In[22]:


t.describe()

training.to_csv(path_or_buf=todo + "ing.csv")


