import pandas as pd
from datetime import datetime
from collections import Counter
from datetime import timedelta
from proj2.tools import change_datatype, change_datatype_float, memory_usage
import time

path_to_data = "/media/raph/Elements/ml1/churn/"
transactions_chunk_size = 30000
todo = "train"

if todo == "train":
    max_date = datetime.strptime("20170201", "%Y%m%d").date()
else:
    max_date = datetime.strptime("20170301", "%Y%m%d").date()

train = pd.read_csv(path_to_data + todo+"_v2.csv")
train = pd.concat((train, pd.read_csv(path_to_data + todo+".csv")), axis=0,
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
    lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")

training.set_index('msno', inplace=True)

# reducing memory usage:
change_datatype(training)
change_datatype_float(training)


def reformat_transactions(df):
    df['transaction_date'] = df.transaction_date.apply(
        lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")
    df['membership_expire_date'] = df.membership_expire_date.apply(
        lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")
    df['payment_method_id'] = df.payment_method_id.apply(lambda x: int(x) if pd.notnull(x) else -1)
    boolean_indexes = df["transaction_date"] < max_date
    indexes_to_drop = [e[0] if e[1] else None for e in boolean_indexes.iteritems()]
    indexes_to_keep = set(range(df.shape[0])) - set(indexes_to_drop)
    df = df.take(list(indexes_to_keep))
    return df

training["total_number_of_transactions"] = 0

training["current_number_of_transactions"] = 0
training.drop(['current_number_of_transactions'], axis=1, inplace=True)


i = 0
df_iter = pd.read_csv(path_to_data + 'transactions.csv', low_memory=False, iterator=True,
                      chunksize=transactions_chunk_size)
print("starting iteration...")
for transactions in df_iter:
    print("i=" + str(i))
    transactions = reformat_transactions(transactions)
    user_count = Counter(transactions['msno']).most_common()
    user_count = pd.DataFrame(user_count)
    user_count.columns = ['msno', 'current_number_of_transactions']
    user_count.set_index('msno', inplace=True)
    training = pd.merge(left=training, right=user_count, how='left', left_index=True, right_index=True)
    training['current_number_of_transactions'] = training.current_number_of_transactions.apply(
        lambda x: int(x) if pd.notnull(x) else 0)
    training["total_number_of_transactions"] += training["current_number_of_transactions"]
    training.drop(['current_number_of_transactions'], axis=1, inplace=True)

    print("memory usage of training: ")
    print(memory_usage(training))
    print("memory usage of transactions: ")
    print(memory_usage(transactions))
    i += 1
print("end of iteration...")


i = 0
training.reset_index(inplace=True)
training_copy = training.copy()

df_iter = pd.read_csv(path_to_data + 'transactions.csv', low_memory=False, iterator=True,
                      chunksize=transactions_chunk_size)
print("starting iteration, looking for most recent transaction...")
for transactions in df_iter:
    print("i=" + str(i))

    reformat_transactions(transactions)
    recent_transactions = transactions.sort_values(['transaction_date']).groupby('msno').first()
    recent_transactions.reset_index(inplace=True)
    temp_training = pd.merge(left=training_copy, right=recent_transactions, how='right', on=['msno'], right_index=True)
    training = pd.concat((training, temp_training))
    training = training.sort_values(['transaction_date']).groupby('msno').first()

    i += 1

del training_copy


i = 0

df_iter = pd.read_csv(path_to_data + 'transactions.csv', low_memory=False, iterator=True,
                      chunksize=transactions_chunk_size)

training["price_per_day"] = training["actual_amount_paid"]/(training["payment_plan_days"]+0.01)
training["usual_price_per_day"] = 0

print("starting iteration, looking for usual payment method...")
for transactions in df_iter:
    print("i=" + str(i))
    i += 1

    transactions = reformat_transactions(transactions)
    transactions["current_price_per_day"] = transactions["actual_amount_paid"] / (transactions["payment_plan_days"] + 0.01)
    transactions = transactions.groupby("msno").sum()
    columns_to_keep = ["current_price_per_day"]
    transactions = transactions[columns_to_keep]

    training = pd.merge(left=training, right=transactions, how='left', left_index=True, right_index=True)

    training["usual_price_per_day"] += training["current_price_per_day"]
    training.drop(['current_price_per_day'], axis=1, inplace=True)

    if i > 0:
        break

training["usual_price_per_day"] /= (training["total_number_of_transactions"] + 0.01)

if todo == "test":
    time_delta = timedelta(days=-31)
else:
    time_delta = timedelta(days=0)


training["membership_expire_date"] = training.membership_expire_date.apply(lambda x: x + time_delta if not pd.isnull(x) else x)
training["transaction_date"] = training.transaction_date.apply(lambda x: x + time_delta if not pd.isnull(x) else x)

training['membership_expire_date'] = training.membership_expire_date.apply(lambda x: time.mktime(x.timetuple()) if not (pd.isnull(x) or type(x)==type(0.1)) else 0.0)
training['transaction_date'] = training.membership_expire_date.apply(lambda x: time.mktime(x.timetuple()) if not (pd.isnull(x) or type(x)==type(0.1)) else 0.0)

training.to_csv(path_or_buf=todo+"ing.csv")
