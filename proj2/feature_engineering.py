import pandas as pd
from datetime import datetime
from collections import Counter
from datetime import timedelta
from tools import change_datatype, change_datatype_float, memory_usage
import time
from config import path_to_data, default_transactions_chunk_size


def make_csv(todo="train", path_to_data=path_to_data, transactions_chunk_size=default_transactions_chunk_size):

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

    def iterate_on_transactions(t, version=1):

        if version == 1:
            print("start iterate on transactions...")
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

            print("memory usage of t: ")
            print(memory_usage(t))
            print("memory usage of transactions: ")
            print(memory_usage(transactions))

        print("end of iteration...")

        i = 0
        t.reset_index(inplace=True)
        t_copy = t.copy()

        df_iter = pd.read_csv(path_to_csv, low_memory=False, iterator=True,
                              chunksize=transactions_chunk_size)
        print("starting iteration, looking for most recent transaction...")
        for transactions in df_iter:
            print("i=" + str(i))

            reformat_transactions(transactions)
            recent_transactions = transactions.sort_values(['transaction_date']).groupby('msno').first()
            recent_transactions.reset_index(inplace=True)
            temp_t = pd.merge(left=t_copy, right=recent_transactions, how='inner', on=['msno'])
            t = pd.concat((t, temp_t))
            t = t.sort_values(['transaction_date'], ascending=False).groupby('msno').first()

            print("memory usage of t: ")
            print(memory_usage(t))
            print("memory usage of transactions: ")
            print(memory_usage(transactions))

            i += 1

        del t_copy

        i = 0

        df_iter = pd.read_csv(path_to_csv, low_memory=False, iterator=True,
                              chunksize=transactions_chunk_size)

        t["price_per_day"] = t["actual_amount_paid"]/(t["payment_plan_days"]+0.01)

        print("starting iteration, looking for usual price per day...")
        for transactions in df_iter:
            print("i=" + str(i))
            i += 1

            transactions = reformat_transactions(transactions)
            transactions["current_price_per_day"] = transactions["actual_amount_paid"] / (transactions["payment_plan_days"] + 0.01)
            transactions = transactions.groupby("msno").sum()
            columns_to_keep = ["current_price_per_day"]
            transactions = transactions[columns_to_keep]

            t = pd.merge(left=t, right=transactions, how='left', left_index=True, right_index=True)

            t["current_price_per_day"] = t.current_price_per_day.apply(lambda x: int(x) if pd.notnull(x) else 0)
            t["usual_price_per_day"] += t["current_price_per_day"]
            t.drop(['current_price_per_day'], axis=1, inplace=True)

        return t

    training["total_number_of_transactions"] = 0
    training["usual_price_per_day"] = 0

    training = iterate_on_transactions(training, version=1)
    training = iterate_on_transactions(training, version=2)

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


if __name__ == "__main__":
    import sys
    if len(sys.argv) is 1:
        make_csv()
    if len(sys.argv) is 2:
        todo = sys.argv[1]
        make_csv(todo=todo)
    if len(sys.argv) is 3:
        todo = sys.argv[1]
        transactions_chunk_size = sys.argv[2]
        make_csv(todo=todo, transactions_chunk_size=int(transactions_chunk_size))
    else:
        print("error: more than 2 arguments")

