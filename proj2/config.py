import os
from os.path import join, dirname

from dotenv import load_dotenv

# DOTENV
dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)

ENV_NAME = os.environ.get("ENV_NAME")

path_to_data = "../../churn/"
default_transaction_chunk_size = 6000000
default_userlogs_chunk_size = 6000000

if ENV_NAME == "raph":
    path_to_data = "/media/raph/Elements/ml1/churn/"
    default_transaction_chunk_size = 30000
    default_userlogs_chunk_size = 30000
    print("running on raph environment...")

if ENV_NAME == "vm":
    print("running on cloud environment...")
    path_to_data = "../../churn/"
    default_transactions_chunk_size = 6000000
    default_userlogs_chunk_size = 6000000

