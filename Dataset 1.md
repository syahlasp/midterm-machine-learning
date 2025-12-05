# Dataset-1
# Rename file

# download semua file dari gdrive dibawah lalu buat folder (ex. midterm_folder)
!pip install -q gdown

!gdown --folder 1JvI5xhPfN3VmjpWYZk9fCHG41xG697um -O midterm_folder

# memuat data transaksi training dan testing
import pandas as pd
import polars as pl

BASE_PATH = "midterm_folder"

train_transaction = pl.read_csv(f"{BASE_PATH}/train_transaction.csv", truncate_ragged_lines=True)
test_transaction  = pl.read_csv(f"{BASE_PATH}/test_transaction.csv", truncate_ragged_lines=True)

print(train_transaction.shape) #590540 row dan 393 feature + 1 target
print(test_transaction.shape) #506691 row dan 393 feature

# menampilkan data training
train_transaction.head() #isFraud adalah target featurenya

# menampilkan data testing
test_transaction.head() #uji data baru, setelah pembuatan model machine learningnya
