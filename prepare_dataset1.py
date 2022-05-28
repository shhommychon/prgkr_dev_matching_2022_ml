# MindBigData MNIST dataset ver 1.06
# http://mindbigdata.com/opendb/index.html


# 압축파일 다운로드
from urllib import request
mindbigdata_url = "http://mindbigdata.com/opendb/MindBigData-IN-v1.06.zip"
mindbigdata_zip = "MindBigData-IN-v1.06.zip"
request.urlretrieve(mindbigdata_url, mindbigdata_zip)


# 압축파일 압축해제
import zipfile
with zipfile.ZipFile(mindbigdata_zip, 'r') as zip_file:
    zip_file.extractall("./")


# 압축파일 삭제
import os
os.remove(mindbigdata_zip)
os.mkdir("temp_dataset1")


# csv 파일 생성
import pandas as pd
df = pd.read_csv("IN.txt", delimiter='\t', header=None)
df.columns = "id", "event", "device", "channel", "code", "size", "data"

df = df.pivot_table(index=["event", "code", "size"], columns=["channel"], values=["data"], aggfunc=lambda x: ' '.join(x)).reset_index().dropna()
df.columns = "event", "code", "size", "AF3", "AF4", "PZ", "T7", "T8"

df = df[df["size"].isin([252, 256])] # 임의로 정한 길이

for _, row in df.iterrows():
    pd.DataFrame.from_dict({
        "AF3": [ float(n) for n in row["AF3"].split(',')[:252] ],
        "AF4": [ float(n) for n in row["AF4"].split(',')[:252] ],
        "T7":  [ float(n) for n in row["T7"].split(',')[:252] ],
        "T8":  [ float(n) for n in row["T8"].split(',')[:252] ],
        "PZ":  [ float(n) for n in row["PZ"].split(',')[:252] ],
    }).to_csv(f"temp_dataset1/MindBigData_Mnist_Insight_class{row['code']}_{row['event']}.csv", index=False)


# 일부만 저장
import os
os.mkdir("dataset1")
os.mkdir("dataset1/train")
os.mkdir("dataset1/test")

from glob import glob
import numpy as np
np.random.seed(53)
from shutil import copy

my_random_file_counts = 129, 377, 93, 730, 351, 196, 80, 92, 68, 68
for i in range(10):
    os.mkdir(f"dataset1/train/class{i}")
    csv_lists = glob(f"temp_dataset1/*_class{i}_*.csv")
    csv_arr = np.random.choice(csv_lists, size=my_random_file_counts[i], replace=False)
    for c in csv_arr:
        copy(c, f"dataset1/train/class{i}/{c.split('/')[-1]}")

for i in range(10):
    os.mkdir(f"dataset1/test/class{i}")
    csv_lists = glob(f"dataset1/train/class{i}/*.csv")
    csv_arr = np.random.choice(csv_lists, size=44, replace=False)
    for c in csv_arr:
        os.rename(c, f"dataset1/test/class{i}/{c.split('/')[-1]}")

for folder in os.listdir("dataset1/train"):
    print(folder)
    print('\t', len(os.listdir(f"dataset1/train/{folder}")))
    print('\t', len(os.listdir(f"dataset1/test/{folder}")))


# 불필요 데이터 삭제
from shutil import rmtree
os.remove("IN.txt")
rmtree("temp_dataset1")
