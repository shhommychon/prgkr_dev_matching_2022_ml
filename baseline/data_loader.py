import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

dataset0_dir = 'dataset0/'
dataset1_dir = 'dataset1/'

dataset0_train_dir = dataset0_dir + 'train/'
dataset0_test_dir = dataset0_dir + 'test/'

dataset1_train_dir = dataset1_dir + 'train/'
dataset1_test_dir = dataset1_dir + 'test/'

dataset0_classes = os.listdir(dataset0_train_dir)
dataset1_classes = os.listdir(dataset1_train_dir)

def create_dataset(dataset_dir):
    X, y = [], []
    labels = os.listdir(dataset_dir)
    for label in labels:
        file_list = os.listdir(dataset_dir + label + '/')
        for f in file_list:
            temp = pd.read_csv(dataset_dir + label + '/' + f)
            X.append(torch.from_numpy(temp.values))
            y.append(label)
    X = pad_sequence(X, batch_first=True)
    return X, y

def create_dataloader(batch_size=256, enclude_dataset1_y=False):
    dataset0_label_encoder = LabelEncoder()
    dataset0_label_encoder.fit(dataset0_classes)
    print("dataset0 labels: ", sorted(dataset0_label_encoder.classes_))
    
    # dataset0's training/test datasets & dataloader
    dataset0_X_train, dataset0_y_train = create_dataset(dataset0_train_dir)
    dataset0_y_train = dataset0_label_encoder.transform(dataset0_y_train)

    dataset0_X_test, dataset0_y_test = create_dataset(dataset0_test_dir)
    dataset0_y_test = dataset0_label_encoder.transform(dataset0_y_test)

    dataset0_train_dataset = TensorDataset(torch.tensor(dataset0_X_train).float(), torch.from_numpy(dataset0_y_train))
    dataset0_test_dataset = TensorDataset(torch.tensor(dataset0_X_test).float(), torch.from_numpy(dataset0_y_test))

    dataset0_train_dataloader = DataLoader(dataset0_train_dataset, batch_size=batch_size)
    dataset0_test_dataloader = DataLoader(dataset0_test_dataset, batch_size=batch_size)
    print("\tdataset0 train shape: ", next(iter(dataset0_train_dataloader))[0].shape)
    print("\tdataset0 test shape: ", next(iter(dataset0_test_dataloader))[0].shape)
    print()
    
    dataset1_label_encoder = LabelEncoder()
    dataset1_label_encoder.fit(dataset1_classes)
    print("dataset1 labels: ", sorted(dataset1_label_encoder.classes_))
    
    print()
    
    # dataset1's training datasets & dataloader
    dataset1_X_train, dataset1_y_train = create_dataset(dataset1_train_dir)
    dataset1_y_train = dataset1_label_encoder.transform(dataset1_y_train)

    dataset1_train_dataset = TensorDataset(torch.tensor(dataset1_X_train).float(), torch.from_numpy(dataset1_y_train))
    dataset1_train_dataloader = DataLoader(dataset1_train_dataset, batch_size=batch_size)
    
    print("\tdataset1 train shape: ", next(iter(dataset1_train_dataloader))[0].shape)
    
    if not enclude_dataset1_y: # 기본값, 문제 설정과 동일
        dataset1_test_X = []
        test_list = os.listdir(dataset1_test_dir)
        for folder_name in test_list:
            for file_name in os.listdir(os.path.join(dataset1_test_dir, folder_name)):
                temp = pd.read_csv(os.path.join(dataset1_test_dir, folder_name, file_name))
                dataset1_test_X.append(torch.from_numpy(temp.values))

        dataset1_test_X = pad_sequence(dataset1_test_X, batch_first=True)
        dataset1_test_dataset = TensorDataset(torch.tensor(dataset1_test_X).float())
        dataset1_test_dataloader = DataLoader(dataset1_test_dataset, batch_size=batch_size, shuffle=False)
        # ※ 위 코드 중 dataloader 부분의 `shuffle=False` 는 수정하면 안됩니다.
        
    else:
        dataset1_X_test, dataset1_y_test = create_dataset(dataset1_test_dir)
        dataset1_y_test = dataset1_label_encoder.transform(dataset1_y_test)

        dataset1_test_dataset = TensorDataset(torch.tensor(dataset1_X_test).float(), torch.from_numpy(dataset1_y_test))
        dataset1_test_dataloader = DataLoader(dataset1_test_dataset, batch_size=batch_size)
    
    print("\tdataset1 test shape: ", next(iter(dataset1_test_dataloader))[0].shape)
    
    return dataset0_train_dataloader, dataset0_test_dataloader, dataset1_train_dataloader, dataset1_test_dataloader
