# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# MODELS
# 1. bert-base-uncased -> (1e-5)
# 2. distilbert-base-uncased -> (1e-5)
# 3. mrm8488/bert-tiny-finetuned-sms-spam-detection -> (2.5e-4)
# 4. prajjwal1/bert-tiny -> (2.5e-4)
# 5. microsoft/deberta-v3-small -> (2e-5)

# #3 converges very fast

MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda"
TRAIN_PATH = "/kaggle/input/utkmls-twitter-spam-detection-competition/train.csv"
TEST_PATH = "/kaggle/input/utkmls-twitter-spam-detection-competition/test.csv"

id2label = {0: "Quality", 1: "Spam"}
label2id = {"Quality": 0, "Spam": 1}

class TwitterDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx, 0]
        if self.df.iloc[idx, 1] in ["Quality", "Spam"]:
            label = label2id[self.df.iloc[idx, 1]]
        else:
            label = label2id[self.df.iloc[idx, 2]]
        return text, label


# Using the first and last column
train_df = pd.read_csv(TRAIN_PATH, usecols=[0, 6, 7])
test_df = pd.read_csv(TEST_PATH, usecols=[1])

train_set = train_df.sample(frac=0.9,random_state=42)
val_set = train_df.drop(train_set.index)
test_set = test_df

twitter_dataset = TwitterDataset(train_df)

print(twitter_dataset[0])


device = torch.device(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id)

model.to(device)

optimizer = AdamW(model.parameters(), lr = 1e-5, betas = (0.9, 0.95), weight_decay = 0.1)

itr = 1
print_point = 100
d_len = len(twitter_dataset)
epochs = 2
total_loss = 0
total_len = 0
total_correct = 0


# Model Training

BATCH = 8

model.train()
train_loader = DataLoader(twitter_dataset, batch_size=BATCH, shuffle=True, num_workers=2)

padding_idx = tokenizer.encode('[PAD]')
print(padding_idx)



for epoch in range(epochs):
    for text, label in train_loader:
        optimizer.zero_grad()
        
        text = list(text)
        sample = tokenizer.batch_encode_plus(text, max_length=512, padding="max_length", truncation=True, return_tensors='pt')

        labels = torch.tensor(label)
        for k, v in sample.items():
            sample[k] = v.to(device)
        labels = labels.to(device)
        
        outputs = model(**sample, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        


        pred = torch.argmax(F.softmax(logits), dim=1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if itr % print_point == 0:
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr/(d_len/BATCH), total_loss/print_point, total_correct/total_len))
            total_loss = 0
            total_len = 0
            total_correct = 0

        itr+=1



# Save fine-tuned model

model.save_pretrained('/kaggle/working/model.pt')


# Model Evaluation

model.eval()

eval_dataset = TwitterDataset(val_set)
eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=2)

total_loss = 0
total_len = 0
total_correct = 0

for text, label in eval_loader:
    text = list(text)
    sample = tokenizer.batch_encode_plus(text, max_length=512, padding="max_length", truncation=True, return_tensors='pt')
    labels = torch.tensor(label)
    for k, v in sample.items():
        sample[k] = v.to(device)
    labels = labels.to(device)
    outputs = model(**sample, labels=labels)
    logits = outputs.logits

    pred = torch.argmax(F.softmax(logits), dim=1)
    correct = pred.eq(labels)
    total_correct += correct.sum().item()
    total_len += len(labels)

print('Test accuracy: ', total_correct / total_len)


# submission csv
import csv
FILE_NAME = "submission.csv"
idx = -1

with open(FILE_NAME, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for text in test_df:
        if idx == -1:
            idx += 1
            continue
        sample = tokenizer.batch_encode_plus([text], max_length=512, padding="max_length", truncation=True, return_tensors='pt')
        for k, v in sample.items():
            sample[k] = v.to(device)
        output = model(**sample)
        logits = output.logits
        pred = torch.argmax(F.softmax(logits), dim=1)
        writer.writerow([idx, id2label[pred]])
        idx += 1