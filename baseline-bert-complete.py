import torch
import csv
import sys
import json
import tqdm
import time
import pandas as pd
import numpy as np
import datasets

from torch.optim import AdamW
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from os import walk

ENV_PATH = f'./dataset/{sys.argv[1]}'
TARGET_LABEL = sys.argv[2]
BATCH_SIZE = 8
MAX_EPOCH = 4096
PATIENCE = 10

# Define Work class for extraction

class Work:
    def __init__(self, i, t, c, l):
        self.title = t
        self.content = c
        self.labels = l
        self.id = i
    def __str__(self):
        return f"id: \"{self.id}\"\ntitle: \"{self.title}\"\ncontent: \"{self.content}\"\nlabels: {self.labels}\n"
    
def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data

def load_csv(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as file:
        rows = csv.reader(file)
        for row in rows:
            label = {"id": row[0], "name": row[1]}
            data.append(label)
    return data

def add_str(datas):
    string = ""
    for data in datas:
        t_str = data['body']
        t_str = t_str.replace(u'\u3000', u'')
        string += t_str
    return string

def create_work(data, labels):
    if data['labels'] != None:
        t_labels = create_label_vector(labels, data['labels'])
    else:
        t_labels = create_label_vector(labels, [""])
    
    w = Work(data['id'], data['metadata']['title'], add_str(data['content'])[:512], t_labels)
    return w

def create_label_vector(total_labels, target_labels):
    return_label = []
    for i, label in enumerate(total_labels):
        for t_label in target_labels:
            if label['name'] == t_label:
                return_label.append(label['id'])
    return return_label

tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', mecab_kwargs={"mecab_dic": "unidic", "mecab_option": None})
def preprocess_data(work):
    text = work['content']
    encoding = tokenizer(text, max_length=512, truncation=True, padding="max_length")
    encoding['labels'] = work['label']
    return encoding

def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    labels = load_csv(f"./dataset/label_list.csv")

    unsortIds = [label['id'] for label in labels]
    ids = []
    for id in unsortIds:
        if id not in ids:
            ids.append(id)

    works = []
    filenames = next(walk(f"{ENV_PATH}/cleaned"),  (None, None, []))[2]

    for filename in filenames:
        datas = load_jsonl(f"{ENV_PATH}/cleaned/{filename}")

        for data in tqdm.tqdm(datas):
            w = create_work(data, labels)
            work = {"id": w.id, "title": w.title, "content": w.content, "labels": w.labels}
            works.append(work)

    dataframe = pd.DataFrame(columns=['id', 'title', 'label', 'content'])

    work_list = []
    for work in tqdm.tqdm(works):

        if (TARGET_LABEL in work['labels']):
            label = 1
        else:
            label = 0
        work_list.append({'id': work['id'], 'title': work['title'], 'label': label, 'content': work['content']})

    works_df = pd.DataFrame(work_list)
    true_df = works_df[works_df['label'] == 1]
    false_df = works_df[works_df['label'] == 0].sample(n=len(true_df))
    data_df = pd.concat([true_df, false_df]).sample(frac=1)

    true_count = len(true_df)
    total_count = len(data_df)
    print(f'\nWorks with label {TARGET_LABEL}: {true_count}')
    print(f'\nTotal works: {total_count}\n')

    contents = data_df.content.values
    labels = data_df.label.values

    dataset = Dataset.from_pandas(data_df)
    train_testvalid = dataset.train_test_split(test_size=0.2)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    dataset = datasets.DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']
    })
    #dataset.save_to_disk(f'{ENV_PATH}/results/{TARGET_LABEL}/dataset.hf')

    
    encoding_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
    
    encoding_dataset.save_to_disk(f'{ENV_PATH}/results/{TARGET_LABEL}/encoded_dataset.hf')
    encoding_dataset.set_format("torch")
    
    train_dataloader = DataLoader(
        encoding_dataset['train'],
        sampler = RandomSampler(encoding_dataset['train']),
        batch_size = BATCH_SIZE
    )

    validation_dataloader = DataLoader(
        encoding_dataset['valid'], 
        sampler = RandomSampler(encoding_dataset['valid']),
        batch_size = BATCH_SIZE
    )

    test_dataloader = DataLoader(
        encoding_dataset['test'],
        sampler = RandomSampler(encoding_dataset['test']),
        batch_size = BATCH_SIZE
    )

    torch.cuda.empty_cache()
    model = BertForSequenceClassification.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking", 
        num_labels=2,
        output_attentions = False,
        output_hidden_states = False,
    )

    model.cuda()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps= 1e-8)

    train_loss = []
    valid_loss = [512]

    torch.cuda.empty_cache()
    
    def train(model):
        train_loss = 0
        model.train()
        for batch in train_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask, labels = b_labels)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        return train_loss

    def validation(model):
        model.eval()
        val_loss = 0
        for batch in validation_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask, labels = b_labels)
            loss = outputs[0]
            val_loss += loss.item()
        return val_loss

    st_time = time.time()
    early_stop_loss = 0
    early_stop_counter = 0
    for epoch in range(MAX_EPOCH):
        train_ = train(model)
        valid_ = validation(model)
        train_loss.append(train_)
        valid_loss.append(valid_)
        if valid_loss[epoch] < valid_loss[epoch-1]:
            early_stop_loss = valid_loss[epoch]
        else:
            early_stop_counter += 1
            if early_stop_counter == PATIENCE:
                print(f'=== early stop at Epoch: {epoch}, valid loss: {valid_:.4f}, , early stop loss: {early_stop_loss:.4f} ===')
                break            
        print(f'Epoch = {epoch} ===> train loss: {train_:.4f}, valid loss: {valid_:.4f}')

    t_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-st_time))
    print(f'\ntraining time: {t_time}\n')

    str_date = datetime.today().strftime('%Y-%m-%d')
    model.save_pretrained(f'{ENV_PATH}/results/{TARGET_LABEL}/{str_date}/BERT')

    try_count = 0
    correct_count = 0
    pred_labels = []
    actl_labels = []
    for batch in tqdm.tqdm(test_dataloader):
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        with torch.no_grad():    
            preds = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits_df = pd.DataFrame(preds[0].cpu().numpy(), columns=['logit_0', 'logit_1'])
            pred_df = pd.DataFrame(np.argmax(preds[0].cpu().numpy().tolist(), axis=1), columns=['pred_label'])
            pred_labels += pred_df['pred_label'].tolist()

            label_df = pd.DataFrame(b_labels.cpu().numpy().tolist(), columns=['true_label'])
            actl_labels += label_df['true_label'].tolist()

    report = classification_report(actl_labels, pred_labels, target_names=['False', 'True'], output_dict=True)

    report_df = pd.DataFrame(report).transpose()
    report_df.applymap('{:.4f}'.format).to_csv(f'{ENV_PATH}/results/{TARGET_LABEL}/{str_date}/result.csv')
    acc = report['accuracy']
    
    with open(f'{ENV_PATH}/results/{str_date}.jsonl', 'a+', encoding='utf-8') as file:
        now = datetime.now()
        str_time = now.strftime("%Y-%m-%d %H:%M:%S")
        data = {"label": TARGET_LABEL, "time": str_time, "training time": t_time, "total": total_count, "accuracy": float("{:.4f}".format(acc))}
        json.dump(data, file)
        file.write("\n")

if __name__ == "__main__":
    print(f'Start training: Target = {TARGET_LABEL}')
    main()
    print("\ndone!\n")