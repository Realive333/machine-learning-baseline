import csv
import sys
import time
import torch
import json
import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
from torch.optim import AdamW
from datasets import Dataset
from datasets import load_from_disk
from transformers import BertForSequenceClassification
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

ENV_PATH = f'./dataset/{sys.argv[1]}'
TARGET_LABEL = sys.argv[2]
BATCH_SIZE = 8
MAX_EPOCH = 4096
PATIENCE = 10

def load_csv(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as file:
        rows = csv.reader(file)
        for row in rows:
            label = {"id": row[0], "name": row[1]}
            data.append(label)
    return data

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    labels = load_csv(f"./dataset/label_list.csv")
    
    encoding_dataset = load_from_disk(f'{ENV_PATH}/results/{TARGET_LABEL}/encoded_dataset.hf')

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
    
    total_count = len(encoding_dataset['train'])+len(encoding_dataset['valid'])+len(encoding_dataset['test'])
    print(total_count)
    
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
    valid_loss = []

    torch.cuda.empty_cache()
    encoding_dataset.set_format("torch")
        
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
    early_stop_loss = 1000
    early_stop_threshold = 0
    for epoch in range(MAX_EPOCH):
        train_ = train(model)
        valid_ = validation(model)
        train_loss.append(train_)
        valid_loss.append(valid_)
        if valid_ < early_stop_loss:
            early_stop_loss = valid_
        else:
            early_stop_threshold += 1
            if early_stop_threshold == PATIENCE:
                print(f'=== early stop at Epoch: {epoch}, valid loss: {valid_:.4f}')
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
    print(f'Start retraining: Target = {TARGET_LABEL}')
    main()
    print("\ndone!\n")