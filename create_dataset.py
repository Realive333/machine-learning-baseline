import os
import sys
import json

import work

import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

ENV_PATH = f'./dataset/{sys.argv[1]}'
TARGET_LABEL = sys.argv[2]
WORK_DIR = f'{ENV_PATH}/seperated/{TARGET_LABEL}'

def save_jsonl(path, datas):
    try:
        os.makedirs(WORK_DIR)
    except FileExistsError:
        print("folder exist")
        
    with open(path, 'w+', encoding='utf-8') as writer:
        for data in tqdm(datas):
            json.dump(data, writer, ensure_ascii=False)
            writer.write("\n")

def main():
    works = work.read_cleaned_works(f"{ENV_PATH}/cleaned")
    work_list = work.get_work_list_by_label(works, TARGET_LABEL)
    works_df = pd.DataFrame(work_list)
    true_df = works_df[works_df['label'] == 1]
    false_df = works_df[works_df['label'] == 0].sample(n=len(true_df))
    data_df = pd.concat([true_df, false_df]).sample(frac=1)
    
    del(works_df)
    del(true_df)
    del(false_df)
    
    print(len(data_df))
    
    train_df = data_df.sample(frac = 0.8)
    vt_df = data_df.drop(train_df.index)
    
    valid_df = vt_df.sample(frac = 0.5)
    test_df = vt_df.drop(valid_df.index)
    
    del(vt_df)
    
    train_dict = train_df.to_dict(orient='records')
    save_jsonl(f'{WORK_DIR}/train.jsonl', train_dict)
    del(train_dict)
    
    valid_dict = valid_df.to_dict(orient='records')
    save_jsonl(f'{WORK_DIR}/valid.jsonl', valid_dict)
    del(valid_dict)
    
    test_dict = test_df.to_dict(orient='records')
    save_jsonl(f'{WORK_DIR}/test.jsonl', test_dict)
    del(test_dict)
    
if __name__ == "__main__":
    print(f'create dataset TARGET:{ENV_PATH} LABEL:{TARGET_LABEL}')
    main()