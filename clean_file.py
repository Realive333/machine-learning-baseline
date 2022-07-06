import sys
import json
import tqdm
import time
import os
from os import walk

ENV_PATH = sys.argv[1]

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data
    
def save_jsonl(path, datas):
    with open(path, 'w+', encoding='utf-8') as writer:
        for data in tqdm.tqdm(datas):
            json.dump(data, writer, ensure_ascii=False)
            writer.write("\n")
   
   
filenames = next(walk(f"./dataset/{ENV_PATH}"),  (None, None, []))[2]
try: 
    os.mkdir(f"./dataset/{ENV_PATH}/cleaned")
except FileExistsError:
    print ("folder exist")
    
for filename in filenames:
    datas = load_jsonl(f"./dataset/{ENV_PATH}/{filename}")
    
    print(f"{filename} original lines: {len(datas)}")
    for i, data in enumerate(datas):
        if data['labels'] == None:
            datas[i]['labels'] == []
    print(f"{filename} cleaned lines: {len(datas)}")

    save_jsonl(f'./dataset/{ENV_PATH}/cleaned/{filename}', datas)