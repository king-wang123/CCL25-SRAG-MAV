import logging
import os
from tqdm import tqdm
import subprocess
import json
import concurrent.futures
import multiprocessing
import logging                

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_json(path):
    return json.load(open(path))

def get_jsonline(path):
    return [json.loads(line) for line in open(path)]

def get_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
    
def save_jsonline(datas, path):
    with open(path, 'w') as fw:
        for data in datas:
            print(json.dumps(data, ensure_ascii=False), file=fw)

def save_json(datas, path):
    with open(path, 'w') as fw:
        print(json.dumps(datas, ensure_ascii=False, indent=4), file=fw)

# 四元组-->三元组
def output2triple(text):
    triple = ''
    seqs = text.split(' [SEP] ')
    for seq in seqs:
        parts = seq.split(' | ')
        triple += f'{parts[0]} | {parts[1]} | {parts[2]} [SEP] '
    return triple[:-7] + ' [END]'

# 三元组-->四元组
def process_triple(output):
    res = ''
    try:
        assert output[-6:] == ' [END]'
        output = output[:-6]
        seqs = output.split(' [SEP] ')
        for seq in seqs:
            parts = seq.split(' | ')
            res += f'{parts[0]} | {parts[1]} | {parts[2]} | '
            hate_classes = parts[2].split(', ')
            hate = 'hate'
            for hate_class in hate_classes:
                if hate_class not in ['Racism', 'Region', 'LGBTQ', 'Sexism', 'others', 'non-hate']:
                    return ''
                if hate_class == 'non-hate':
                    hate = 'non-hate'
            res += f'{hate} [SEP] '
        res = res[:-7] + ' [END]'
        return res
    except:
        return ''

# 判断是否为符合要求的四元组
def check_response(output):
    try:
        assert output[-6:] == ' [END]'
        output = output[:-6]
        seqs = output.split(' [SEP] ')
        for seq in seqs:
            parts = seq.split(' | ')
            hate_classes = parts[2].split(', ')
            hate = parts[3]
            for hate_class in hate_classes:
                if hate_class not in ['Racism', 'Region', 'LGBTQ', 'Sexism', 'others', 'non-hate']:
                    return False
                if hate_class == 'non-hate':
                    if hate != 'non-hate':
                        return False
                else:
                    if hate != 'hate':
                        return False
        return True
    except:
        return False