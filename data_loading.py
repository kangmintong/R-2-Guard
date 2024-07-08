import os
import json
from datasets import load_dataset
import numpy as np
import random
from utils import load_field_name
import csv

def preprocess_openai(root='data/openai'):
    instances, categories = [], []
    with open(os.path.join(root, 'samples-1680.jsonl')) as f:
        lines = f.readlines()
    malicious_num = 0
    for line in lines:
        data = json.loads(line)
        instance = data['prompt']
        instances.append(instance)
        category = 0
        for key in data.keys():
            if key != 'prompt' and data[key] != 0:
                category = 1
                malicious_num += 1
                break
        categories.append(category)
    return instances, categories

def preprocess_openai_hate(root='data/openai'):
    instances, categories = [], []
    with open(os.path.join(root, 'samples-1680.jsonl')) as f:
        lines = f.readlines()
    malicious_num = 0
    for line in lines:
        data = json.loads(line)

        if ('H' in list(data.keys()) and data['H']==1) or ('H2' in list(data.keys()) and data['H2']==1):
            instance = data['prompt']
            instances.append(instance)
            categories.append(1)

        flag_mal=0
        for key in data.keys():
            if key != 'prompt' and data[key] != 0:
                flag_mal=1
        if flag_mal==0:
            instance = data['prompt']
            instances.append(instance)
            categories.append(0)
    return instances, categories

def preprocess_openai_sex(root='data/openai'):
    instances, categories = [], []
    with open(os.path.join(root, 'samples-1680.jsonl')) as f:
        lines = f.readlines()
    malicious_num = 0
    for line in lines:
        data = json.loads(line)

        if ('S' in list(data.keys()) and data['S']==1) or ('S3' in list(data.keys()) and data['S3']==1):
            instance = data['prompt']
            instances.append(instance)
            categories.append(1)

        flag_mal = 0
        for key in data.keys():
            if key != 'prompt' and data[key] != 0:
                flag_mal = 1
        if flag_mal == 0:
            instance = data['prompt']
            instances.append(instance)
            categories.append(0)
    return instances, categories

def preprocess_openai_harassment(root='data/openai'):
    instances, categories = [], []
    with open(os.path.join(root, 'samples-1680.jsonl')) as f:
        lines = f.readlines()
    malicious_num = 0
    for line in lines:
        data = json.loads(line)

        if ('HR' in list(data.keys()) and data['HR']==1):
            instance = data['prompt']
            instances.append(instance)
            categories.append(1)

        flag_mal = 0
        for key in data.keys():
            if key != 'prompt' and data[key] != 0:
                flag_mal = 1
        if flag_mal == 0:
            instance = data['prompt']
            instances.append(instance)
            categories.append(0)
    return instances, categories

def preprocess_openai_selfharm(root='data/openai'):
    instances, categories = [], []
    with open(os.path.join(root, 'samples-1680.jsonl')) as f:
        lines = f.readlines()
    malicious_num = 0
    for line in lines:
        data = json.loads(line)

        if ('SH' in list(data.keys()) and data['SH']==1):
            instance = data['prompt']
            instances.append(instance)
            categories.append(1)

        flag_mal = 0
        for key in data.keys():
            if key != 'prompt' and data[key] != 0:
                flag_mal = 1
        if flag_mal == 0:
            instance = data['prompt']
            instances.append(instance)
            categories.append(0)
    return instances, categories

def preprocess_openai_violence(root='data/openai'):
    instances, categories = [], []
    with open(os.path.join(root, 'samples-1680.jsonl')) as f:
        lines = f.readlines()
    malicious_num = 0
    for line in lines:
        data = json.loads(line)

        if ('V' in list(data.keys()) and data['V']==1) or ('V2' in list(data.keys()) and data['V2']==1):
            instance = data['prompt']
            instances.append(instance)
            categories.append(1)

        flag_mal = 0
        for key in data.keys():
            if key != 'prompt' and data[key] != 0:
                flag_mal = 1
        if flag_mal == 0:
            instance = data['prompt']
            instances.append(instance)
            categories.append(0)
    return instances, categories

def preprocess_toxicchat(train=False, size=200):
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    if not train:
        dataset = dataset['test']
        instances = dataset['user_input']
        categories = dataset['toxicity']
    else:
        dataset = dataset['train']
        instances = dataset['user_input'][:size]
        categories = dataset['toxicity'][:size]
    return instances, categories

def preprocess_advbench_string(file_path, adv_suffix=None):
    prompts, targets = [], []
    categories = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            prom = row[0]
            if adv_suffix:
                prom = prom + adv_suffix
            prompts.append(prom)
            categories.append(1)
            if len(row) > 1:
                targets.append(row[1])
    return prompts, categories

def preprocess_hotpot(benign_num):
    instances, categories = [], []
    file = 'hotpot_test_fullwiki_v1.json'
    root = 'data/hotpot'
    with open(os.path.join(root, file)) as f:
        all_data = json.load(f)
    for data in all_data[:benign_num]:
        instance = data['question']
        instances.append(instance)
        categories.append(0)
    return instances, categories

def preprocess_advbench_string_hotpot(root='data/advbench', adv_suffix=None):
    instances, categories = [], []
    with open(os.path.join(root, 'advbench_string_hotpot.jsonl')) as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        if data['label']==1 and adv_suffix:
            instances.append(data['input'] + adv_suffix)
        else:
            instances.append(data['input'])
        categories.append(data['label'])
    return instances, categories

def preprocess_advbench_behaviour_hotpot(root='data/advbench', adv_suffix=None):
    instances, categories = [], []
    with open(os.path.join(root, 'advbench_behaviour_hotpot.jsonl')) as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        if data['label'] == 1 and adv_suffix:
            instances.append(data['input'] + adv_suffix)
        else:
            instances.append(data['input'])
        categories.append(data['label'])
    return instances, categories

def save_data(data, file_path):
    with open(file_path,'w',encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line,ensure_ascii=False)+'\n')
            f.flush()
    return

def fuse_advbench_hotpot(adv_path, output_path, benign_num):
    instance_adv, cat_adv = preprocess_advbench_string(file_path = adv_path)
    instance_hotpot, cat_hotpot = preprocess_hotpot(benign_num)
    instances = instance_adv + instance_hotpot
    categories = cat_adv + cat_hotpot
    output_data = []
    for inst, cate in zip(instances, categories):
        output_data.append({"input": inst, "label": cate})
    save_data(output_data, output_path)

def preprocess_dro(path='./data/DRO'):
    instances, categories = [], []
    path_benign = os.path.join(path, 'dro_benign.txt')
    path_malicious = os.path.join(path, 'dro_malicious.txt')

    with open(path_benign, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            instances.append(line)
            categories.append(0)
    with open(path_malicious, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            instances.append(line)
            categories.append(1)
    return instances, categories

def preprocess_xstest(path='./data/XSTest/xstest_v2_prompts.csv'):
    instances, categories = [], []
    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            instances.append(row[3])
            categories.append(int('contrast' in row[2]))
    return instances, categories


def preprocess_overkill(path='./data/overkill'):
    instances, categories = [], []
    path_benign = os.path.join(path, 'benign.csv')
    path_malicious = os.path.join(path, 'xstest_v2_prompts.csv')

    with open(path_benign, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            instances.append(row[0])
            categories.append(0)
    with open(path_malicious, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            if int('contrast' in row[2]):
                instances.append(row[3])
                categories.append(int('contrast' in row[2]))
    return instances, categories


def preprocess_ours(path='./data/ours/ours.csv'):
    instances, categories = [], []
    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            instances.append(','.join(row[1:]))
            categories.append(int('malicious' in row[0] or 'Malicious' in row[0]))
    return instances, categories

def preprocess_test(root='data/openai'):
    instances, categories = [], []
    with open(os.path.join(root, 'samples-1680.jsonl')) as f:
        lines = f.readlines()
    malicious_num = 0
    for line in lines:
        data = json.loads(line)
        instance = data['prompt']
        instances.append(instance)
        category = 0
        for key in data.keys():
            if key != 'prompt' and data[key] != 0:
                category = 1
                malicious_num += 1
                break
        categories.append(category)
    return instances, categories

def preprocess_beavertail():
    instances, categories = [], []
    round0_dataset = load_dataset('PKU-Alignment/BeaverTails')
    test_data = round0_dataset["30k_test"]
    for x in test_data:
        instances.append(x['prompt']+'\n'+x['response'])
        if x['is_safe']:
            categories.append(0)
        else:
            categories.append(1)
    return instances, categories

def load_data(dataset, args):
    if "openai" in dataset:
        return preprocess_openai()
    elif "toxicchat" in dataset:
        return preprocess_toxicchat()
    elif dataset=='advbench_string':
        return preprocess_advbench_string(file_path = './data/advbench/harmful_strings.csv', adv_suffix=args.advbench_suffix)
    elif dataset=='advbench_behaviour':
        return preprocess_advbench_string(file_path = './data/advbench/harmful_behaviors.csv', adv_suffix=args.advbench_suffix)
    elif dataset=='advbench_string_hotpot':
        return preprocess_advbench_string_hotpot(adv_suffix=args.advbench_suffix)
    elif dataset=='advbench_behaviour_hotpot':
        return preprocess_advbench_behaviour_hotpot(adv_suffix=args.advbench_suffix)
    elif dataset=='dro':
        return preprocess_dro()
    elif dataset=='xstest':
        return preprocess_xstest()
    elif dataset=='overkill':
        return preprocess_overkill()
    elif dataset=='ours':
        return preprocess_ours()
    elif dataset=='test':
        return preprocess_test()
    elif dataset=='beavertail':
        return preprocess_beavertail()
    elif dataset=='mod_hate':
        return preprocess_openai_hate()
    elif dataset=='mod_sex':
        return preprocess_openai_sex()
    elif dataset=='mod_harassment':
        return preprocess_openai_harassment()
    elif dataset=='mod_selfharm':
        return preprocess_openai_selfharm()
    elif dataset=='mod_violence':
        return preprocess_openai_violence()
    else:
        raise ValueError(f"Dataset {dataset} is not supported!")

def construct_pseudo_training_set(data_size, dim_list):
    scores_all = []
    labels = []
    num_models = len(dim_list)

    for i in range(data_size):
        if np.random.uniform(0, 1) > 0.5:
            labels.append(1)
        else:
            labels.append(0)
        scores_one_instance = []
        for j in range(num_models):
            cur_scores = []
            for k in range(dim_list[j]):
                cur_scores.append(np.random.uniform(0, 0.1))
            if labels[-1] == 1:
                cur_scores[random.choice(list(range(dim_list[j])))] = np.random.uniform(0.9, 1)
            scores_one_instance.append(cur_scores)
        scores_all.append(scores_one_instance)
    return scores_all, labels

def sample_real(data_size, model_names, dataset, args):
    scores_all = []
    scores_total = []
    for j in range(len(model_names)):
        if 'train' in dataset:
            dataset = dataset.split('_')[0]
        score_path =  f'./cache/{model_names[j]}_{dataset}_scores.json'
        with open(score_path, 'r') as file:
            scores = json.load(file)
        scores_total.append(scores)

    instances, categories = load_data(args.dataset, args)
    size_positive_negative = [int(data_size * args.pos_ratio), data_size - int(data_size * args.pos_ratio)]
    for i in range(len(categories)):
        if size_positive_negative[categories[i]]<=0:
            continue
        size_positive_negative[categories[i]] -= 1
        scores_one_instance = []
        for j in range(len(model_names)):
            scores_one_instance.append([scores_total[j][fil][i] for fil in load_field_name(model_names[j])])
        scores_all.append(scores_one_instance)

    return scores_all, categories