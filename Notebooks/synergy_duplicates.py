import sys
import os
import random
import gc
import re
import time
import torch
import numpy as np
import pandas as pd
import pickle
import ast
from tqdm import tqdm
from scipy.sparse import csr_matrix
import itertools
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata
from sklearn.metrics import ndcg_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
# import nltk
# nltk.download('punkt')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from SHapRAG import *
from SHapRAG.utils import *

df=pd.read_parquet("../data/train_2wiki.parquet")
df=df[df.type=="comparison"]
df=df.reset_index().drop(columns=["index"],inplace=False)

# 2wiki
def annotate(gt, context):
    docs = []
    sents = []
    gt_docs = set()
    gt_sents = set()
    gt_anns = {}
    for gt_ann in gt:
        try:
            gt_anns[gt_ann[0]].append(gt_ann[1])
        except:
            gt_anns[gt_ann[0]]= [gt_ann[1]]
    gt_i = 0
    for para in range(len(context)):
        title, sentences = context[para]
        if title in gt_anns:
            gt_indices = gt_anns[title]
            #print('title', title, '\nsentences', sentences)
            gt_docs.add(para)
            for idx in gt_indices:
                gt_id = gt_i + idx
                gt_sents.add(gt_id)
        gt_i += len(sentences)
        doc = ''.join(sentences)
        docs.append(doc)
        sents = sents + [x.strip() for x in sentences]
    return docs, sents, gt_docs, gt_sents
 
all_docs = []
all_sents = []
all_gt_docs = []
all_gt_sents = []
all_reordered_docs = []
all_reordered_sents = []
all_len_gt = []
#all_labels = []
num_gt_docs = 0
num_gt_sents = 0
for i in range(len(df.context)):
    gt = ast.literal_eval(df.supporting_facts[i])
    context = ast.literal_eval(df.context[i])
    docs, sents, gt_docs, gt_sents = annotate(gt, context)
    arr_gt_docs = list(gt_docs)
    num_gt_docs += len(arr_gt_docs)
    arr_gt_sents = list(gt_sents)
    num_gt_sents += len(arr_gt_sents)
    not_gt = [i for i in range(len(docs)) if i not in gt_docs]
    not_gt_sents = [i for i in range(len(sents)) if i not in gt_sents]

    if len(arr_gt_docs) == 2:
        all_len_gt.append(2)
        # NO DUPLICATES
        reordered_docs = [docs[arr_gt_docs[0]]] + [docs[arr_gt_docs[1]]] + [docs[not_gt[0]]] + [docs[not_gt[1]]] + [docs[not_gt[2]]] + \
                         [docs[x] for x in not_gt[3:]]
        reordered_sents = [sents[arr_gt_sents[0]]] + [sents[arr_gt_sents[1]]] + [sents[not_gt_sents[0]]] + [sents[not_gt_sents[1]]] + \
                         [sents[not_gt_sents[2]]] + [sents[x] for x in not_gt_sents[3:]]

    elif len(arr_gt_docs) == 3:
        all_len_gt.append(3)
        # NO DUPLICATES
        reordered_docs = [docs[arr_gt_docs[0]]] + [docs[arr_gt_docs[1]]] + [docs[arr_gt_docs[2]]] + [docs[not_gt[0]]] + [docs[not_gt[1]]] + \
                         [docs[x] for x in not_gt[2:]]
        reordered_sents = [sents[arr_gt_sents[0]]] + [sents[arr_gt_sents[1]]] + [sents[arr_gt_sents[2]]] + [sents[not_gt_sents[0]]] + \
                         [sents[not_gt_sents[1]]] + [sents[x] for x in not_gt_sents[2:]]

    elif len(arr_gt_docs) == 4:
        all_len_gt.append(4)
        # NO DUPLICATES
        reordered_docs = [docs[arr_gt_docs[0]]] + [docs[arr_gt_docs[1]]] + [docs[arr_gt_docs[2]]] + [docs[arr_gt_docs[3]]] + \
                         [docs[not_gt[0]]] + [docs[x] for x in not_gt[1:]]
        reordered_sents = [sents[arr_gt_sents[0]]] + [sents[arr_gt_sents[1]]] + [sents[arr_gt_sents[2]]] + [sents[arr_gt_sents[3]]] + \
                         [sents[not_gt_sents[0]]] + [sents[x] for x in not_gt_sents[1:]]

    all_docs.append(docs)
    all_sents.append(sents)
    all_gt_docs.append(list(gt_docs))
    all_gt_sents.append(list(gt_sents))
    all_reordered_docs.append(reordered_docs)
    all_reordered_sents.append(reordered_sents)
av_gt_docs = num_gt_docs / len(df.context)
av_gt_sents = num_gt_sents / len(df.context)
print('average gt docs: ', av_gt_docs, '\naverage gt sents: ', av_gt_sents)
 
df['paragraphs'] = all_docs
df['sentences'] = all_sents
df['gt_paragraphs'] = all_gt_docs
df['gt_sentences'] = all_gt_sents
df['reordered_paragraphs'] = all_reordered_docs
df['reordered_sentences'] = all_reordered_sents
df['len_gt'] = all_len_gt

with open("../data/new_questions_2wiki.txt", "r", encoding="utf-8") as f:
    new_questions = [line.strip() for line in f.readlines()]

    SEED = 42
# Initialize Accelerator
accelerator_main = Accelerator(mixed_precision="fp16")

# Load Model
if accelerator_main.is_main_process:
    print("Main Script: Loading model...")
# model_path = "mistralai/Mistral-7B-Instruct-v0.3"
model_path = "meta-llama/Llama-3.1-8B-Instruct"
# model_path = "Qwen/Qwen2.5-3B-Instruct"

model_cpu = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model_cpu.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model_cpu, 'generation_config') and model_cpu.generation_config is not None:
        model_cpu.generation_config.pad_token_id = tokenizer.pad_token_id

if accelerator_main.is_main_process:
    print("Main Script: Preparing model with Accelerator...")
prepared_model = accelerator_main.prepare(model_cpu)
unwrapped_prepared_model = accelerator_main.unwrap_model(prepared_model)
unwrapped_prepared_model.eval()
if accelerator_main.is_main_process:
    print("Main Script: Model prepared and set to eval.")

accelerator_main.wait_for_everyone()
utility_cache_base_dir = f"../Experiment_data/complementary/{model_path.split('/')[1]}/new"

num_questions_to_run = 100
K_VALUES = [1, 2, 3, 4, 5]
all_results = []
extras = []

for i in range(num_questions_to_run):
    query = new_questions[i]
    # if res[i]=="True":
    if accelerator_main.is_main_process:
        print(f"\n--- Question {i+1}/{num_questions_to_run}: {query[:60]}... ---")

    docs = df.reordered_paragraphs[i]
    utility_cache_filename = f"utilities_q_idx{i}.pkl"
    current_utility_path = os.path.join(utility_cache_base_dir, utility_cache_filename)

    if accelerator_main.is_main_process:
        os.makedirs(os.path.dirname(current_utility_path), exist_ok=True)

    harness = ContextAttribution(
        items=docs,
        query=query,
        prepared_model=prepared_model,
        prepared_tokenizer=tokenizer,
        accelerator=accelerator_main,
        utility_cache_path=current_utility_path,
        utility_mode='log-perplexity'
    )
    
    full_budget=pow(2,harness.n_items)
    # res = evaluate(df.question[i], harness.target_response, df.answer[i])
    # print(res)
    if accelerator_main.is_main_process:
        methods_results = {}
        metrics_results = {}
        extra_results = {}

        m_samples_map = {"XS":32, "S":64, "M":128, "L":264, "XL":528, "XXL":724}

        # Store FM models for later R²/MSE
        fm_models = {}
        methods_results['Exact-Shap']=harness._calculate_shapley()
        harness.save_utility_cache(current_utility_path)
