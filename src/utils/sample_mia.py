"""
Sample the shadow datasets from auxiliary dataset
Modified from https://github.com/computationalprivacy/querycheetah/blob/main/src/dataset_sampler.py
"""

import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import abstractmethod
from multiprocessing import Pool
from collections import defaultdict
from .general import _find_record_indexs

def filter_queries(released_queries, sens_attribute):
    quasi_ids_only_released_queries = []
    for query in released_queries:
        _query = {col: values for col, values in query}
        if sens_attribute in _query:
            del _query[sens_attribute]
        quasi_ids_only_released_queries.append(tuple((col, _query[col]) for col in _query))
    return list(set(quasi_ids_only_released_queries))

def get_query2indices_of_full_dataset(full_dataset, quasi_ids_only_released_queries, domain_config):
    # The attribute of queries in query2indices should only contain the ones in quasi-ids
    query2indices = {}
    all_records = [tuple([(col, value) for col, value in zip(domain_config, record.tolist())]) for _, record in full_dataset.iterrows()]
    with Pool(maxtasksperchild = 100000) as pool:
        pbar = zip(quasi_ids_only_released_queries, [all_records] * len(quasi_ids_only_released_queries))
        for query, record_indexs in pool.imap_unordered(_find_record_indexs, pbar):
            query2indices[query] = record_indexs
    return query2indices

def get_train_and_eval_shadow_labels(args, domain_config, target_records: pd.DataFrame, aux_split: pd.DataFrame, target_split: pd.DataFrame, M, S):
    indexs_train_val, y_train_eval = {}, {}
    tmp_sampler = get_dataset_sampler(args, domain_config, target_records, aux_split, target_split)
    qbs_types = ['train' if i < args.num_training_qbses * (1 - 0.3333) else 'eval' for i in range(args.num_training_qbses)]
    pbar = zip(range(0, args.num_training_qbses), qbs_types, range(args.num_training_qbses))

    for seed, qbs_type, dataset_index in pbar:
        indexs, label = tmp_sampler.sample_dataset(seed, qbs_type)
        indexs_train_val[dataset_index] = indexs
        y_train_eval[dataset_index] = label
        for sens_value in indexs:
            if sens_value is None:
                M[dataset_index, indexs[None]] = 1
            else:
                S[(tmp_sampler.sens_attribute, sens_value)][dataset_index, indexs[sens_value]] = 1
    return [x[1] for x in sorted(indexs_train_val.items(), key = lambda x: x[0])], [x[1] for x in sorted(y_train_eval.items(), key = lambda x: x[0])]

def get_dataset_sampler(args, domain_config, target_records, aux_split, target_split):
    return AuxiliaryWithoutReplacementSampler(args, domain_config, target_records, aux_split, target_split)

def get_query2answers(full_dataset, query2indices, M, S, device):
    query2answers = defaultdict(list)
    # Over the full dataset
    query2index = {query: i for (i, query) in enumerate(query2indices)}
    index2query = {query2index[q]: q for q in query2index}

    evaluate_queries_on_shadow_datasets(M, S, full_dataset, index2query, query2answers, query2indices, device)
    return query2answers

def evaluate_queries_on_shadow_datasets(M, S, full_dataset, index2query, query2answers, query2indices_of_full_dataset, device):
    # The batch size to process queries and allocate memory for shadow datasets
    query_per_batch, shadow_qbs_per_batch = 20000, 20000

    for start_query in range(0, len(index2query), query_per_batch):
        T = np.zeros((len(full_dataset), min(query_per_batch, len(index2query) - start_query + 1)))
        for i in range(start_query, min(start_query + query_per_batch, len(index2query))):
            x = np.array(list(query2indices_of_full_dataset[index2query[i]]))
            if len(x) > 0:
                T[x, i % start_query if start_query != 0 else i] = 1
        if device.type == 'cuda':
            T = torch.from_numpy(T).to(device)

        for start_qbs in range(0, M.shape[0], shadow_qbs_per_batch):
            selected_qbs = min(shadow_qbs_per_batch, M.shape[0] - start_qbs + 1)
            all_answers = {}
            if device.type == 'cuda':
                M_batch = torch.from_numpy(M[start_qbs : start_qbs + selected_qbs, :]).to(device)
                all_answers[None] = torch.matmul(M_batch, T).cpu().numpy()
                for key in S:
                    S_batch = torch.from_numpy(S[key][start_qbs : start_qbs + selected_qbs, :]).to(device)
                    all_answers[key] = torch.matmul(S_batch, T).cpu().numpy()
            else:
                M_batch = M[start_qbs : start_qbs + selected_qbs, :]
                all_answers[None] = np.matmul(M_batch, T)
                for key in S:
                    S_batch = S[key][start_qbs : start_qbs + selected_qbs, :]
                    all_answers[key] = np.matmul(S_batch, T)

            for i in range(start_query, min(start_query + query_per_batch, len(index2query))):
                query2answers[tuple(sorted(index2query[i], key = lambda x: x[0]))] += all_answers[None][:, i % start_query if start_query != 0 else i].tolist()
                for col, value in S:
                    sen_col_value_pair = (col, (value,))
                    query2answers[tuple(sorted(index2query[i] + (sen_col_value_pair,), key = lambda x: x[0]))] += all_answers[(col, value)][:, i % start_query if start_query != 0 else i].tolist()

        del T, M_batch, S_batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        else:
            gc.collect()

class AuxiliaryWithoutReplacementSampler():
    """
    Generates shadow datasets having records sampled without replacement from
    a given auxiliary dataset. The auxiliary dataset is first partitioned into
    a train and a validation split.
    """
    def __init__(self, args, domain_config, target_records: pd.DataFrame, auxiliary_dataset: pd.DataFrame, target_dataset: pd.DataFrame):
        if "ppmf" in args.dataset: 
            self.sens_attribute = "cenhisp"
        elif "acs" in args.dataset:
            self.sens_attribute = "label"

        self.auxiliary_dataset = auxiliary_dataset
        self.target_dataset = target_dataset
        self.target_records = target_records
        assert(self.target_records.shape[0] == 1)
        # MIA target dataset is having one more record than AIA target dataset
        self.shadow_size = len(self.target_dataset) - 1
        assert self.shadow_size < len(auxiliary_dataset), f'ERROR: Cannot sample without replacement {self.shadow_size}/{len(self.auxiliary_dataset)} records.'
        
        self.domain_config = domain_config

    def sample_dataset(self, seed, eval_type):
        np.random.seed(seed)
        target_record_in_target_dataset = np.random.randint(0, 2)

        self.split = {}
        # Remove from the auxiliary dataset all the records that are identical to the target record.
        aux_indexs = self.auxiliary_dataset.index
        non_sens_attributes = [key for key in self.domain_config.keys() if key != self.sens_attribute]
        merged_df = pd.merge(self.auxiliary_dataset.reset_index(), self.target_records[non_sens_attributes], on=non_sens_attributes, how='left', indicator=True).set_index(aux_indexs)
        auxiliary_dataset_without_target = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        randomized_auxiliary_dataset_without_target = auxiliary_dataset_without_target.sample(frac = 1.0, random_state = seed)
        self.split['train'] = randomized_auxiliary_dataset_without_target[:len(auxiliary_dataset_without_target) // 2]
        self.split['eval'] = randomized_auxiliary_dataset_without_target[len(auxiliary_dataset_without_target) // 2:]

        if target_record_in_target_dataset == 1:
            # If target record is in the shadow dataset
            num_other_records = min(len(self.split[eval_type]), self.shadow_size - 1)
        elif target_record_in_target_dataset == 0:
            # Otherwise
            num_other_records = min(len(self.split[eval_type]), self.shadow_size)
        aux_indexs = np.random.permutation(range(len(self.split[eval_type])))[:num_other_records]
        other_records = self.split[eval_type].iloc[aux_indexs][self.target_records.columns]

        idx_record = np.random.choice(num_other_records + 1, size=1, replace=False)
        list_dfs = []
        start = 0
        for idx, end in enumerate(sorted(idx_record)):
            list_dfs.append(other_records[start: end])
            if target_record_in_target_dataset == 1:
                list_dfs.append(self.target_records.iloc[[idx], :])
            start = end
        list_dfs.append(other_records[end:])
        dataset = pd.concat(list_dfs)
        assert(dataset.shape[0] == self.shadow_size)

        indexs = {None: dataset.index.to_numpy()}
        for sens_value in range(self.domain_config[self.sens_attribute]):
            indexs[sens_value] = dataset[dataset[self.sens_attribute] == sens_value].index.to_numpy()
        
        return indexs, target_record_in_target_dataset
