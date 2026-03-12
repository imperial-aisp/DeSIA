import os
import time
import torch
import shutil
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from .domain import Domain
from collections import defaultdict
from .iter import IterativeAlgoNonDP
from ....utils.parser import Parser
from .model import FixedGenerator
from multiprocessing import Process, Pool
from .qm import KWayMarginalQMTorch, KWayMarginalSetQMTorch
from .qm import get_data, get_my_workloads, get_data_onehot, hamming_distance

class RAPAttack():
    def __init__(self, args: Parser, model_inputs: list, repo_directory: str):
        self.args = args
        self.density = True
        self.model_inputs = model_inputs
        self.dataset_directory = os.path.join(repo_directory, "datasets")

        args_list = []
        for arg, value in vars(self.args).items():
            if arg not in ["filenames", "attack", "multi_cpus", "chosen_gpus"]:
                if type(value) is not list:
                    args_list.append("{}-{}".format(arg, value))
                else:
                    args_list.append("{}-{}".format(arg, "_".join([str(x) for x in value])))
        self.base_dir = os.path.join(repo_directory, "tmp" if len(self.args.note) == 0 else f"tmp_{self.args.note}", "model", "{}".format(self.args.attack), *args_list, "filename-{}")
        
        if len(set(range(torch.cuda.device_count())).intersection(self.args.chosen_gpus)) > 0:
            self.devices = [torch.device('cuda:{}'.format(gpu_idx)) for gpu_idx in range(torch.cuda.device_count()) if (torch.cuda.is_available() and (gpu_idx in self.args.chosen_gpus))]
        else:
            self.devices = [torch.device('cpu')]

    def init_model(self, model_dict, aux_onehot_split, seed = None):
        G = model_dict['G']
        weights = G.generator.syndata.weight
        if len(weights) != len(aux_onehot_split):
            prng = np.random.RandomState(seed)
            idxs = prng.choice(aux_onehot_split.shape[0], size=len(weights))
        else:
            idxs = np.arange(len(aux_onehot_split))
        mask = aux_onehot_split[idxs]
            
        maxes = weights.max(axis=1)[0].detach()
        mins = weights.min(axis=1)[0].detach() * 2 
        # new_std = np.abs(mins.mean().item() * 0.1)
        weights_new = []
        st = 0
        for item in G.transformer.output_info:
            ed = st + item[0]
            x = weights[:, st:ed].detach()
            _mask = torch.tensor(mask[:, st:ed])

            x[_mask] = maxes
            x[~_mask] = mins.repeat_interleave(_mask.shape[-1] - 1)
            weights_new.append(x)
            st = ed
        weights_new = torch.cat(weights_new, axis=1)
        G.generator.syndata.weight = torch.nn.Parameter(weights_new)
        return G

    def get_qm(self, data, released_queries: list, domain: Domain, device):
        workloads = get_my_workloads(released_queries)
        if len(np.unique([len(col_val_pairs) for col_val_pairs in workloads])) == 1:
            query_manager = KWayMarginalQMTorch(data, workloads, released_queries, domain, verbose=False, device=device)
        else:
            query_manager = KWayMarginalSetQMTorch(data, workloads, released_queries, domain, verbose=False, device=device)
        return query_manager

    def get_model(self, query_manager, device, seed):
        model_dict = {}
        model_dict['G'] = FixedGenerator(query_manager, K=1000, device=device, init_seed=seed)
        model_dict['lr'] = 1e-1
        model_dict['eta_min'] = None
        return model_dict
    
    def get_syn_aux_dataset(self, filename):
        released_aggregates, init_dataset, target_dataset, domain_config, _ = self.model_inputs[filename]
        syndata_size = len(target_dataset)
        domain = Domain(domain_config.keys(), domain_config.values())
        data, _ = get_data(target_dataset, domain), get_data(init_dataset, domain)

        model_save_dir = None
        query_manager = self.get_qm(data, list(released_aggregates.keys()), domain, self.devices[0 % len(self.devices)])
        # If density parameter is False, return the number of records. Else, return percentage of records.
        true_answers = torch.tensor(list(released_aggregates.values())) / syndata_size
        true_answers = true_answers.to(self.devices[0 % len(self.devices)])
        syn_data = []

        for model_idx in tqdm(range(100)):
            model_dict = self.get_model(query_manager, self.devices[0 % len(self.devices)], model_idx)
            syn_generator = IterativeAlgoNonDP(model_dict['G'], 1000,
                                    default_dir=model_save_dir, verbose=False, seed=model_idx,
                                    lr=model_dict['lr'], eta_min=model_dict['eta_min'],
                                    max_idxs=1024, max_iters=1,
                                    sample_by_error=True, log_freq=1,
                                    density=self.density)
            syn_generator.fit(true_answers)
            # syn_generator.save('model-{}.zip'.format(model_idx), model_save_dir)
            syn_data.append(syn_generator.G.get_syndata(num_samples = syndata_size, how = "sample").df)
            # Recollect garbage to release memory 
            del syn_generator, model_dict
        del query_manager, data, true_answers
        return pd.concat(syn_data)

    def train(self):
        np.random.seed(0)
        pbar = tqdm(enumerate(self.model_inputs))
        for target_qbs_idx, filename in pbar:
            pbar.set_description("Training the generator model of {}/{}th dataset".format(target_qbs_idx + 1, len(self.model_inputs)))
            released_aggregates, _, target_dataset, _, sens_attribute = self.model_inputs[filename]

            # Unique quasi-id records are attack/eval targets only.
            quasi_ids = [col for col in target_dataset.columns if col != sens_attribute]
            unique_mask = target_dataset.groupby(by = quasi_ids).transform('size') == 1
            unique_records = target_dataset[unique_mask]
            if not os.path.exists(self.base_dir.format(filename)):
                os.makedirs(self.base_dir.format(filename))
            unique_records.to_csv(os.path.join(self.base_dir.format(filename), "unique_records.csv"), index = True)

            assert self.args.mode == "aia"
            self._train_worker(0, unique_records.index[0], released_aggregates, filename)
            tmp_model_save_dir = os.path.join(self.base_dir, "record-{}").format(filename, unique_records.index[0])
            for unique_record_idx, _ in unique_records.iterrows():
                record_dir = os.path.join(self.base_dir, "record-{}")
                model_save_dir = record_dir.format(filename, unique_record_idx)
                if os.path.exists(model_save_dir):
                    continue
                shutil.copytree(tmp_model_save_dir, model_save_dir)

    def _train_worker(self, process_idx: int, unique_record_idx: int, input_constraints: dict, filename: str):
        train_start_time = time.time()
        _, init_dataset, target_dataset, domain_config, _ = self.model_inputs[filename]
        syndata_size = len(target_dataset)
        domain = Domain(domain_config.keys(), domain_config.values())
        data, aux_data = get_data(target_dataset, domain), get_data(init_dataset, domain)
        aux_onehot_split = get_data_onehot(aux_data)

        record_dir = os.path.join(self.base_dir, "record-{}")
        if unique_record_idx == -1:
            # Generate synthetic aux dataset
            model_save_dir = None
        else:
            # Obtain candidate datasets for AIA
            model_save_dir = record_dir.format(filename, unique_record_idx)
        query_manager = self.get_qm(data, list(input_constraints.keys()), domain, self.devices[process_idx % len(self.devices)])
        # If density parameter is False, return the number of records. Else, return percentage of records.
        true_answers = torch.tensor(list(input_constraints.values())) / syndata_size
        true_answers = true_answers.to(self.devices[process_idx % len(self.devices)])
        syn_data = []

        for model_idx in range(self.args.num_models):
            model_dict = self.get_model(query_manager, self.devices[process_idx % len(self.devices)], model_idx)
            if self.args.warm_start:
                init_G = self.init_model(model_dict, aux_onehot_split, model_idx)
                del model_dict['G']
            else:
                init_G = model_dict['G']
            syn_generator = IterativeAlgoNonDP(init_G, 1000,
                                    default_dir=model_save_dir, verbose=False, seed=model_idx,
                                    lr=model_dict['lr'], eta_min=model_dict['eta_min'],
                                    max_idxs=1024, max_iters=1,
                                    sample_by_error=True, log_freq=1,
                                    density=self.density)
            syn_generator.fit(true_answers)
            # syn_generator.save('model-{}.zip'.format(model_idx), model_save_dir)
            syn_data.append(syn_generator.G.get_syndata(num_samples = syndata_size, how = "sample").df)
            # Recollect garbage to release memory 
            del syn_generator, model_dict
        del query_manager, data, true_answers

        # Save synthetic datasets
        df_syn_raw_solutions = pd.DataFrame(pd.concat(syn_data))
        df_syn_target = target_dataset.groupby(by = target_dataset.columns.tolist()).size().reset_index(name="count").sort_values(by = ["count"], ascending = False)
        df_syn_target.to_csv(os.path.join(model_save_dir, "target_by_unique_rows.csv"), index = False)
        df_syn_solution = df_syn_raw_solutions.groupby(by = target_dataset.columns.tolist()).size().reset_index(name="count").sort_values(by = ["count"], ascending = False)
        df_syn_solution.to_csv(os.path.join(model_save_dir, "solution_by_unique_rows.csv"), index = False)

        train_time = time.time() - train_start_time
        with open(os.path.join(model_save_dir, "logs.txt"), "w") as fp:
            fp.write("Train time (seconds): {}\n".format(train_time))

    def eval(self):
        for filename in self.model_inputs:
            unique_records = pd.read_csv(os.path.join(self.base_dir.format(filename), "unique_records.csv"), index_col = 0)
            pbar = tqdm(unique_records.iterrows())
            for indexed_row in pbar:
                pbar.set_description("Evaluating dataset {}".format(filename))
                self._eval((indexed_row, filename))

    def _eval(self, _input_args):
        eval_start_time = time.time()
        (unique_record_idx, unique_record), filename = _input_args
        _, _, _, domain_config, sens_attribute = self.model_inputs[filename]

        record_dir = os.path.join(self.base_dir, "record-{}")
        model_save_dir = record_dir.format(filename, unique_record_idx)
        df_syn_target = pd.read_csv(os.path.join(model_save_dir, "target_by_unique_rows.csv"))
        df_syn_solution = pd.read_csv(os.path.join(model_save_dir, "solution_by_unique_rows.csv"))

        # Convert from dataset to dictionary, record_in_tuple_format: conuts
        syn_target = {tuple(row[:-1].tolist()): row[-1] for _, row in df_syn_target.iterrows()}
        syn_solution = {tuple(row[:-1].tolist()): row[-1] for _, row in df_syn_solution.iterrows()}

        # By attribute inference: querying upon unique quasi-ids
        syn_target_by_quasi_ids = {}
        for row in syn_target:
            if row[:-1] not in syn_target_by_quasi_ids:
                syn_target_by_quasi_ids[row[:-1]] = {}
            syn_target_by_quasi_ids[row[:-1]][row[-1]] = syn_target[row]
        unique_record_quasi_ids = tuple(val for col, val in unique_record.items() if col != sens_attribute)

        target_sensitive_value = list(syn_target_by_quasi_ids[unique_record_quasi_ids].keys())[0]
        votes, distance, solution_sensitive_value = defaultdict(int), 0, None
        while solution_sensitive_value is None:
            neighbors = [_row[:-1] for _row in syn_solution if hamming_distance(_row[:-1], unique_record_quasi_ids) == distance]
            for neighbor, sensitive_value in itertools.product(neighbors, range(domain_config[sens_attribute])):
                if neighbor + (sensitive_value, ) in syn_solution:
                    votes[sensitive_value] += syn_solution[neighbor + (sensitive_value, )]
            if len(votes) > 0:
                sorted_values = sorted(votes.items(), key=lambda x: x[1], reverse=True)
                solution_sensitive_value = sorted_values[0][0]

                proba = np.zeros(domain_config[sens_attribute])
                if len(sorted_values) == 1:
                    proba[solution_sensitive_value] = 1
                elif len(sorted_values) > 1:
                    total_votes = sum([sorted_values[idx][1] for idx in range(len(sorted_values))])
                    for idx in range(len(sorted_values)):
                        proba[sorted_values[idx][0]] = sorted_values[idx][1] / total_votes
                    if sorted_values[0][1] == sorted_values[1][1]:  # if tie
                        solution_sensitive_value = np.random.choice([sorted_values[0][0], sorted_values[1][0]])
            else:
                distance += 1
            aia_guess = 1 if distance > 0 else 0
        aia_result = int(target_sensitive_value == solution_sensitive_value)

        # Append the logs with evaluation results
        eval_time = time.time() - eval_start_time
        file_log = ["Quasi-id: {}, Found in solution: {}".format(unique_record_quasi_ids, False if aia_guess == 1 else True)]
        file_log.append(f'Proba {proba} y_test {target_sensitive_value} y_pred {solution_sensitive_value}\n')
        with open(os.path.join(model_save_dir, "logs.txt"), "a") as fp:
            fp.write("\nEvaluation time (seconds): {}\n".format(eval_time))
            fp.write("\n".join(file_log))
        return aia_result
