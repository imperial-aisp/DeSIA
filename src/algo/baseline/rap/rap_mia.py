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

    def train(self):
        np.random.seed(0)
        pbar = tqdm(enumerate(self.model_inputs))
        for target_qbs_idx, filename in pbar:
            pbar.set_description("Training the generator model of {}/{}th dataset".format(target_qbs_idx + 1, len(self.model_inputs)))
            released_aggregates, _, target_dataset, domain_config, sens_attribute = self.model_inputs[filename]

            # Unique quasi-id records are attack/eval targets only.
            quasi_ids = [col for col in target_dataset.columns if col != sens_attribute]
            unique_mask = target_dataset.groupby(by = quasi_ids).transform('size') == 1
            unique_records = target_dataset[unique_mask]
            if not os.path.exists(self.base_dir.format(filename)):
                os.makedirs(self.base_dir.format(filename))
            unique_records.to_csv(os.path.join(self.base_dir.format(filename), "unique_records.csv"), index = True)

            # Multiprocessing on training models for different records
            if self.args.multi_cpus is not None and self.args.multi_cpus > 1:
                process_queue = []
                for process_idx, (unique_record_idx, _) in enumerate(unique_records.iterrows()):
                    # Add coin filp to decide which record to drop
                    np.random.seed(unique_record_idx)
                    target_record_in_target_dataset = np.random.randint(0, domain_config[sens_attribute])
                    if target_record_in_target_dataset == 0:
                        # Drop the target record
                        tmp_target_dataset = target_dataset.drop(unique_record_idx)
                    else:
                        # Drop a record other than target record
                        unique_records_indexs = unique_records.index.tolist()
                        unique_records_indexs.remove(unique_record_idx)
                        dropped_record_idx = np.random.choice(unique_records_indexs)
                        tmp_target_dataset = target_dataset.drop(dropped_record_idx)
                    
                    # Compute released aggregates on target dataset
                    input_constraints = {}
                    for query in released_aggregates:
                        records_touched_by_query = tmp_target_dataset.copy()
                        for col, values in query:
                            records_touched_by_query = records_touched_by_query[records_touched_by_query[col].isin(values)]
                        input_constraints[query] = len(records_touched_by_query)
                    
                    process = Process(target = self._train_worker, args=(process_idx, unique_record_idx, target_record_in_target_dataset, input_constraints, tmp_target_dataset, filename))
                    process.start()
                    process_queue.append(process)
                    while len(process_queue) == self.args.multi_cpus:
                        time.sleep(0.1)
                        for proc_idx, process in enumerate(process_queue):
                            if not process.is_alive():
                                process.join()
                                process_queue.pop(proc_idx)
                # Finish the last bunch of processes
                while len(process_queue) > 0:
                    time.sleep(0.1)
                    for proc_idx, process in enumerate(process_queue):
                        if not process.is_alive():
                            process.join()
                            process_queue.pop(proc_idx)
    
    def _train_worker(self, process_idx, unique_record_idx, target_record_in_target_dataset, input_constraints, tmp_target_dataset, filename):
        train_start_time = time.time()
        _, init_dataset, _, domain_config, _ = self.model_inputs[filename]
        syndata_size = len(tmp_target_dataset)
        domain = Domain(domain_config.keys(), domain_config.values())
        data, aux_data = get_data(tmp_target_dataset, domain), get_data(init_dataset, domain)
        aux_onehot_split = get_data_onehot(aux_data)

        record_dir = os.path.join(self.base_dir, "record-{}")
        model_save_dir = record_dir.format(filename, unique_record_idx)
        query_manager = self.get_qm(data, list(input_constraints.keys()), domain, self.devices[process_idx % len(self.devices)])
        # If density parameter is False, return the number of records. Else, return percentage of records.
        true_answers = query_manager.get_answers(data, density = self.density)

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
            df_syn_raw_solution = syn_generator.G.get_syndata(num_samples = syndata_size, how = "sample").df
            # Recollect garbage to release memory 
            del syn_generator, model_dict
            # Save synthetic datasets
            df_syn_solution = df_syn_raw_solution.groupby(by = tmp_target_dataset.columns.tolist()).size().reset_index(name="count").sort_values(by = ["count"], ascending = False)
            df_syn_solution.to_csv(os.path.join(model_save_dir, "solution_by_unique_rows-{}-{}.csv".format(unique_record_idx, model_idx)), index = False)
        del query_manager, data, true_answers

        df_syn_target = tmp_target_dataset.groupby(by = tmp_target_dataset.columns.tolist()).size().reset_index(name="count").sort_values(by = ["count"], ascending = False)
        df_syn_target.to_csv(os.path.join(model_save_dir, "target_by_unique_rows-{}.csv".format(unique_record_idx)), index = False)
        train_time = time.time() - train_start_time
        with open(os.path.join(model_save_dir, "logs.txt"), "w") as fp:
            fp.write("Target record is a member in private dataset: {}\n".format(True if target_record_in_target_dataset == 1 else False))
            fp.write("Train time (seconds): {}\n".format(train_time))

    def eval(self):
        for target_qbs_idx, filename in enumerate(self.model_inputs):
            unique_records = pd.read_csv(os.path.join(self.base_dir.format(filename), "unique_records.csv"), index_col = 0)
            pbar = tqdm(unique_records.iterrows())
            for indexed_row in pbar:
                pbar.set_description("Evaluating dataset {}".format(filename))
                self._eval((indexed_row, filename))

    def _eval(self, _input_args):
        eval_start_time = time.time()
        (unique_record_idx, unique_record), filename = _input_args

        record_dir = os.path.join(self.base_dir, "record-{}")
        model_save_dir = record_dir.format(filename, unique_record_idx)
        unique_record = tuple(val for _, val in unique_record.items())

        file_log, total_mia_guess = [], []
        for model_idx in range(self.args.num_models):
            df_syn_solution = pd.read_csv(os.path.join(model_save_dir, "solution_by_unique_rows-{}-{}.csv".format(unique_record_idx, model_idx)))
            syn_solution = {tuple(row[:-1].tolist()): row[-1] for _, row in df_syn_solution.iterrows()}

            if unique_record in syn_solution:
                mia_guess = 1
            else:
                mia_guess = 0
            total_mia_guess.append(mia_guess)
            file_log.append("Target record is a member in solution-{}: {}".format(model_idx, True if mia_guess == 1 else False))
        
        eval_time = time.time() - eval_start_time    
        with open(os.path.join(model_save_dir, "logs.txt".format(unique_record_idx)), "a") as fp:
            fp.write("\nEvaluation time (seconds): {}\n".format(eval_time))
            fp.write("\nThere are {}/{} solutions with a membership of target record {}".format(sum(total_mia_guess), self.args.num_models, unique_record))
            fp.write("\n".join(file_log))
