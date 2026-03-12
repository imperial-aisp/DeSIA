import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import abstractmethod
from multiprocessing import Pool
from ...utils.parser import Parser

class BaseAttack():
    def __init__(self, args: Parser, model_inputs, repo_directory: str):
        """
        Init function.
        Params:
            args: The argument parser Parser instance
            model_inputs: The inputs to construct the solver, containing a tuple of (released_queries, aux_split, test_split, config)
            repo_directory: The directory of the reconstruction repo.
        """
        self.row_indent = "    "
        self.args = args
        self.model_inputs = model_inputs
        
        args_list = []
        for arg, value in vars(self.args).items():
            if arg not in ["filenames", "attack", "multi_cpus", "chosen_gpus"]:
                if type(value) is not list:
                    args_list.append("{}-{}".format(arg, value))
                else:
                    args_list.append("{}-{}".format(arg, "_".join([str(x) for x in value])))
        self.save_dir = os.path.join(repo_directory, "tmp" if len(self.args.note) == 0 else f"tmp_{self.args.note}", "model", "{}".format(self.args.attack), *args_list, "filename-{}")
        
        self.majority_races = ["white", "black_or_african_american", "american_indian_and_alaska_native", "asian", "native_hawaiian_and_other_pacific_islander", "some_other_race"]

        if len(set(range(torch.cuda.device_count())).intersection(self.args.chosen_gpus)) > 0:
            self.devices = [torch.device('cuda:{}'.format(gpu_idx)) for gpu_idx in range(torch.cuda.device_count()) if (torch.cuda.is_available() and (gpu_idx in self.args.chosen_gpus))]
        else:
            self.devices = [torch.device('cpu')]
    
    def _setup_potential_config(self, domain_config: dict):
        potential_record_config, race_start_idx = {}, 0
        for col_idx, (col, val) in enumerate(domain_config.items()):
            if col != "cenrace":
                potential_record_config[col] = list(range(val))
            else:
                race_start_idx = col_idx
                for majority_race in self.majority_races:
                    potential_record_config[majority_race] = [0, 1]
        return race_start_idx, potential_record_config       

    def preprocess(self):
        if self.args.mode == "mia":
            from ...utils.sample_mia import filter_queries, get_query2indices_of_full_dataset
        else:
            from ...utils.sample_aia import filter_queries, get_query2indices_of_full_dataset

        for filename in self.model_inputs:
            released_aggregates, aux_dataset, target_dataset, domain_config, sens_attribute = self.model_inputs[filename]
            if not os.path.exists(self.save_dir.format(filename)):
                os.makedirs(self.save_dir.format(filename))
            # Unique record constraints, adding them into query evaluation
            quasi_ids = [col for col in target_dataset.columns if col != sens_attribute]
            unique_mask = target_dataset.groupby(by = quasi_ids).transform('size') == 1
            unique_records = target_dataset[unique_mask]
            quasi_ids_only_released_queries = filter_queries(list(released_aggregates.keys()), sens_attribute)
            quasi_ids_only_released_queries += [tuple((col, (record[col],)) for col in quasi_ids) for _, record in unique_records.iterrows()]            
            query2indices = get_query2indices_of_full_dataset(pd.concat([aux_dataset, target_dataset]).sort_index(), quasi_ids_only_released_queries, domain_config)
            unique_records.to_csv(os.path.join(self.save_dir.format(filename), "unique_records.csv"), index = True)
            with open(os.path.join(self.save_dir.format(filename), "query2indices.pkl"), "wb") as fp:
                pickle.dump(query2indices, fp)

    @abstractmethod
    def _attack(self):
        pass

    @abstractmethod
    def attack(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
