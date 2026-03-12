import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import abstractmethod
from multiprocessing import Pool
from ....utils.parser import Parser

class SolverBaseAttack():
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
        
        if len(set(range(torch.cuda.device_count())).intersection(self.args.chosen_gpus)) > 0:
            self.devices = [torch.device('cuda:{}'.format(gpu_idx)) for gpu_idx in range(torch.cuda.device_count()) if (torch.cuda.is_available() and (gpu_idx in self.args.chosen_gpus))]
        else:
            self.devices = [torch.device('cpu')]
        
    def attack(self):
        """
        Reconstruct the dataset by solving the Integer programming problem with Gurobi solver
        Params:
            num_solutions: The maximum number of feasible solutions returned by the solver
        """
        for filename in self.model_inputs:
            _, _, target_dataset, _, sens_attribute = self.model_inputs[filename]
            if not os.path.exists(self.save_dir.format(filename)):
                os.makedirs(self.save_dir.format(filename))
            quasi_ids = [col for col in target_dataset.columns if col != sens_attribute]
            unique_mask = target_dataset.groupby(by = quasi_ids).transform('size') == 1
            unique_records = target_dataset[unique_mask]
            if not os.path.exists(self.save_dir.format(filename)):
                os.makedirs(self.save_dir.format(filename))
            unique_records.to_csv(os.path.join(self.save_dir.format(filename), "unique_records.csv"), index = True)

            if self.args.mode == "aia":
                pbar = tqdm(enumerate(unique_records.iterrows()))
                for row_idx, indexed_row in pbar:
                    pbar.set_description("Generating Gurobi script of unique record {}/{} in dataset {}".format(row_idx + 1, len(unique_records), filename))
                    self._attack((filename, indexed_row))
            elif self.args.mode == "mia":
                pbar = tqdm(enumerate(unique_records.iterrows()))
                for row_idx, indexed_row in pbar:
                    pbar.set_description("Generating Gurobi script of unique record {}/{} in dataset {}".format(row_idx + 1, len(unique_records), filename))
                    self._attack((filename, indexed_row, unique_records))

    def evaluate(self):
        """
        Evaluate the results
        Params: None
        """
        for filename in self.model_inputs:
            unique_records = pd.read_csv(os.path.join(self.save_dir.format(filename), "unique_records.csv"), index_col = 0)
            pbar = tqdm(unique_records.iterrows())
            for indexed_row in pbar:
                pbar.set_description("Evaluating dataset {}".format(filename))
                self._evaluate((filename, indexed_row))

    @abstractmethod    
    def generate_script(self):
        pass

    @abstractmethod    
    def _evaluate(self):
        pass

    @abstractmethod
    def _attack(self):
        pass

    @abstractmethod
    def _evaluate(self):
        pass
