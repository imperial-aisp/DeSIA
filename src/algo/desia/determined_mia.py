import itertools
from .base import *
from collections import defaultdict
from ...utils.general import _find_potential_records, _generate_constraints, _to_potential_record

class StepOneAttack(BaseAttack):
    """
    The class implementing step-one finding out determined records (proven to be correct prediction) in attribute inference attack
    """
    def __init__(self, args: Parser, model_inputs, repo_directory: str):
        super().__init__(args, model_inputs, repo_directory)

    def _step_one_generate_script(self, input_constraints: dict, indexed_row: pd.Series, filename: str):
        """
        Step one, generate script to find out whether the target record is determined.
        Params: 
            input_constraints: The marginal constraints to build a constraint satisfaction problem.
                    The key is the marginal querying conditions, while the value is the number of rows satisfying the query of the key.
            indexed_row: The target record with record index
            filename: The filename of the target dataset
        """
        _, _, target_dataset, domain_config, sens_attribute = self.model_inputs[filename]
        unique_record_idx, unique_record = indexed_row
        row_constraint = len(target_dataset)

        # Check whether the Gurobi license is stored under ~/
        _step_one_script = "import os\nimport re\nimport sys\nimport pickle\nimport zipfile\nimport itertools\nimport gurobipy as gp\nimport pandas as pd\nfrom numpy import random\nfrom gurobipy import GRB, quicksum\nfrom collections import defaultdict"
        # Control the randomness of the Gurobi solver here
        _step_one_script += "\nV, C, index_to_var = {}, {}, {}"
        _step_one_script += "\nwith gp.Model() as model:"
        _step_one_script += "\n{}model.Params.PoolSearchMode = 2".format(self.row_indent)
        _step_one_script += "\n{}model.Params.PoolSolutions = 1".format(self.row_indent)
        _step_one_script += "\n{}model.Params.Seed = 0".format(self.row_indent)
        _step_one_script += "\n{}model.Params.Threads = {}".format(self.row_indent, 1 if self.args.multi_cpus is None else 5)
        _step_one_script += "\n{}model.Params.LogToConsole = 0".format(self.row_indent)

        # Define the config for potential records 
        race_start_idx, potential_record_config = self._setup_potential_config(domain_config)
        all_potential_records = []
        if set(self.majority_races).issubset(set(potential_record_config.keys())):
            for potential_record in itertools.product(*(potential_record_config.values())):
                # No person will have zero majority_race
                # At least each person will have equal to or more than one majority_race
                if np.sum(potential_record[race_start_idx: race_start_idx + len(self.majority_races)]) > 0:
                    all_potential_records.append(potential_record)
        else:
            for potential_record in itertools.product(*(potential_record_config.values())):
                all_potential_records.append(potential_record)
        all_potential_records_to_indexs = {record: idx for idx, record in enumerate(all_potential_records)}
        target_potential_record = _to_potential_record((domain_config, unique_record))

        target_quasi_id_potential_record_indexs = []
        for value in range(domain_config[sens_attribute]):
            target_quasi_id_record = unique_record.copy()
            target_quasi_id_record[sens_attribute] = value
            target_quasi_id_potential_record = _to_potential_record((domain_config, target_quasi_id_record))
            target_quasi_id_potential_record_indexs.append(all_potential_records_to_indexs[target_quasi_id_potential_record])

        # Constraints encoded by queries
        potential_records_by_each_query, non_zero_potential_records = {}, {}
        pbar = tqdm(zip(list(input_constraints.items()), [domain_config] * len(input_constraints)))
        for _input_args in pbar:
            pbar.set_description("Find out potential records being queried")
            query, potential_records = _find_potential_records(tuple(_input_args))
            potential_records_by_each_query[query] = list(potential_records)

        upper_bounds_potential_records = defaultdict(list)
        for query in potential_records_by_each_query:
            for record in potential_records_by_each_query[query]:
                upper_bounds_potential_records[record].append(input_constraints[query])
        min_upper_bound_potential_records = {record: min(upper_bounds_potential_records[record]) for record in upper_bounds_potential_records}

        # Define unknown variables
        tmp_variable_scripts = []
        pbar = tqdm(all_potential_records)
        pbar.set_description("Generate unknown variable for each potential record in the Gurobi model")
        for record_idx, record in enumerate(pbar):
            if record in min_upper_bound_potential_records:
                tmp_variable_scripts.append("\n{}V[{}] = model.addVar(lb = 0, ub = {}, vtype='I')".format(self.row_indent, all_potential_records_to_indexs[record], min_upper_bound_potential_records[record]))
                # Take notes of such potential record which value is always zero
                if min_upper_bound_potential_records[record] > 0:
                    non_zero_potential_records[record] = True
            else:
                tmp_variable_scripts.append("\n{}V[{}] = model.addVar(lb = 0, ub = {}, vtype='I')".format(self.row_indent, all_potential_records_to_indexs[record], row_constraint))
                non_zero_potential_records[record] = True
            tmp_variable_scripts.append("\n{}index_to_var[{}] = {}".format(self.row_indent, record_idx, record))
        print("{}/{} potential records may have non-zero values".format(len(non_zero_potential_records), len(all_potential_records)))

        # Define constraints
        non_zero_potential_record_indexs = {record: all_potential_records_to_indexs[record] for record in non_zero_potential_records}
        constraint_variables = ["V[{}]".format(non_zero_potential_record_indexs[record]) for record in non_zero_potential_records]
        # Add the sum of unknown variables equal to the number of rows at the end
        tmp_constraint_scripts = ["\n{}constraint_variables = [{}]".format(self.row_indent, ",".join(constraint_variables))
                                + "\n{}C[()] = model.addConstr(quicksum(constraint_variables) == {})".format(self.row_indent, row_constraint)]
        # Generate constraints for non unique-record-queries
        pbar = tqdm(zip(list(input_constraints.items()), 
                        list(potential_records_by_each_query.values()),
                        [self.row_indent] * len(input_constraints), 
                        [non_zero_potential_record_indexs] * len(input_constraints)))
        pbar.set_description("Generate constraints in the Gurobi model")
        for _input_args in pbar:
            tmp_constraint_scripts.append(_generate_constraints(tuple(_input_args)))

        # Assemble scripts together
        _step_one_script += "".join(tmp_variable_scripts)
        _step_one_script += "\n{}print('Variables setup finished')".format(self.row_indent)
        _step_one_script += "".join(sorted(tmp_constraint_scripts, key = lambda x: len(x)))
        _step_one_script += "\n{}print('Constraints setup finished')".format(self.row_indent)
        # _step_one_script += "\n{}model.setParam('Presolve', 2)".format(self.row_indent)
        _step_one_script += "\n{}model.setObjective(0)".format(self.row_indent)
        _step_one_script += "\n{}model.optimize()".format(self.row_indent)
        # New null query
        _step_one_script += "\n{}if model.Status == 2:".format(self.row_indent)
        _step_one_script += "\n{}{}target_record_count = V[{}].Xn".format(self.row_indent, self.row_indent, all_potential_records_to_indexs[target_potential_record])
        _step_one_script += "\n{}{}if target_record_count == 0:".format(self.row_indent, self.row_indent)
        _step_one_script += "\n{}{}{}model.addConstr(V[{}] == 1)".format(self.row_indent, self.row_indent, self.row_indent, all_potential_records_to_indexs[target_potential_record])
        _step_one_script += "\n{}{}else:".format(self.row_indent, self.row_indent)
        _step_one_script += "\n{}{}{}model.addConstr(V[{}] == 0)".format(self.row_indent, self.row_indent, self.row_indent, all_potential_records_to_indexs[target_potential_record])

        _step_one_script += "\n{}{}model.setObjective(0)".format(self.row_indent, self.row_indent)
        _step_one_script += "\n{}{}model.optimize()".format(self.row_indent, self.row_indent)
        _step_one_script += "\n{}{}if model.Status == 2:".format(self.row_indent, self.row_indent)
        _step_one_script += "\n{}{}{}output_solution = {{}}".format(self.row_indent, self.row_indent, self.row_indent)
        _step_one_script += "\n{}{}{}for potential_record_idx in V:".format(self.row_indent, self.row_indent, self.row_indent)
        _step_one_script += "\n{}{}{}{}output_solution[index_to_var[potential_record_idx]] = round(V[potential_record_idx].Xn)".format(self.row_indent, self.row_indent, self.row_indent, self.row_indent)
        _step_one_script += "\n{}{}{}with open('{}', 'wb') as fp:".format(self.row_indent, self.row_indent, self.row_indent, os.path.join(self.save_dir.format(filename), "feasible-solutions-{}_.pkl".format(unique_record_idx)))
        _step_one_script += "\n{}{}{}{}pickle.dump(output_solution, fp)".format(self.row_indent, self.row_indent, self.row_indent, self.row_indent)
        # First run: For unfeasible models, kill.
        _step_one_script += "\n{}if model.Status == 3:".format(self.row_indent)
        _step_one_script += "\n{}{}exit(1)".format(self.row_indent, self.row_indent)

        print("Starting to write {}".format(len(_step_one_script)))
        with open(os.path.join(self.save_dir.format(filename), "script_1-{}.py".format(unique_record_idx)), "w") as fp:
            fp.write(_step_one_script)
    
    def _attack(self, _input_args):
        """
        Step-one attack per record instance
        Params:
            filename: The filename of the target dataset
            indexed_row: The target record and its index
        """
        step_one_attack_start_time = time.time()
        filename, unique_records, indexed_row = _input_args
        unique_record_idx, _ = indexed_row
        released_aggregates, _, target_dataset, domain_config, sens_attribute = self.model_inputs[filename]

        # Add coin filp to decide which record to drop
        np.random.seed(unique_record_idx)
        target_record_in_target_dataset = np.random.randint(0, domain_config[sens_attribute])
        if target_record_in_target_dataset == 0:
            tmp_target_dataset = target_dataset.drop(unique_record_idx)
        else:
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

        self._step_one_generate_script(input_constraints, indexed_row, filename)
        print("Start running the solver")
        os.system("python3 {} > /dev/null".format(os.path.join(self.save_dir.format(filename), "script_1-{}.py".format(unique_record_idx))))
        os.remove(os.path.join(self.save_dir.format(filename), "script_1-{}.py".format(unique_record_idx)))
        step_one_attack_time = time.time() - step_one_attack_start_time

        tmp_target_dataset.to_csv(os.path.join(self.save_dir.format(filename), "target-{}.csv".format(unique_record_idx)), index = True)
        with open(os.path.join(self.save_dir.format(filename), "input_constraint-{}.pkl".format(unique_record_idx)), "wb") as fp:
            pickle.dump(input_constraints, fp)
        # Save the attack time cost
        with open(os.path.join(self.save_dir.format(filename), "logs-{}.txt".format(unique_record_idx)), "w") as fp:
            fp.write("Target record is a member in private dataset: {}\n".format(True if target_record_in_target_dataset == 1 else False))
            fp.write("Step-one attack time (seconds): {}\n".format(step_one_attack_time))

    def attack(self):
        """
        Step-one attack, finding out whether the given target record is determined in reconstruction or not
        Params: None
        """
        for filename in self.model_inputs:
            unique_records = pd.read_csv(os.path.join(self.save_dir.format(filename), "unique_records.csv"), index_col = 0)
            
            if self.args.multi_cpus is not None and self.args.multi_cpus > 1:
                with Pool(maxtasksperchild = 100000, processes = self.args.multi_cpus) as pool:
                    pbar = tqdm(pool.imap_unordered(self._attack, zip(
                        [filename] * len(unique_records),
                        [unique_records] * len(unique_records),
                        list(unique_records.iterrows()))))
                    for row_idx, _ in enumerate(pbar):
                        pbar.set_description("Step one attack on unique record {}/{} in dataset {}".format(row_idx + 1, len(unique_records), filename))
            else:
                pbar = tqdm(enumerate(unique_records.iterrows()))
                for row_idx, indexed_row in pbar:
                    pbar.set_description("Step one attack on unique record {}/{} in dataset {}".format(row_idx + 1, len(unique_records), filename))
                    self._attack((filename, unique_records, indexed_row))
            
            # Find out the non-determined records and write them into another dataframe
            unique_records_not_determined = []
            for indexed_row in unique_records.iterrows():
                if os.path.exists(os.path.join(self.save_dir.format(filename), "feasible-solutions-{}_.pkl".format(indexed_row[0]))):
                    unique_records_not_determined.append(indexed_row)
            unique_records_not_determined_df = pd.DataFrame([row for _, row in unique_records_not_determined])
            unique_records_not_determined_df.index = [index for index, _ in unique_records_not_determined]
            unique_records_not_determined_df.to_csv(os.path.join(self.save_dir.format(filename), "unique_records_not_determined.csv"), index = True)
