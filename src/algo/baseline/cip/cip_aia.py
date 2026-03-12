import time
import pickle
import itertools
import pandas as pd
from .base import *
from collections import defaultdict
from ....utils.general import race_to_id, CensusRace, hamming_distance, _find_potential_records, _generate_constraints, _to_potential_record

class CIPAttack(SolverBaseAttack):
    """
    The class implementing reconstruction attacks by solving constraint satisfaction problems
    This method is presented in the paper The 2010 Census Confidentiality Protections Failed Here is How and Why
    """
    def __init__(self, args: Parser, model_inputs, repo_directory: str):
        super().__init__(args, model_inputs, repo_directory)
        self.majority_races = ["white", "black_or_african_american", "american_indian_and_alaska_native", "asian", "native_hawaiian_and_other_pacific_islander", "some_other_race"]

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

    def prepare_script(self, input_constraints: dict, filename: str):        
        """
        Generate the python program running solver to solve the IP problem.
        Params: 
            input_constraints: The marginal constraints to build a constraint satisfaction problem.
                    The key is the marginal querying conditions, while the value is the number of rows satisfying the query of the key.
            filename: The filename of the target dataset
        """
        _, aux_dataset, target_dataset, domain_config, _ = self.model_inputs[filename]
        row_constraint = len(target_dataset)

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
        tmp_constraint_scripts = ["\n{}constraint_variables = [{}]".format(self.row_indent, ",".join(constraint_variables)) + "\n{}C[()] = model.addConstr(quicksum(constraint_variables) == {})".format(self.row_indent, row_constraint)]
        # Generate constraints from released aggregate queries.
        pbar = tqdm(zip(list(input_constraints.items()), 
                        list(potential_records_by_each_query.values()),
                        [self.row_indent] * len(input_constraints), 
                        [non_zero_potential_record_indexs] * len(input_constraints)))
        pbar.set_description("Generate constraints in the Gurobi model")
        for _input_args in pbar:
            tmp_constraint_scripts.append(_generate_constraints(tuple(_input_args)))

        # If auxiliary dataset is used for initializing the unknwon variables
        tmp_init_scripts = []
        if self.args.init_solver is True:
            aux_record_counts = aux_dataset.value_counts().to_dict()
            aux_potential_record_counts = defaultdict(int)
            # Convert into the counts of potential records
            pbar = tqdm(aux_record_counts)
            pbar.set_description("Initialize value for unknown variables in the Gurobi model")
            for aux_record in pbar:
                aux_potential_record = _to_potential_record((domain_config, aux_record))
                aux_potential_record_counts[aux_potential_record] = aux_record_counts[aux_record]
            aux_test_size_ratio = len(aux_dataset) / len(target_dataset)
            for record in non_zero_potential_records:
                # Do not cause value conflicts for unknown variables when initialzing them with auxiliary knowledge
                if record in min_upper_bound_potential_records:
                    init_value = min(aux_potential_record_counts[record] // aux_test_size_ratio, min_upper_bound_potential_records[record])
                    if init_value > 0:
                        tmp_init_scripts += ["\n{}V[{}].start = {}".format(self.row_indent, non_zero_potential_record_indexs[record], init_value)]
            tmp_init_scripts += "\n{}print('Variables initialization finished')".format(self.row_indent)
            # Change the way of solver taking use of the initial values
            tmp_init_scripts += "\n{}model.setParam('MIPFocus', 2)".format(self.row_indent)
            tmp_init_scripts += "\n{}model.setParam('ImproveStartTime', GRB.INFINITY)".format(self.row_indent)
        return tmp_init_scripts, tmp_variable_scripts, tmp_constraint_scripts

    def generate_script(self, filename, indexed_row, input_seed, scripts):
        tmp_init_scripts, tmp_variable_scripts, tmp_constraint_scripts = scripts
        unique_record_idx, _ = indexed_row

        # Check whether the Gurobi license is stored under ~/
        _census_script = "import os\nimport re\nimport sys\nimport pickle\nimport itertools\nimport gurobipy as gp\nimport pandas as pd\nfrom numpy import random\nfrom gurobipy import GRB, quicksum\nfrom collections import defaultdict"
        # Control the randomness of the Gurobi solver here
        _census_script += "\nV, C, index_to_var = {}, {}, {}"
        _census_script += "\nwith gp.Model() as model:"
        _census_script += "\n{}model.Params.PoolSearchMode = 2".format(self.row_indent)
        _census_script += "\n{}model.Params.PoolSolutions = 1".format(self.row_indent)
        _census_script += "\n{}model.Params.Seed = {}".format(self.row_indent, input_seed)
        _census_script += "\n{}model.Params.Threads = {}".format(self.row_indent, 1 if self.args.multi_cpus is None else 4)
        _census_script += "\n{}model.Params.LogToConsole = 0".format(self.row_indent)
        # Shuffle the variables when building the CP problem
        np.random.seed(input_seed)
        np.random.shuffle(tmp_variable_scripts)
        _census_script += "".join(tmp_variable_scripts)
        _census_script += "\n{}print('Variables setup finished')".format(self.row_indent)
        # Shuffle the constraint when building the CP problem
        np.random.seed(input_seed)
        np.random.shuffle(tmp_constraint_scripts)
        _census_script += "".join(sorted(tmp_constraint_scripts, key = lambda x: len(x)))
        _census_script += "\n{}print('Constraints setup finished')".format(self.row_indent)
        _census_script += "".join(sorted(tmp_init_scripts, key = lambda x: len(x)))

        # Run solver
        # _census_script += "\n{}model.setParam('Presolve', 2)".format(self.row_indent)
        _census_script += "\n{}model.setObjective(0)".format(self.row_indent)
        _census_script += "\n{}model.optimize()".format(self.row_indent)
        # For feasible models, read all feasible solutions and write them into disk
        _census_script += "\n{}if model.Status == 2:".format(self.row_indent)
        _census_script += "\n{}{}output_solution = {{}}".format(self.row_indent, self.row_indent)
        _census_script += "\n{}{}for potential_record_idx in V:".format(self.row_indent, self.row_indent)
        _census_script += "\n{}{}{}output_solution[index_to_var[potential_record_idx]] = round(V[potential_record_idx].Xn)".format(self.row_indent, self.row_indent, self.row_indent)
        _census_script += "\n{}{}with open('{}', 'wb') as fp:".format(self.row_indent, self.row_indent, os.path.join(self.save_dir.format(filename), "feasible-solutions-{}-{}.pkl".format(unique_record_idx, input_seed)))
        _census_script += "\n{}{}{}pickle.dump(output_solution, fp)".format(self.row_indent, self.row_indent, self.row_indent)
        # First run: For models reaching time limit, kill
        _census_script += "\n{}elif model.Status == 9:".format(self.row_indent)
        _census_script += "\n{}{}exit(1)".format(self.row_indent, self.row_indent)
        # First run: For unfeasible models, kill.
        _census_script += "\n{}elif model.Status == 3:".format(self.row_indent)
        _census_script += "\n{}{}exit(2)".format(self.row_indent, self.row_indent)
        
        print("Starting to write {}".format(len(_census_script)))
        with open(os.path.join(self.save_dir.format(filename), "script-{}-{}.py".format(unique_record_idx, input_seed)), "w") as fp:
            fp.write(_census_script)

    def _attack_worker(self, _input_args):
        filename, indexed_row, input_seed, scripts = _input_args
        unique_record_idx, _ = indexed_row
        self.generate_script(filename, indexed_row, input_seed, scripts)
        os.system("python3 {} > /dev/null".format(os.path.join(self.save_dir.format(filename), "script-{}-{}.py".format(unique_record_idx, input_seed))))
        os.remove(os.path.join(self.save_dir.format(filename), "script-{}-{}.py".format(unique_record_idx, input_seed)))
        if self.args.noise_budget is None:
            with open(os.path.join(self.save_dir.format(filename), "feasible-solutions-{}-{}.pkl".format(unique_record_idx, input_seed)), "rb") as fp:
                feasible_solution = pickle.load(fp)
            os.remove(os.path.join(self.save_dir.format(filename), "feasible-solutions-{}-{}.pkl".format(unique_record_idx, input_seed)))
            return feasible_solution
        if os.path.exists(os.path.join(self.save_dir.format(filename), "feasible-solutions-{}-{}.pkl".format(unique_record_idx, input_seed))):
            with open(os.path.join(self.save_dir.format(filename), "feasible-solutions-{}-{}.pkl".format(unique_record_idx, input_seed)), "rb") as fp:
                feasible_solution = pickle.load(fp)
            os.remove(os.path.join(self.save_dir.format(filename), "feasible-solutions-{}-{}.pkl".format(unique_record_idx, input_seed)))
            return feasible_solution
        return None

    def _attack(self, _input_args):
        """
        The attack instance targeting against target record
        Params:
            filename: The filename of the target dataset
            indexed_row: The target record and its index
            num_solutions: The maximum number of feasible solutions returned by the solver
        """
        attack_start_time = time.time()
        filename, indexed_row = _input_args
        released_aggregates, _, target_dataset, domain_config, sens_attribute = self.model_inputs[filename]
        _, potential_record_config = self._setup_potential_config(domain_config)

        unique_record_idx, _ = indexed_row
        assert self.args.mode == "aia"

        # Use released aggregate constraints only (no unique-record constraint).
        scripts = self.prepare_script(released_aggregates, filename)
        feasible_solutions = []
        if self.args.multi_cpus is not None:
            with Pool(maxtasksperchild = 100000, processes = self.args.multi_cpus if (self.args.multi_cpus is not None) and (self.args.multi_cpus > 1) else 1) as pool:
                np.random.seed(0)
                pbar = tqdm(pool.imap_unordered(self._attack_worker, zip(
                    [filename] * self.args.num_solutions,
                    [indexed_row] * self.args.num_solutions,
                    np.random.permutation(10000)[:self.args.num_solutions],
                    [scripts] * self.args.num_solutions)))
                for solution_idx, feasible_solution in enumerate(pbar):
                    pbar.set_description("Generating solution {}/{}".format(solution_idx + 1, self.args.num_solutions))
                    feasible_solutions.append(feasible_solution)
        else:
            pbar = tqdm(np.random.permutation(10000)[:self.args.num_solutions])
            for solution_idx, input_seed in enumerate(pbar):
                pbar.set_description("Generating solution {}/{}".format(solution_idx + 1, self.args.num_solutions))
                feasible_solution = self._attack_worker((filename, indexed_row, input_seed, scripts))
                feasible_solutions.append(feasible_solution)

        # Convert to tabular dataframe
        df_census_raw_solutions = []
        for feasible_solution in feasible_solutions:
            if feasible_solution is None:
                continue
            dict_solution = defaultdict(list)
            for record, counts in feasible_solution.items():
                detail_race = CensusRace()
                cenrace_exist = False
                for col, val in zip(potential_record_config, record):
                    if col in self.majority_races:
                        detail_race.__dict__[col] = bool(val)
                        cenrace_exist = True
                    else:
                        dict_solution[col] += [val] * counts
                if cenrace_exist:
                    dict_solution["cenrace"] += [race_to_id[detail_race]] * counts
            df_census_raw_solutions.append(pd.DataFrame(dict_solution))

        if len(df_census_raw_solutions) > 0:
            df_census_raw_solutions = pd.concat(df_census_raw_solutions)
            # Count the number of times each record appears in solution and target
            df_census_target = target_dataset.groupby(by = target_dataset.columns.tolist()).size().reset_index(name="count").sort_values(by = ["count"], ascending = False)
            df_census_target.to_csv(os.path.join(self.save_dir.format(filename), "target_by_unique_rows-{}.csv".format(unique_record_idx)), index = False)
            df_census_solution = df_census_raw_solutions.groupby(by = target_dataset.columns.tolist()).size().reset_index(name="count").sort_values(by = ["count"], ascending = False)
            df_census_solution.to_csv(os.path.join(self.save_dir.format(filename), "solution_by_unique_rows-{}.csv".format(unique_record_idx)), index = False)

        # Save the attack time cost
        attack_time = time.time() - attack_start_time
        with open(os.path.join(self.save_dir.format(filename), "logs-{}.txt".format(unique_record_idx)), "w") as fp:
            fp.write("Attack time (seconds): {}\n".format(attack_time))

    def _evaluate(self, _input_args):
        """
        Evaluate the results from reconstruction of each file
        Params: 
            filename: The filename of the target dataset
            indexed_row: The target record with record index
        """
        eval_start_time = time.time()
        filename, indexed_row = _input_args
        unique_record_idx, unique_record = indexed_row
        _, _, _, domain_config, sens_attribute = self.model_inputs[filename]
        unique_record_quasi_ids = tuple(val for col, val in unique_record.items() if col != sens_attribute)

        if os.path.exists(os.path.join(self.save_dir.format(filename), "solution_by_unique_rows-{}.csv".format(unique_record_idx))):
            df_census_target = pd.read_csv(os.path.join(self.save_dir.format(filename), "target_by_unique_rows-{}.csv".format(unique_record_idx)))
            df_census_solution = pd.read_csv(os.path.join(self.save_dir.format(filename), "solution_by_unique_rows-{}.csv".format(unique_record_idx)))
            
            # Convert from dataset to dictionary, record_in_tuple_format: conuts
            census_target = {tuple(row[:-1].tolist()): row[-1] for _, row in df_census_target.iterrows()}
            census_solution = {tuple(row[:-1].tolist()): row[-1] for _, row in df_census_solution.iterrows()}

            # By attribute inference: querying upon unique quasi-ids
            census_target_by_quasi_ids = {}
            for row in census_target:
                if row[:-1] not in census_target_by_quasi_ids:
                    census_target_by_quasi_ids[row[:-1]] = {}
                census_target_by_quasi_ids[row[:-1]][row[-1]] = census_target[row]

            target_sensitive_value = list(census_target_by_quasi_ids[unique_record_quasi_ids].keys())[0]
            votes, distance, solution_sensitive_value = defaultdict(int), 0, None
            while solution_sensitive_value is None:
                # Note that neighbors is the list of neighboring quasi-ids
                # Which would be counted twice in neighborhood: neighbor + (0,) and neighbor + (1,)
                neighbors = list(set([_row[:-1] for _row in census_solution if hamming_distance(_row[:-1], unique_record_quasi_ids) == distance]))
                for neighbor, sensitive_value in itertools.product(neighbors, range(domain_config[sens_attribute])):
                    if neighbor + (sensitive_value, ) in census_solution:
                        votes[sensitive_value] += census_solution[neighbor + (sensitive_value, )]
                # If neighbors found under current distance
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
                # Otherwise, extend the neighborhood
                else:
                    distance += 1
                aia_guess = 1 if distance > 0 else 0
            aia_result = int(target_sensitive_value == solution_sensitive_value)
            # Append the logs with evaluation results
            eval_time = time.time() - eval_start_time
            file_log = ["Quasi-id: {}, Found in solution: {}".format(unique_record_quasi_ids, False if aia_guess == 1 else True)]
            file_log.append(f'Proba {proba} y_test {target_sensitive_value} y_pred {solution_sensitive_value}')
        else:
            eval_time = time.time() - eval_start_time
            np.random.seed(int(time.time()))
            aia_result = np.random.randint(0, 2)
            file_log = ["Quasi-id: {}, Found in solution: False".format(unique_record_quasi_ids)]
            file_log.append("Solution infeasible, predict by a coin filp")

        file_log.append("AIA prediction: {}".format("correct" if aia_result == 1 else "wrong"))
        with open(os.path.join(self.save_dir.format(filename), "logs-{}.txt".format(unique_record_idx)), "a") as fp:
            fp.write("\nEvaluation time (seconds): {}\n".format(eval_time))
            fp.write("\n".join(file_log))
        return aia_result
