import os
import pickle
import numpy as np
import pandas as pd
import multiprocessing
from src.utils.parser import Parser
from src.utils.dataset import DatasetLoader

if __name__ == "__main__":
    """
    Launch target user attribute inference attacks with python script or cmdline. 
    """
    parser = Parser()
    args = parser.parse_cmdline_args()
    
    if args.chosen_gpus == -1:
        multiprocessing.set_start_method('fork')
    else:
        multiprocessing.set_start_method('spawn')

    repo_directory = os.path.dirname(os.path.abspath(__file__))
    args_list, model_inputs = [], {}
    for arg, value in vars(args).items():
        if "attack" in arg:
            break
        if arg in ["dataset", "aggregate_seed", "test_size", "num_aggregates"]:
            if type(value) is not list:
                args_list.append("{}-{}".format(arg, value))
            else:
                args_list.append("{}-{}".format(arg, "_".join([str(x) for x in value])))
    save_dir = os.path.join(repo_directory, "tmp" if len(args.note) == 0 else f"tmp_{args.note}", "load", *args_list)

    for filename in args.filenames:
        # Processing datasets
        if ("ppmf" not in args.dataset) and ("acs" not in args.dataset):
            raise NotImplementedError("This repo is only instantiated with PPMF/ACS dataset")
        data = DatasetLoader(os.path.join(repo_directory, "datasets"), args.dataset, filename)
        config = data.get_config()
        sens_attribute = data.sens_attribute
        if not os.path.exists(os.path.join(save_dir, filename)):
            os.makedirs(os.path.join(save_dir, filename))

        # Processing queries
        if ("ppmf" not in args.dataset) and ("acs" not in args.dataset):
            raise NotImplementedError("This repo is only instantiated with PPMF/ACS dataset")
        queries_path = os.path.join(repo_directory, "datasets", args.dataset, "queries", f'{filename}-set.pkl')
        with open(queries_path, 'rb') as fp:
            queries = pickle.load(fp)
        queries = list(dict.fromkeys([tuple((col.lower(), tuple(values.tolist())) for col, values in query.items()) for query in queries]))

        # Dataset of interest
        # Apply our setup on sensitive attribute
        np.random.seed(0)
        if args.mode == "mia":
            # Since we are doing MIA, then we add one to the size of test-split
            # We don't apply for a resample here, since the task is inferring the membership or not
            aux_split, test_split = data.split_dataset(test_size = args.test_size + 1, aux_size = None)
        else:
            # By default doing AIA, apply resampling on the sensitive attribute
            data.split_dataset(test_size = args.test_size, aux_size = None)
            aux_split, test_split = data.apply_resample()            

        # Select a subset of queries and evaluate aggregates on the test_split
        np.random.seed(args.seed)
        released_queries_indexs = np.random.permutation(range(len(queries)))[:args.num_aggregates]
        released_queries = [q for idx, q in enumerate(queries) if idx in released_queries_indexs]

        # Evaluate queries on test_split to obtain aggregates
        released_aggregates = {}
        if args.mode == "mia":
            # Release clean aggregates for MIA target datasets.
            for query_idx, query in enumerate(released_queries):
                records_touched_by_query = test_split.copy()
                for col, values in query:
                    records_touched_by_query = records_touched_by_query[records_touched_by_query[col].isin(values)]
                released_aggregates[query] = len(records_touched_by_query)
        elif args.mode == "aia":
            # In AIA, noise is enabled only when noise_budget is explicitly provided.
            if args.noise_budget is not None:
                np.random.seed(args.noise_seed)
                noise_vector = np.round(np.random.laplace(0, 1 / args.noise_budget, size = len(released_queries)))
            for query_idx, query in enumerate(released_queries):
                records_touched_by_query = test_split.copy()
                for col, values in query:
                    records_touched_by_query = records_touched_by_query[records_touched_by_query[col].isin(values)]
                released_aggregates[query] = len(records_touched_by_query)
                if args.noise_budget is not None:
                    released_aggregates[query] += noise_vector[query_idx]
        else:
            raise NotImplementedError("This repo only contains MIA/AIA")
        
        if args.aux_type == "syn":
            np.random.seed(0)
            idxs = np.random.choice(len(data.all_records), len(data.all_records), replace=False)
            new_aux_idxs, new_test_idxs = idxs[args.test_size:], idxs[:args.test_size]
            # Generate auxiliary dataset with RAP generator if required
            from src.algo.baseline.rap.rap_aia import RAPAttack
            model_inputs_for_aux = {filename: (released_aggregates, aux_split, test_split, config, sens_attribute)}
            rap_for_aux = RAPAttack(args, model_inputs_for_aux, repo_directory)
            syn_aux_dataset = rap_for_aux.get_syn_aux_dataset(filename)
            quasi_ids = [col for col in test_split.columns if col != sens_attribute]
            # Replace the real-world auxiliary dataset with synthetic ones
            data.auxiliary_split[quasi_ids] = syn_aux_dataset[quasi_ids].sample(n=len(data.auxiliary_split), random_state=0).reset_index(drop=True).values
            aux_split, test_split = data.replace(new_aux_idxs, new_test_idxs)
            
        print("Starting to write query files into disk")
        with open(os.path.join(save_dir, filename, "released_aggregates.pkl"), 'wb') as file:
            pickle.dump(released_aggregates, file)

        print("Starting to write data files into disk")
        aux_split.to_csv(os.path.join(save_dir, filename, "aux_split.csv"), index = True)
        test_split.to_csv(os.path.join(save_dir, filename, "test_split.csv"), index = True)
        with open(os.path.join(save_dir, filename, "config.pkl"), 'wb') as file:
            pickle.dump(config, file)

        # Put all inputs together
        model_inputs[filename] = (released_aggregates, aux_split, test_split, config, sens_attribute)
    print("Data and query processing completed!")

    # Launching attacks
    if args.attack == "rap":
        if args.mode == "aia":
            from src.algo.baseline.rap.rap_aia import RAPAttack
        elif args.mode == "mia":
            from src.algo.baseline.rap.rap_mia import RAPAttack
        rap = RAPAttack(args, model_inputs, repo_directory)
        rap.train()
        rap.eval()

    elif args.attack == "cip":
        if args.mode == "aia":
            from src.algo.baseline.cip.cip_aia import CIPAttack
        elif args.mode == "mia":
            from src.algo.baseline.cip.cip_mia import CIPAttack
        cip = CIPAttack(args, model_inputs, repo_directory)
        cip.attack()
        cip.evaluate()
    
    elif args.attack == "desia":
        """
        1. Solving the Integer programming problem with Gurobi solver, to find the determined target records
        2. For not-determined target records, run shadow modeling and solver attack seperately
        """
        if args.mode == "aia":
            from src.algo.desia.determined_aia import StepOneAttack
            deterministic_attack = StepOneAttack(args, model_inputs, repo_directory)
            deterministic_attack.preprocess()
            deterministic_attack.attack()

            if args.stochastic_method == "stochastic-method":
                from src.algo.desia.shadow_qbs_aia import StepTwoShadowQbsAttack
                stochastic_attack = StepTwoShadowQbsAttack(args, model_inputs, repo_directory)
            # Thresholding attack only applies to binary sensitive attributes
            if config[sens_attribute] == 2:
                if args.stochastic_method == "threshold":
                    from src.algo.desia.threshold import StepTwoThresholdAttack
                    stochastic_attack = StepTwoThresholdAttack(args, model_inputs, repo_directory)
            stochastic_attack.attack()
            stochastic_attack.evaluate()
        elif args.mode == "mia":
            from src.algo.desia.determined_mia import StepOneAttack
            deterministic_attack = StepOneAttack(args, model_inputs, repo_directory)
            deterministic_attack.preprocess()
            deterministic_attack.attack()

            from src.algo.desia.shadow_qbs_mia import StepTwoShadowQbsAttack
            stochastic_attack = StepTwoShadowQbsAttack(args, model_inputs, repo_directory)
            stochastic_attack.attack()
            stochastic_attack.evaluate()
