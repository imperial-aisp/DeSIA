import gc
import re
from .base import *
from collections import defaultdict
from ...utils.general import _find_useful_query, _find_record_indexs
from ...utils.sample_aia import get_train_and_eval_shadow_labels, get_query2answers
import math
from sklearn.metrics import accuracy_score, roc_auc_score

EVAL_FRACTION = 0.3333

class StepTwoThresholdAttack(BaseAttack):
    """
    The class implementing coin flip attribute inference attack
    """

    def __init__(self, args: Parser, model_inputs, repo_directory: str):
        super().__init__(args, model_inputs, repo_directory)

    def get_X_train_eval(self, queries, query2answers):
        X = np.zeros((self.args.num_training_qbses, len(queries)))
        for i, query in enumerate(queries):
            X[:, i] = np.array(query2answers[tuple(sorted(query, key = lambda x: x[0]))])
        train_size = int(self.args.num_training_qbses * (1 - EVAL_FRACTION))
        return X[:train_size, :], X[train_size:, :]

    def get_y_train_eval(self, y_train_eval):
        train_size = int(self.args.num_training_qbses * (1 - EVAL_FRACTION))
        return y_train_eval[:train_size], y_train_eval[train_size:]

    @staticmethod
    def gaussian_pdf(x, mu, std_dev):
        if std_dev == 0:
            std_dev = 0.00000001
        return 1 / (std_dev * (math.pi * 2) ** 0.5) * np.exp(- 0.5 * ((x - mu) / std_dev) ** 2)

    def _attack(self, _input_args):
        step_two_attack_start_time = time.time()
        filename, indexed_row, query2indices = _input_args
        released_aggregates, aux_dataset, target_dataset, domain_config, sens_attribute = self.model_inputs[
            filename]
        # Add unique-record query as one constraint
        unique_record_idx, unique_record = indexed_row
        quasi_ids = [col for col in target_dataset.columns if col != sens_attribute]
        assert self.args.mode == "aia"
        unique_record_query = [tuple((col, (unique_record[col],)) for col in quasi_ids)]
        input_released_queries = list(released_aggregates.keys()) + unique_record_query

        # Set up the device specifically used for shadow modeling
        proc_device = self.devices[unique_record_idx % len(self.devices)]
        # M: contains 1 in row i, column j, if the shadow dataset i contains the user j
        M = np.zeros((self.args.num_training_qbses, len(aux_dataset) + len(target_dataset)))
        # S: contains 1 in row i, column j, if the shadow dataset i contains the user j AND user j has sens=key in dataset i
        S = {(sens_attribute, key): np.zeros((self.args.num_training_qbses, len(aux_dataset) + len(target_dataset)))
             for key in range(domain_config[sens_attribute])}
        _, y_train_val = get_train_and_eval_shadow_labels(self.args, domain_config, pd.DataFrame([unique_record]),
                                                          aux_dataset, target_dataset, M, S)
        query2answers = get_query2answers(pd.concat([aux_dataset, target_dataset]).sort_index(), query2indices, M,
                                          S, proc_device)
        del M, S
        gc.collect()
        X_train, X_val = self.get_X_train_eval(list(released_aggregates.keys()), query2answers)
        # The answers to unique-record-query across all shadow datasets should be evaluated as 1
        X_train, X_val = np.hstack((X_train, np.ones((X_train.shape[0], 1)))), np.hstack(
            (X_val, np.ones((X_val.shape[0], 1))))
        y_train, y_val = self.get_y_train_eval(y_train_val)
        y_train, y_val = np.squeeze(np.array(y_train)), np.squeeze(np.array(y_val))

        useful_queries = []
        useful_query_indices = []
        for query_idx, query in enumerate(input_released_queries):
            useful_query = _find_useful_query((query, unique_record.to_dict(), sens_attribute))
            if (useful_query is not None):
                useful_queries.append(useful_query)
                useful_query_indices.append(query_idx)

        queryindx2likelihood_ratio = {}

        mask_1_train = (y_train == 1)
        mask_0_train = (y_train == 0)

        y_val_pred = []
        y_val_prob = []

        for query_idx, query in zip(useful_query_indices, useful_queries):
            all_answers = X_train[:, query_idx]
            mu_1_train = np.mean(all_answers[mask_1_train])
            sigma_1_train = np.std(all_answers[mask_1_train])
            mu_0_train = np.mean(all_answers[mask_0_train])
            sigma_0_train = np.std(all_answers[mask_0_train])

            assert query[-1][0] == 'cenhisp'
            queryindx2likelihood_ratio[query_idx] = {
                'mu_1': mu_1_train,
                'sigma_1': sigma_1_train,
                'mu_0': mu_0_train,
                'sigma_0': sigma_0_train,

            }
        for i in range(len(y_val)):
            votes = defaultdict(int)
            correct_y_val = y_val[i]
            if len(useful_queries) > 0:
                for query_idx, query in zip(useful_query_indices, useful_queries):
                    sample = X_val[i, query_idx]
                    likelihood1 = self.gaussian_pdf(sample, queryindx2likelihood_ratio[query_idx]['mu_1'], queryindx2likelihood_ratio[query_idx]['sigma_1'])
                    likelihood0 = self.gaussian_pdf(sample, queryindx2likelihood_ratio[query_idx]['mu_0'], queryindx2likelihood_ratio[query_idx]['sigma_0'])
                    if likelihood1 >= likelihood0:
                        votes[1] += 1
                    else:
                        votes[0] += 1
                y_val_prob_tmp = [votes[0] / (votes[0] + votes[1]), votes[1] / (votes[0] + votes[1])]
                y_val_pred_tmp = np.argmax(y_val_prob_tmp)
                y_val_prob.append(y_val_prob_tmp)
                y_val_pred.append(y_val_pred_tmp)
            else:
                # If no useful queries. then random toos a coin to decide the output
                y_val_prob_tmp = [0.5, 0.5]
                y_val_pred_tmp = np.random.randint(2)
                y_val_prob.append(y_val_prob_tmp)
                y_val_pred.append(y_val_pred_tmp)

        y_val_pred = np.array(y_val_pred)
        y_val_prob = np.array(y_val_prob)

        with open(os.path.join(self.save_dir.format(filename), "queryindx2likelihood_ratio-{}.pkl".format(unique_record_idx)),
                  "wb") as fp:
            pickle.dump(queryindx2likelihood_ratio, fp)

        train_log = ["Idx {} / Record: {}".format(unique_record_idx, unique_record.to_dict())]
        train_log.append(
            "Prediction accuracy of validation shadow datasets: {:.4f}".format(accuracy_score(y_val, y_val_pred)))
        train_log.append("Prediction Area Under ROC curve of validation shadow datasets: {:.4f}".format(
            roc_auc_score(y_val, y_val_prob[:, 1])))

        step_two_attack_time = time.time() - step_two_attack_start_time

        with open(os.path.join(self.save_dir.format(filename), "logs-{}.txt".format(unique_record_idx)), "a") as fp:
            fp.write("Step-two attack time (seconds): {}\n".format(step_two_attack_time))
            fp.write("\n".join(train_log))
        return

    def attack(self):
        for filename in self.model_inputs:
            unique_records_not_determined = pd.read_csv(os.path.join(self.save_dir.format(filename), "unique_records_not_determined.csv"), index_col = 0)
            with open(os.path.join(self.save_dir.format(filename), "query2indices.pkl"), "rb") as fp:
                query2indices = pickle.load(fp)

            if self.args.multi_cpus is not None and self.args.multi_cpus > 1:
                with Pool(maxtasksperchild = 100000, processes = self.args.multi_cpus) as pool:
                    # Parallelize two methods (shadow modeling and solver sampling) in step two differently
                    # For shadow modeling, use multiprocessing.Pool on different target record
                    pbar = tqdm(pool.imap_unordered(self._attack, zip(
                        [filename] * len(unique_records_not_determined),
                        list(unique_records_not_determined.iterrows()),
                        [query2indices] * len(unique_records_not_determined))))
                    for row_idx, _ in enumerate(pbar):
                        pbar.set_description("Step two threshold AIA on non-determined unique record {}/{} in dataset {}".format(row_idx + 1, len(unique_records_not_determined), filename))
            else:
                pbar = tqdm(enumerate(unique_records_not_determined.iterrows()))
                for row_idx, indexed_row in pbar:
                    pbar.set_description("Step two threshold AIA on non-determined unique record {}/{} in dataset {}".format(row_idx + 1, len(unique_records_not_determined), filename))
                    self._attack((filename, indexed_row, query2indices))

    def _evaluate_shadow_qbses(self, _input_args):
        """
        Evaluate the AIA results of each target record
        Params:
            filename: The filename of the target dataset
            indexed_row: The target record with record index
        """
        eval_start_time = time.time()
        filename, indexed_row = _input_args
        released_aggregates, _, target_dataset, domain_config, sens_attribute = self.model_inputs[filename]
        unique_record_idx, unique_record = indexed_row

        eval_log = []
        # If target record is not determined, then returns aia_shadow_modeling in number
        if os.path.exists(
                os.path.join(self.save_dir.format(filename), "feasible-solutions-{}_.pkl".format(unique_record_idx))):
            # Add unique-record query
            quasi_ids = [col for col in target_dataset.columns if col != sens_attribute]
            assert self.args.mode == "aia"
            unique_record_query = [tuple((col, (unique_record[col],)) for col in quasi_ids)]
            input_released_queries = list(released_aggregates.keys()) + unique_record_query

            # For multiple models trained by different train-val split, take the majority vote
            # aia_shadow_qbses_votes, aia_shadow_qbses_prob = [], []
            X_test = np.zeros((1, len(input_released_queries)))
            for query_idx, query in enumerate(released_aggregates):
                X_test[0, query_idx] = released_aggregates[query]
            # The answer to unique-record-query in target dataset should also be 1.
            X_test[0, -1] = 1
            y_test = unique_record[sens_attribute]

            useful_queries = []
            useful_query_indices = []
            for query_idx, query in enumerate(input_released_queries):
                useful_query = _find_useful_query((query, unique_record.to_dict(), sens_attribute))
                if (useful_query is not None):
                    useful_queries.append(useful_query)
                    useful_query_indices.append(query_idx)

            with open(os.path.join(self.save_dir.format(filename), "queryindx2likelihood_ratio-{}.pkl".format(unique_record_idx)),
                      "rb") as fp:
                queryindx2likelihood_ratio = pickle.load(fp)

            votes = defaultdict(int)
            if len(useful_queries) > 0:
                for query_idx, query in zip(useful_query_indices, useful_queries):
                    sample = X_test[0, query_idx]
                    likelihood1 = self.gaussian_pdf(sample, queryindx2likelihood_ratio[query_idx]['mu_1'], queryindx2likelihood_ratio[query_idx]['sigma_1'])
                    likelihood0 = self.gaussian_pdf(sample, queryindx2likelihood_ratio[query_idx]['mu_0'], queryindx2likelihood_ratio[query_idx]['sigma_0'])
                    if likelihood1 >= likelihood0:
                        votes[1] += 1
                    else:
                        votes[0] += 1
                y_test_proba = [votes[0] / (votes[0] + votes[1]), votes[1] / (votes[0] + votes[1])]
                y_test_pred = np.argmax(y_test_proba)
            
            else:
                # If no useful queries. then random toos a coin to decide the output
                y_test_proba = [0.5, 0.5]
                y_test_pred = np.random.randint(2)

            with open(os.path.join(self.save_dir.format(filename), "test_proba-{}.pkl".format(unique_record_idx)),
                      "wb") as fp:
                pickle.dump((y_test, y_test_proba), fp)

            eval_log.append("AIA prediction probability by thesholding: {}".format(y_test_proba))
            eval_log.append("AIA prediction prediction by thesholding: {}".format(y_test_pred))
            eval_log.append("AIA label: {}".format(y_test))
            aia_shadow_modeling = y_test == y_test_pred
        # Else the record is determined, return aia_shadow_modeling as None
        else:
            y_test = unique_record[sens_attribute]
            if y_test == 1:
                proba = np.array([[0, 1]])
            else:
                proba = np.array([[1, 0]])
            with open(os.path.join(self.save_dir.format(filename),
                                   "test_proba-determined-{}.pkl".format(unique_record_idx)), "wb") as fp:
                pickle.dump((y_test, proba), fp)
            aia_shadow_modeling = None

        # Append the logs with evaluation results
        eval_time = time.time() - eval_start_time
        with open(os.path.join(self.save_dir.format(filename), "logs-{}.txt".format(unique_record_idx)), "a") as fp:
            fp.write("\nEvaluation time (seconds): {}\n".format(eval_time))
            fp.write("\n".join(eval_log))
        return aia_shadow_modeling

    def evaluate(self):
        """
        Evaluate the results for two-step method
        Params: None
        """
        for filename in self.model_inputs:
            unique_records = pd.read_csv(os.path.join(self.save_dir.format(filename), "unique_records.csv"),
                                         index_col=0)
            if self.args.multi_cpus is not None and self.args.multi_cpus > 1:
                with Pool(maxtasksperchild=100000, processes=self.args.multi_cpus) as pool:
                    pbar = tqdm(pool.imap_unordered(self._evaluate_shadow_qbses, zip([filename] * len(unique_records),
                                                                                     list(unique_records.iterrows()))))
                    for row_idx, aia_shadow_modeling in enumerate(pbar):
                        pbar.set_description(
                            "Evaluating {}/{} unique record in dataset {}".format(row_idx + 1, len(unique_records),
                                                                                  filename))
            else:
                pbar = tqdm(enumerate(unique_records.iterrows()))
                for row_idx, indexed_row in pbar:
                    pbar.set_description(
                        "Evaluating {}/{} unique record in dataset {}".format(row_idx + 1, len(unique_records),
                                                                              filename))
                    aia_shadow_modeling = self._evaluate_shadow_qbses((filename, indexed_row))
