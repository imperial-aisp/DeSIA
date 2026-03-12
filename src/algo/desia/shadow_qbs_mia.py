import gc
import re
from .base import *
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from ...utils.sample_mia import get_train_and_eval_shadow_labels, get_query2answers

class StepTwoShadowQbsAttack(BaseAttack):
    """
    The class implementing step two shadow-dataset based attribute inference attack
    """
    def __init__(self, args: Parser, model_inputs, repo_directory: str):
        super().__init__(args, model_inputs, repo_directory)

    def _save_model(self, filename, unique_record_idx, model):
        if not os.path.exists(self.save_dir.format(filename)):
            os.makedirs(self.save_dir.format(filename))
        with open(os.path.join(self.save_dir.format(filename), "model-{}.pkl".format(unique_record_idx)), "wb") as fp:
            pickle.dump(model, fp)
        print("Model saved for the record index {} inside dataset {}".format(unique_record_idx, filename))

    def get_X_train_eval(self, queries, query2answers):
        X = np.zeros((self.args.num_training_qbses, len(queries)))
        for i, query in enumerate(queries):
            X[:, i] = np.array(query2answers[tuple(sorted(query, key = lambda x: x[0]))])
        train_size = int(self.args.num_training_qbses * (1 - 0.3333))
        return X[:train_size, :], X[train_size:, :]

    def get_y_train_eval(self, y_train_eval):
        train_size = int(self.args.num_training_qbses * (1 - 0.3333))
        return y_train_eval[:train_size], y_train_eval[train_size:]
    
    def _attack(self, _input_args):
        step_two_attack_start_time = time.time()
        filename, indexed_row, query2indices = _input_args
        released_aggregates, aux_dataset, target_dataset, domain_config, sens_attribute = self.model_inputs[filename]
        # Add unique-record query as one constraint
        unique_record_idx, unique_record = indexed_row

        # Set up the device specifically used for shadow modeling
        proc_device = self.devices[unique_record_idx % len(self.devices)]
        # M: contains 1 in row i, column j, if the shadow dataset i contains the user j
        M = np.zeros((self.args.num_training_qbses, len(aux_dataset) + len(target_dataset)))
        # S: contains 1 in row i, column j, if the shadow dataset i contains the user j AND user j has sens=key in dataset i
        S = {(sens_attribute, key) : np.zeros((self.args.num_training_qbses, len(aux_dataset) + len(target_dataset))) for key in range(domain_config[sens_attribute])}
        _, y_train_val = get_train_and_eval_shadow_labels(self.args, domain_config, pd.DataFrame([unique_record]), aux_dataset, target_dataset, M, S)
        query2answers = get_query2answers(pd.concat([aux_dataset, target_dataset]).sort_index(), query2indices, M, S, proc_device)
        del M, S
        gc.collect()
        X_train, X_val = self.get_X_train_eval(list(released_aggregates.keys()), query2answers)
        y_train, y_val = self.get_y_train_eval(y_train_val)
        y_train, y_val = np.squeeze(np.array(y_train)), np.squeeze(np.array(y_val))

        # Different models for stochastic module
        if self.args.aia_model == 'logit':
            parameters = {
                'LR__C': np.logspace(-5, -2, 10).tolist() + np.logspace(-2, 2, 40).tolist() + np.logspace(2, 5, 10).tolist()
            }
            pipeline = Pipeline([('scaler', StandardScaler()),
                ('LR', LogisticRegression(random_state=0, max_iter=1000))])
            clf = GridSearchCV(pipeline, param_grid=parameters)
        elif self.args.aia_model == 'mlp':
            clf = Pipeline([('scaler', StandardScaler()),
                ('MLP', MLPClassifier(random_state=0, hidden_layer_sizes=(50, 20),
                    early_stopping=True))])
        elif self.args.aia_model == 'rf':
            clf = RandomForestClassifier(random_state=0)
        elif self.args.aia_model == 'svm':
            clf = Pipeline([('scaler', StandardScaler()),
                ('SVM', SVC(random_state=0, probability=True))])
        else:
            raise NotImplementedError

        clf.fit(X_train, y_train)
        y_train_pred, y_val_pred = clf.predict(X_train), clf.predict(X_val)
        y_train_prob, y_val_prob = clf.predict_proba(X_train), clf.predict_proba(X_val)

        # Print out the debugging logs
        train_log = ["Idx {} / Record: {}".format(unique_record_idx, unique_record.to_dict())]
        y_train_pred, y_val_pred = clf.predict(X_train), clf.predict(X_val)
        y_train_prob, y_val_prob = clf.predict_proba(X_train), clf.predict_proba(X_val)
        
        # Print out the debugging logs
        train_log = ["Idx {} / Record: {}".format(unique_record_idx, unique_record.to_dict())]
        train_log.append("Prediction accuracy of training shadow datasets: {:.4f}".format(accuracy_score(y_train, y_train_pred)))
        train_log.append("Prediction accuracy of validation shadow datasets: {:.4f}".format(accuracy_score(y_val, y_val_pred)))
        train_log.append("Prediction Area Under ROC curve of training shadow datasets: {:.4f}".format(roc_auc_score(y_train, y_train_prob[:, 1])))
        train_log.append("Prediction Area Under ROC curve of validation shadow datasets: {:.4f}".format(roc_auc_score(y_val, y_val_prob[:, 1])))
        # Save the model
        self._save_model(filename, unique_record_idx, clf)
        
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
                        pbar.set_description("Step two shadow modeling AIA on non-determined unique record {}/{} in dataset {}".format(row_idx + 1, len(unique_records_not_determined), filename))
            else:
                pbar = tqdm(enumerate(unique_records_not_determined.iterrows()))
                for row_idx, indexed_row in pbar:
                    pbar.set_description("Step two shadow modeling AIA on non-determined unique record {}/{} in dataset {}".format(row_idx + 1, len(unique_records_not_determined), filename))
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
        released_aggregates, _, _, _, _ = self.model_inputs[filename]
        unique_record_idx, _ = indexed_row

        # Compute released aggregates
        tmp_target_dataset = pd.read_csv(os.path.join(self.save_dir.format(filename), "target-{}.csv".format(unique_record_idx)), index_col = 0)
        tmp_released_aggregates = {}
        for query in released_aggregates:
            records_touched_by_query = tmp_target_dataset.copy()
            for col, values in query:
                records_touched_by_query = records_touched_by_query[records_touched_by_query[col].isin(values)]
            tmp_released_aggregates[query] = len(records_touched_by_query)

        eval_log = []
        # If target record is not determined, then returns aia_shadow_modeling in number
        with open(os.path.join(self.save_dir.format(filename), "logs-{}.txt".format(unique_record_idx)), "r") as fp:
            lines = fp.readlines()
            if "True" in lines[0]:
                target_record_in_target_dataset = 1
            elif "False" in lines[0]:
                target_record_in_target_dataset = 0
        y_test = target_record_in_target_dataset

        if os.path.exists(os.path.join(self.save_dir.format(filename), "feasible-solutions-{}_.pkl".format(unique_record_idx))):
            # For multiple models trained by different train-val split, take the majority vote
            with open(os.path.join(self.save_dir.format(filename), "model-{}.pkl".format(unique_record_idx)), "rb") as fp:
                clf = pickle.load(fp)
            X_test = np.zeros((1, len(released_aggregates)))
            for query_idx, query in enumerate(released_aggregates):
                X_test[0, query_idx] = tmp_released_aggregates[query]
            with open(os.path.join(self.save_dir.format(filename), "test_proba-{}.pkl".format(unique_record_idx)), "wb") as fp:
                pickle.dump((y_test, clf.predict_proba(X_test)), fp)
            eval_log.append("Target record is a member from prediction: {}".format(True if int(clf.predict(X_test)) == 1 else False))
        # Else the record is determined
        else:
            if y_test == 1:
                proba = np.array([[0, 1]])
            else:
                proba = np.array([[1, 0]])
            with open(os.path.join(self.save_dir.format(filename), "test_proba-determined-{}.pkl".format(unique_record_idx)), "wb") as fp:
                pickle.dump((y_test, proba), fp)
            eval_log.append("Target record is a member from prediction: {}".format(True if y_test == 1 else False))

        eval_time = time.time() - eval_start_time
        with open(os.path.join(self.save_dir.format(filename), "logs-{}.txt".format(unique_record_idx)), "a") as fp:
            fp.write("\nEvaluation time (seconds): {}\n".format(eval_time))
            fp.write("\n".join(eval_log))
        return None
    
    def evaluate(self):
        """
        Evaluate the results for two-step method
        Params: None
        """
        for filename in self.model_inputs:
            unique_records = pd.read_csv(os.path.join(self.save_dir.format(filename), "unique_records.csv"), index_col = 0)
            if self.args.multi_cpus is not None and self.args.multi_cpus > 1:
                with Pool(maxtasksperchild = 100000, processes = self.args.multi_cpus) as pool:
                    pbar = tqdm(pool.imap_unordered(self._evaluate_shadow_qbses, zip([filename] * len(unique_records), list(unique_records.iterrows()))))
                    for row_idx, _ in enumerate(pbar):
                        pbar.set_description("Evaluating {}/{} unique record in dataset {}".format(row_idx + 1, len(unique_records), filename))
            else:
                pbar = tqdm(unique_records.iterrows())
                for row_idx, indexed_row in pbar:
                    pbar.set_description("Evaluating dataset {}".format(filename))
                    self._evaluate_shadow_qbses((filename, indexed_row))
