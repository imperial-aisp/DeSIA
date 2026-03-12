import os
import json
import numpy as np
import pandas as pd

class DatasetLoader(object):
    """
    Generic dataset loader function.

    It provides functions to sample an auxiliary and a target dataset from a
    given dataset as well as a subset of attributes.
    The user can specify the size of the target and auxiliary splits.
    """

    def __init__(self, dataset_dir, folder, filename):
        self.dataset_dir = dataset_dir
        self.folder = folder
        self.filename = filename
        self.all_records = self._get_all_records()
        self.attributes = np.arange(len(self.all_records.columns))
        self._print_success()

    def _check_num_attributes_valid(self, num_attributes):
        if num_attributes is None:
            return len(self.all_records.columns)
        elif isinstance(num_attributes, int):
            if 0 < num_attributes <= len(self.all_records.columns):
                return num_attributes
            else:
                raise ValueError('Invalid `num_attributes` value passed.')
        raise RuntimeError('Invalid `num_attributes` parameter passed.')

    def _check_size_valid(self, size, param_name):
        if size is None:
            return size
        if isinstance(size, int):
            if 0 < size < len(self.all_records):
                return size
            raise ValueError(f'Invalid `{param_name}` value passed.')
        raise RuntimeError(f'Invalid `{param_name}` parameter passed.')

    def _get_all_records(self):
        """
        Returns a numpy array where each row is a record.

        Each value is a discrete int,  where the integers denote different
        categories.
        """
        dataset = pd.read_csv(os.path.join(self.dataset_dir, self.folder, f'{self.filename}.csv'))
        dataset.columns = [column_name.lower().replace('-', '') for column_name in dataset.columns]
        return dataset
    
    def get_config(self):
        """
        Obtain the dataset metadata
        """
        if ("ppmf" in self.filename) or ("acs" in self.filename):
            with open(os.path.join(self.dataset_dir, "{}/domain".format(self.folder), '{}-domain.json'.format(self.filename)), "r") as fp:
                self.config = json.load(fp)
            self.config = {col.lower(): self.config[col] for col in self.config}
            self.attributes = np.arange(len(self.all_records.columns))
            if "ppmf" in self.filename:
                self.sens_attribute = "cenhisp"
            elif "acs" in self.filename:
                self.sens_attribute = "label"
        else:
            raise NotImplementedError
        return self.config

    def split_dataset(self, test_size, aux_size, verbose=False):
        """
        Split into a target and auxiliary split.

        test_size: size of the split from which target datasets will be drawn.
        aux_size: size of the split from which shadow datasets will be drawn.
        """
        test_size = self._check_size_valid(test_size, 'test_size')
        aux_size = self._check_size_valid(aux_size, 'aux_size')
        if test_size is None and aux_size is None:
            raise ValueError('Both `test_size` and `aux_size` cannot be None.')
        elif test_size is None:
            test_size = len(self.all_records) - aux_size
        elif aux_size is None:
            aux_size = len(self.all_records) - test_size
        idxs = np.random.choice(len(self.all_records), test_size + aux_size,
                                replace=False)
        self.test_idxs = idxs[:test_size]
        self.auxiliary_idxs = idxs[test_size:]
        if verbose:
            print('Test indexs: ', self.test_idxs)
            print('Auxiliary indexs: ', self.auxiliary_idxs)
        self.test_split = self.all_records.loc[self.test_idxs]
        self.auxiliary_split = self.all_records.loc[self.auxiliary_idxs]
        return self.auxiliary_split, self.test_split

    def sample_attributes(self, num_attributes):
        """
        Sample a subset of attributes.
        """
        num_attributes = self._check_num_attributes_valid(num_attributes)
        attributes = np.arange(len(self.all_records.columns))
        self.attributes, _ = self._sampling_helper(attributes, num_attributes)
        print(f'Sampled {num_attributes} attributes : ', self.attributes)
        return self.attributes
    
    def apply_resample(self):
        self.auxiliary_split.drop(labels = [self.sens_attribute], axis = 1)
        self.auxiliary_split[self.sens_attribute] = np.random.binomial(n = 1, p = 0.5, size = len(self.auxiliary_split))
        self.test_split.drop(labels = [self.sens_attribute], axis = 1)
        self.test_split[self.sens_attribute] = np.random.binomial(n = 1, p = 0.5, size = len(self.test_split))
        return self.auxiliary_split, self.test_split
    
    def replace(self, new_aux_idxs, new_test_idxs):
        self.auxiliary_idxs = new_aux_idxs
        self.auxiliary_split = self.auxiliary_split.iloc[list(range(len(new_aux_idxs))),:]
        self.auxiliary_split.index = new_aux_idxs
        self.test_idxs = new_test_idxs
        self.test_split.index = new_test_idxs
        self.all_records = pd.concat([self.auxiliary_split, self.test_split]).sort_index()
        return self.auxiliary_split, self.test_split

    @staticmethod
    def _sampling_helper(dataset, num_samples):
        """
        Returns a sample of size `num_samples` from `dataset`, where the
        records are sampled without replacement.
        """
        if num_samples is None:
            num_samples = len(dataset)
        if type(num_samples) == int:
            if num_samples > len(dataset):
                raise ValueError('Invalid value for `num_samples`, it should be smaller than the dataset size.')
            # elif num_samples == len(dataset):
            #    return dataset, np.arange(len(dataset))
            else:
                idxs = np.random.choice(len(dataset), num_samples,
                                        replace=False)
                return dataset[idxs], idxs
        else:
            raise TypeError('Invalid type for `num_samples`, should be None or int.')

    def _print_success(self):
        print(f'Successfully loaded {self.folder} of {len(self.all_records)} ' +
              f'records and {len(self.attributes)} attributes.')