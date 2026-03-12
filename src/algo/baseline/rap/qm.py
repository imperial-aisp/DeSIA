"""
The instance to manage aggregate statistics in RAP method
Modified from https://github.com/terranceliu/dp-query-release/blob/main/src/qm/qm.py
"""
import torch
import itertools
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
from abc import ABC, abstractmethod
from .domain import Dataset, Domain

def get_num_queries(domain, workloads, return_workload_lens=False):
    col_map = {}
    for i, col in enumerate(domain.attrs):
        col_map[col] = i
    feat_pos = []
    cur = 0
    for f, sz in enumerate(domain.shape):
        feat_pos.append(list(range(cur, cur + sz)))
        cur += sz

    num_queries = 0
    workload_lens = []
    for feat in workloads:
        positions = []
        for col in feat:
            i = col_map[col]
            positions.append(feat_pos[i])
        x = list(itertools.product(*positions))
        num_queries += len(x)
        workload_lens.append(len(x))

    if return_workload_lens:
        return num_queries, workload_lens
    return num_queries

def get_min_dtype(arr):
    max_val_abs = np.abs(arr).max()
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        if max_val_abs < np.iinfo(dtype).max:
            return dtype

def get_data_onehot(data):
    df_data = data.df.copy()
    dim = np.sum(data.domain.shape)

    i = 0
    for attr in data.domain.attrs:
        df_data.loc[df_data[attr] >= 0, attr] += i # ignore -1
        i += data.domain[attr]
    data_values = df_data.values

    data_onehot = np.zeros((len(data_values), dim))
    arange = np.arange(len(data_values))
    arange = np.tile(arange, (data_values.shape[1], 1)).T

    assert (data_values[data_values < 0] == -1).all(), "negative values, possible overflow error due to dtype"
    x = np.tile(data_values[:, 0] + 1, (data_values.shape[-1], 1)).T
    x[data_values != -1] = 0
    data_values += x

    data_onehot[arange, data_values] = 1

    return data_onehot.astype(bool)

def hamming_distance(record1, record2):
    return sum(x != y for x, y in zip(record1, record2))

def get_data(df, domain = None):
    # If there is no predefined Domain instance, compute a new Domain instance
    if domain is None:
        config= {}
        for key in df.columns:
            unique_values = df[key].unique()
            num_unique_values = len(unique_values)
            config[key] = num_unique_values
        domain = Domain(config.keys(), config.values())
    dtype = get_min_dtype(sum(domain.config.values()))
    df = df.astype(dtype)
    data = Dataset(df, domain)
    return data

def get_my_workloads(released_queries: list):
    workloads = [tuple([col for col, _ in query]) for query in released_queries]
    return list(set(workloads))

class QueryManager(ABC):
    def __init__(self, data, workloads, domain, sensitivity=None, verbose=False):
        self.N = len(data)
        self.domain = domain
        self.workloads = sorted([tuple(sorted(x)) for x in workloads])
        self.sensitivity = sensitivity
        self.verbose = verbose

        self.dim = np.sum(self.domain.shape)
        self.num_workloads = len(self.workloads)
        self.num_queries, _ = get_num_queries(self.domain, self.workloads, return_workload_lens=True)

        """
        To provide documentation, we initialize all class variables to None and describe them in comments.
        These variables will be initialized via self._setup()
        """
        # dictionaries mapping attributes to various values
        self.col_map = None
        self.feat_pos_map = None
        self.col_pos_map = None
        self.pos_col_map = None
        # query related
        self._queries = None
        self._setup()

    def _setup(self):
        if self.verbose:
            print("Setting up query manager...")
        self._setup_maps()
        self._setup_queries()

    def _setup_maps(self):
        self.col_map = {}
        for i, col in enumerate(self.domain.attrs):
            self.col_map[col] = i

        self.feat_pos_map = []
        cur = 0
        for sz in self.domain.shape:
            self.feat_pos_map.append(list(range(cur, cur + sz)))
            cur += sz

        self.col_pos_map = {}
        for col, i in self.col_map.items():
            self.col_pos_map[col] = self.feat_pos_map[i]

        self.pos_col_map = {}
        for i, col in enumerate(self.col_map.keys()):
            for pos in self.feat_pos_map[i]:
                attr_val = pos - self.feat_pos_map[i][0]
                self.pos_col_map[pos] = (col, attr_val)

    @abstractmethod
    def _setup_queries(self):
        pass

    @abstractmethod
    def get_answers(self, *args, **kwargs):
        pass

"""
Base K-way marginal query manager class
"""
class BaseKWayMarginalQM(QueryManager):
    def __init__(self, data, workloads, released_queries, domain, sensitivity=None, verbose=False):
        self.released_queries = released_queries
        super().__init__(data, workloads, domain, sensitivity=sensitivity, verbose=verbose)
        if sensitivity is None:
            self.sensitivity = 1 / self.N

    def _setup_queries(self):
        max_marginal = np.array([len(x) for x in self.workloads]).max()
        _queries = -1 * np.ones((len(self.released_queries), max_marginal), dtype=get_min_dtype([self.dim]))

        domain_values = [0] + list(self.domain.config.values())
        domain_values = np.cumsum(domain_values)[:-1]

        iterable = tqdm(self.released_queries) if self.verbose else self.released_queries
        for idx, col_value_pair in enumerate(iterable):
            x = []
            for col, value in col_value_pair:
                if type(value) is int:
                    x.extend([self.feat_pos_map[self.col_map[col]][0] + value])
                elif type(value) is tuple:
                    x.extend([self.feat_pos_map[self.col_map[col]][0] + value[0]])
            _queries[idx, :len(x)] = x
        self._queries = np.array(_queries)

"""
K-way marginal query manager
"""
class KWayMarginalQM(BaseKWayMarginalQM):
    def get_answers(self, data, weights=None, by_workload=False, density=True):
        if self.verbose:
            print("Calculating query answers...")
        ans_vec = []
        iterable = self.workloads
        if self.verbose:
            iterable = tqdm(iterable)
        for proj in iterable:
            x = data.project(proj).datavector(weights=weights, density=density)
            ans_vec.append(x)

        if not by_workload:
            return np.concatenate(ans_vec)
        return ans_vec

    def get_query_onehot(self, q_ids):
        if not isinstance(q_ids, Iterable):
            q_ids = [q_ids]
        W = []
        for q_id in q_ids:
            w = np.zeros(self.dim)
            for p in self._queries[q_id]:
                if p < 0:
                    break
                w[p] = 1
            W.append(w)
        W = np.array(W)
        if len(W) == 1:
            W = W.reshape(1, -1)
        return W

class KWayMarginalQMTorch(KWayMarginalQM):
    def __init__(self, data, workloads, released_queries, domain, device=None, sensitivity=None, verbose=False):
        self.device = torch.device("cpu") if device is None else device
        super().__init__(data, workloads, released_queries, domain, sensitivity=sensitivity, verbose=verbose)

    def _setup_queries(self):
        super()._setup_queries()
        self._queries = torch.tensor(self._queries).long().to(self.device)
        self.num_queries = len(self._queries)

    def get_answers_helper(self, data_onehot, weights, query_idxs=None, batch_size=1000, verbose=False):
        _queries = self._queries
        if query_idxs is not None:
            _queries = _queries[query_idxs]
        queries_iterable = torch.split(_queries, batch_size)
        if verbose:
            queries_iterable = tqdm(queries_iterable)

        answers = []
        for queries_batch in queries_iterable:
            answers_batch = data_onehot[:, queries_batch]
            answers_batch[:, queries_batch == -1] = 1
            answers_batch = answers_batch.prod(axis=-1)
            answers_batch = answers_batch * weights
            answers_batch = answers_batch.sum(0)
            answers.append(answers_batch)
        answers = torch.cat(answers)
        return answers

    # Currently (torch=1.11.0), torch.histogramdd doesn't support CUDA operations (rewrite below if support is added)
    def get_answers(self, data, weights=None, density=True, batch_size=1000):
        if self.verbose:
            print("Calculating query answers...")
        if weights is None:
            weights = np.ones(len(data))
        weights = torch.tensor(weights, dtype=torch.float).unsqueeze(-1).to(self.device)
        data_onehot = torch.tensor(get_data_onehot(data), dtype=torch.float, device=self.device)
        answers = self.get_answers_helper(data_onehot, weights, batch_size=batch_size, verbose=self.verbose)
        if density:
            answers = answers/weights.sum()
        return answers

class KWayMarginalSetQMTorch(KWayMarginalQMTorch):
    def __init__(self, data, workloads, released_queries, domain, sensitivity=None, device=None, verbose=False):
        super().__init__(data, workloads, released_queries, domain, sensitivity=sensitivity, device=device, verbose=verbose)

    def _setup_queries(self):
        super()._setup_queries()
        
        _released_queries = [{col_vals_pair[0]: np.array(col_vals_pair[1]) for col_vals_pair in query} for query in self.released_queries]
        self.num_queries = len(_released_queries)

        queries = {}
        for attr in self.domain.attrs:
            x = [query[attr] if attr in query.keys() else np.array([]) for query in _released_queries]
            x = list(zip(*itertools.zip_longest(*x, fillvalue=-1)))
            x = np.array(list(x))
            queries[attr] = x
        iter_queries = queries.items()
        if self.verbose:
            iter_queries = tqdm(iter_queries)

        shape = (self.num_queries, len(queries), self.dim + 1)
        self._queries = torch.zeros(shape, dtype=bool, device=self.device)
        for i, (attr, vals) in enumerate(iter_queries):
            idxs = torch.tensor(self.col_pos_map[attr] + [self.dim], device=self.device)
            idxs = idxs[vals]
            self._queries[:, i] = self._queries[:, i].scatter_(1, idxs, True)
        self._queries = self._queries[:, :, :-1]
        self.num_queries = len(self._queries)

    def _get_workloads(self, queries):
        workloads = [list(q.keys()) for q in queries]
        workloads = np.array(workloads, dtype=object)
        workloads = np.unique(workloads)
        workloads = list(sorted(workloads, key=len, reverse=False))
        workloads = [tuple(workload) for workload in workloads]
        return workloads
    
    def get_answers_helper(self, data_onehot, weights, query_idxs=None, batch_size=1000, verbose=False):
        queries = self._queries
        if query_idxs is not None:
            queries = queries[query_idxs]
        queries_iterable = torch.split(queries, batch_size)
        if verbose:
            queries_iterable = tqdm(queries_iterable)

        answers = []
        for queries_batch in queries_iterable:
            queries_batch = queries_batch.permute((2, 1, 0))
            answers_batch = torch.zeros((queries_batch.shape[1], len(data_onehot), queries_batch.shape[2]), device=self.device)
            for i in range(queries_batch.shape[1]):
                mask = queries_batch[:, i]
                x = data_onehot.mm(mask.float())
                x[:, ~mask.any(0)] = 1
                answers_batch[i] = x
            answers_batch = answers_batch.prod(0)
            answers_batch = answers_batch * weights
            answers_batch = answers_batch.sum(0)
            answers.append(answers_batch)
        answers = torch.cat(answers)

        return answers