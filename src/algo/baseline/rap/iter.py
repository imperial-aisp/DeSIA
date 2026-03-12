"""
Iterative algorithm frameworks class, to generate synthetic dataset
Modified from https://github.com/terranceliu/dp-query-release/blob/main/src/algo/nondp/gen_nondp.py
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from .qm import KWayMarginalQMTorch, KWayMarginalSetQMTorch

def get_errors(true_answers, fake_answers):
    if torch.is_tensor(true_answers):
        errors = (true_answers - fake_answers).abs()
        results = {'error_max': errors.max().item(),
                   'error_mean': errors.mean().item(),
                   'error_mean_squared': torch.linalg.norm(errors, ord=2).item() ** 2 / len(errors),
                   }
    else:
        errors = np.abs(true_answers - fake_answers)
        results = {'error_max': np.max(errors),
                   'error_mean': np.mean(errors),
                   'error_mean_squared': np.linalg.norm(errors, ord=2) ** 2 / len(errors),
                   }
    return results

class IterativeAlgorithm(ABC):
    """
    Arguments:
        qm (QueryManager): query manager for defining queries and calculating answers
        T (int): Number of rounds to run algorithm
        eps0 (float): Privacy budget per round (zCDP)
        alpha (float, optional): Changes the allocation of the per-round privacy budget
            Selection mechanism uses ``alpha * eps0`` and Measurement mechanism uses ``(1-alpha) * eps0``.
            If given, it must be between 0 and 1.
        default_dir (string, optional): Path for saving the class state. If None is passed, a random directory is generated.
        verbose (boolean, optional): Flag for whether to print progress while fitting to true answers
        seed (int, optional): seed for reproducibility
    """
    def __init__(self, G, T, eps0,
                 alpha=0.5, default_dir=None, verbose=False, seed=None):
        assert 0 <= alpha <= 1, "alpha must be between 0 and 1"

        self.G = G
        self.qm = G.qm
        self.queries = G.qm._queries

        self.T = T
        self.eps0 = eps0
        self.alpha = alpha
        self.default_dir = default_dir
        self.verbose = verbose
        self.seed = seed

        self.sampled_max_errors = []
        self.true_max_errors = []
        self.true_mean_errors = []
        self.true_mean_squared_errors = []

        self.past_workload_idxs = [] # only used for sensitivity trick implementations
        self.past_query_idxs = []
        self.past_measurements = []

        # validate QueryManager is correct
        assert isinstance(self.qm, self._valid_qm()), \
            "QueryManager must be chosen from the following classes: {}".format(
                ", ".join([x.__name__ for x in self._valid_qm()]))

        # create directory for saving algo files
        if self.default_dir is not None:
            if not os.path.exists(self.default_dir):
                os.makedirs(self.default_dir)
        if self.verbose:
            print("Saving algorithm files to: {}".format(self.default_dir))

        # set seed for reproducibility
        if self.seed is not None:
            self._set_seed()

    def _set_seed(self):
        np.random.seed(self.seed)

    """
    Save current state
    Input:
        path (string): file path to save to
    """
    def save(self, filename, directory=None):
        if directory is None:
            directory = self.default_dir
        path = os.path.join(directory, filename)
        torch.save(self.G.generator.state_dict(), path)

    """
    Load state
    Input:
        path (string): file path to load from
    """
    def load(self, filename, directory=None):
        if directory is None:
            directory = self.default_dir
        path = os.path.join(directory, filename)
        state_dict = torch.load(path)
        self.G.generator.load_state_dict(state_dict)

    def record_errors(self, true_answers, fake_answers):
        errors_dict = get_errors(true_answers, fake_answers)
        self.true_max_errors.append(errors_dict['error_max'])
        self.true_mean_errors.append(errors_dict['error_mean'])
        self.true_mean_squared_errors.append(errors_dict['error_mean_squared'])

    """
    Returns tuple of valid QueryManager classes
    """
    @abstractmethod
    def _valid_qm(self):
        pass

    """
    Algorithm fits to a list of answers.
    Input:
        true_answers (np.array): true answers the algorithm is fitting to
    """
    @abstractmethod
    def fit(self, true_answers):
        pass

    """
    Uses differentially private mechanism to sample query
    Input:
        scores (np.array): score function applied to each query
    """
    @abstractmethod
    def _sample(self, scores):
        pass

    """
    Uses differentially private mechanism to get a noisy measure of query answers
    Input:
        answers (np.array): true answers that the noisy measurements are approximating 
    """
    @abstractmethod
    def _measure(self, answers):
        pass

class IterativeAlgorithmTorch(IterativeAlgorithm):
    def __init__(self, G, T, eps0,
                 alpha=0.5, default_dir=None, verbose=False, seed=None):
        super().__init__(G, T, eps0,
                         alpha=alpha, default_dir=default_dir, verbose=verbose, seed=seed)
        self.device = self.G.device

        # convert these lists into tensors for Pytorch code
        self.past_workload_idxs = torch.tensor([], device=self.device).long() # only used for sensitivity trick implementations
        self.past_query_idxs = torch.tensor([], device=self.device).long()
        self.past_measurements = torch.tensor([], device=self.device)

    def _set_seed(self):
        super()._set_seed()
        torch.manual_seed(self.seed)

    def get_answers(self):
        return self.G.get_qm_answers()

    def _get_sampled_query_errors(self, idxs=None, density=False):
        q_t_idxs = self.past_query_idxs.clone().to(self.device)
        real_answers = self.past_measurements.to(self.device)
        if idxs is not None:
            q_t_idxs = q_t_idxs[idxs]
            real_answers = real_answers[idxs]

        syn = self.G.generate()
        syn_answers = self.G.get_answers(syn, density=density, idxs=q_t_idxs)
        errors = real_answers - syn_answers
        # Implement MAPE loss to replace MAE loss
        # errors = torch.div(real_answers - syn_answers, real_answers)
        return errors

class IterativeAlgoNonDP(IterativeAlgorithmTorch):
    def __init__(self, G, T,
                 loss_p=2, lr=1e-4, eta_min=1e-5, max_idxs=10000, max_iters=1,
                 sample_by_error=False, log_freq=1, default_dir=None, verbose=False,
                 seed=None, density=False):
        super().__init__(G, T, eps0=0, alpha=0,
                         default_dir=default_dir, verbose=verbose, seed=seed)
        self.loss_p = loss_p
        self.lr = lr
        self.eta_min = eta_min
        self.max_idxs = max_idxs
        self.max_iters = max_iters
        self.density = density
        self.loss_records = [float('inf')]
        self.iter_no_improve = 0

        # Frequency to save logs
        self.log_freq = log_freq
        assert log_freq >= 0, 'record_all_errors must be >= 0'

        self.sample_by_error = sample_by_error
        if sample_by_error and log_freq != 1:
            self.log_freq = 1
            print("sample_by_error=True -> defaulting log_freq to 1")

        self.optimizerG = torch.optim.Adam(self.G.generator.parameters(), lr=self.lr)
        self.schedulerG = None
        if self.eta_min is not None:
            self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizerG, self.T, eta_min=self.eta_min)
        # num_trainable_params = sum(p.numel() for p in self.G.generator.parameters() if p.requires_grad)
        # print('The number of trainable parameters: {}'.format(num_trainable_params))

    def _valid_qm(self):
        return (KWayMarginalQMTorch, KWayMarginalSetQMTorch)

    def _sample(self, scores):
        pass

    def _measure(self, answers):
        pass

    def _get_loss(self, idxs, density=False):
        errors = self._get_sampled_query_errors(idxs=idxs, density=density)
        loss = torch.norm(errors, p=self.loss_p) / len(errors)
        return loss

    def fit(self, true_answers):
        if self.verbose:
            print('Fitting to query answers...')
        self.past_query_idxs = torch.arange(self.qm.num_queries)
        self.past_measurements = true_answers.clone()

        syn_answers = self.G.get_qm_answers(density = self.density)
        # If sample_by_error is True, then optimize on queries with the largest errors
        errors = (true_answers - syn_answers).abs()
        p = errors / errors.sum()

        pbar = tqdm(range(self.T)) if self.verbose else range(self.T)
        for iteration in pbar:
            if (self.verbose) and (self.log_freq > 0):
                pbar.set_description('Max Error: {:.6}'.format(errors.max()))
                print('Max Error: {:.6}'.format(errors.max()))

            if self.sample_by_error:
                for _ in range(self.max_iters):
                    self.optimizerG.zero_grad()
                    idxs = torch.multinomial(p, num_samples=self.max_idxs, replacement=True)
                    loss = self._get_loss(idxs, density=self.density)
                    loss.backward()
                    self.optimizerG.step()
            else:
                idxs_all = torch.randperm(len(errors))
                for idxs in torch.split(idxs_all, self.max_idxs):
                    self.optimizerG.zero_grad()
                    loss = self._get_loss(idxs, density=self.density)
                    loss.backward()
                    self.optimizerG.step()

            if self.schedulerG is not None:
                self.schedulerG.step()

            if (self.log_freq > 0) and (iteration % self.log_freq == 0):
                syn_answers = self.G.get_qm_answers(density = self.density)
                errors = (true_answers - syn_answers).abs()
                p = errors / errors.sum()
                self.record_errors(true_answers, syn_answers)