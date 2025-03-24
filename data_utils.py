# These are the main classes and methods.
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from scipy.stats import truncnorm
from collections import deque
import operator
import math
import random
import os

import copy

from conf import *
from sklearn.metrics import confusion_matrix
device = torch.device('cuda:{}'.format(device_num) if torch.cuda.is_available() else 'cpu')
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SymptomVocab:

    def __init__(self, samples: list = None, add_special_sxs: bool = False,
                 min_sx_freq: int = None, max_voc_size: int = None, prior_feat: list = None):

        # sx is short for symptom
        self.sx2idx = {}  # map from symptom to symptom id
        self.idx2sx = {}  # map from symptom id to symptom
        self.sx2count = {}  # map from symptom to symptom count
        self.num_sxs = 0  # number of symptoms
        self.prior_sx_attr = {} # key symptoms with True
        self.prior_sx_attr_2 = {} # key symptoms with False
        self.no_key_sx = []


        # symptom attrs
        self.SX_ATTR_PAD_IDX = 0  # symptom attribute id for PAD
        self.SX_ATTR_POS_IDX = 1  # symptom attribute id for YES
        self.SX_ATTR_NEG_IDX = 2  # symptom attribute id for NO
        self.SX_ATTR_NS_IDX = 3  # symptom attribute id for NOT SURE
        self.SX_ATTR_NM_IDX = 4  # symptom attribute id for NOT MENTIONED

        # symptom askers
        self.SX_EXE_PAD_IDX = 0  # PAD
        self.SX_EXE_AI_IDX = 1  # AI
        self.SX_EXE_DOC_IDX = 2  # Human

        self.SX_ATTR_MAP = {  # map from symptom attribute to symptom attribute id
            '0': self.SX_ATTR_NEG_IDX,
            '1': self.SX_ATTR_POS_IDX,
            '2': self.SX_ATTR_NS_IDX,
        }

        self.SX_ATTR_MAP_INV = {
            self.SX_ATTR_NEG_IDX: '0',
            self.SX_ATTR_POS_IDX: '1',
            self.SX_ATTR_NS_IDX: '2',
        }

        # special symptoms
        self.num_special = 0  # number of special symptoms
        self.special_sxs = []

        # vocabulary
        self.min_sx_freq = min_sx_freq  # minimal symptom frequency
        self.max_voc_size = max_voc_size  # maximal symptom size

        # add special symptoms
        if add_special_sxs:  # special symptoms
            self.SX_PAD = '[PAD]'
            self.SX_START = '[START]'
            self.SX_END = '[END]'
            self.SX_UNK = '[UNKNOWN]'
            self.SX_CLS = '[CLS]'
            self.SX_MASK = '[MASK]'
            self.special_sxs.extend([self.SX_PAD, self.SX_START, self.SX_END, self.SX_UNK, self.SX_CLS, self.SX_MASK])
            self.sx2idx = {sx: idx for idx, sx in enumerate(self.special_sxs)}
            self.idx2sx = {idx: sx for idx, sx in enumerate(self.special_sxs)}
            self.num_special = len(self.special_sxs)
            self.num_sxs += self.num_special


        # update vocabulary
        if samples is not None:
            if not isinstance(samples, tuple):
                samples = (samples,)
            num_samples = 0
            for split in samples:
                num_samples += self.update_voc(split)
            print('symptom vocabulary constructed using {} split and {} samples '
                  '({} symptoms with {} special symptoms)'.
                  format(len(samples), num_samples, self.num_sxs - self.num_special, self.num_special))

        # trim vocabulary
        self.trim_voc()

        assert self.num_sxs == len(self.sx2idx) == len(self.idx2sx)

    def add_symptom(self, sx: str) -> None:
        if sx not in self.sx2idx:
            self.sx2idx[sx] = self.num_sxs
            self.sx2count[sx] = 1
            self.idx2sx[self.num_sxs] = sx
            self.num_sxs += 1
        else:
            self.sx2count[sx] += 1

    def update_voc(self, samples: list) -> int:
        for sample in samples:
            for sx in sample['exp_sxs']:
                self.add_symptom(sx)
            for sx in sample['imp_sxs']:
                self.add_symptom(sx)
        return len(samples)

    def trim_voc(self):
        sxs = [sx for sx in sorted(self.sx2count, key=self.sx2count.get, reverse=True)]
        if self.min_sx_freq is not None:
            sxs = [sx for sx in sxs if self.sx2count.get(sx) >= self.min_sx_freq]
        if self.max_voc_size is not None:
            sxs = sxs[: self.max_voc_size]
        sxs = self.special_sxs + sxs
        self.sx2idx = {sx: idx for idx, sx in enumerate(sxs)}
        self.idx2sx = {idx: sx for idx, sx in enumerate(sxs)}
        self.sx2count = {sx: self.sx2count.get(sx) for sx in sxs if sx in self.sx2count}
        self.num_sxs = len(self.sx2idx)
        print('trimmed to {} symptoms with {} special symptoms'.
              format(self.num_sxs - self.num_special, self.num_special))

    def encode(self, sxs: dict, keep_unk=True, add_start=False, add_end=False):
        sx_ids, attr_ids = [], []
        if add_start:
            sx_ids.append(self.start_idx)
            attr_ids.append(self.SX_ATTR_PAD_IDX)
        for sx, attr in sxs.items():
            if sx in self.sx2idx:
                sx_ids.append(self.sx2idx.get(sx))
                attr_ids.append(self.SX_ATTR_MAP.get(attr))
            else:
                if keep_unk:
                    sx_ids.append(self.unk_idx)
                    attr_ids.append(self.SX_ATTR_MAP.get(attr))
        if add_end:
            sx_ids.append(self.end_idx)
            attr_ids.append(self.SX_ATTR_PAD_IDX)
        return sx_ids, attr_ids

    def decoder(self, sx_ids, attr_ids):
        sx_attr = {}
        for sx_id, attr_id in zip(sx_ids, attr_ids):
            if attr_id not in [self.SX_ATTR_PAD_IDX, self.SX_ATTR_NM_IDX]:
                sx_attr.update({self.idx2sx.get(sx_id): self.SX_ATTR_MAP_INV.get(attr_id)})
        return sx_attr

    def __len__(self) -> int:
        return self.num_sxs

    @property
    def pad_idx(self) -> int:
        return self.sx2idx.get(self.SX_PAD)

    @property
    def start_idx(self) -> int:
        return self.sx2idx.get(self.SX_START)

    @property
    def end_idx(self) -> int:
        return self.sx2idx.get(self.SX_END)

    @property
    def unk_idx(self) -> int:
        return self.sx2idx.get(self.SX_UNK)

    @property
    def cls_idx(self) -> int:
        return self.sx2idx.get(self.SX_CLS)

    @property
    def mask_idx(self) -> int:
        return self.sx2idx.get(self.SX_MASK)

    @property
    def pad_sx(self) -> str:
        return self.SX_PAD

    @property
    def start_sx(self) -> str:
        return self.SX_START

    @property
    def end_sx(self) -> str:
        return self.SX_END

    @property
    def unk_sx(self) -> str:
        return self.SX_UNK

    @property
    def cls_sx(self) -> str:
        return self.SX_CLS

    @property
    def mask_sx(self) -> str:
        return self.SX_MASK


class DiseaseVocab:

    def __init__(self, samples: list = None):

        # dis is short for disease
        self.dis2idx = {}
        self.idx2dis = {}
        self.dis2count = {}
        self.num_dis = 0

        # update vocabulary
        if samples is not None:
            if not isinstance(samples, tuple):
                samples = (samples,)
            num_samples = 0
            for split in samples:
                num_samples += self.update_voc(split)
            print('disease vocabulary constructed using {} split and {} samples\nnum of unique diseases: {}'.
                  format(len(samples), num_samples, self.num_dis))

    def add_disease(self, dis: str) -> None:
        if dis not in self.dis2idx:
            self.dis2idx[dis] = self.num_dis
            self.dis2count[dis] = 1
            self.idx2dis[self.num_dis] = dis
            self.num_dis += 1
        else:
            self.dis2count[dis] += 1

    def update_voc(self, samples: list) -> int:
        for sample in samples:
            self.add_disease(sample['label'])
        return len(samples)

    def __len__(self) -> int:
        return self.num_dis

    def encode(self, dis):
        return self.dis2idx.get(dis)


class SymptomDataset(Dataset):

    def __init__(self, samples, sv: SymptomVocab, dv: DiseaseVocab, keep_unk: bool,
                 add_src_start: bool = False, add_tgt_start: bool = False, add_tgt_end: bool = False, train_mode: bool = False):
        self.samples = samples
        self.sv = sv
        self.dv = dv
        self.keep_unk = keep_unk
        self.size = len(self.sv)
        self.add_src_start = add_src_start
        self.add_tgt_start = add_tgt_start
        self.add_tgt_end = add_tgt_end
        self.train_mode = train_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        exp_sx_ids, exp_attr_ids = self.sv.encode(
            sample['exp_sxs'], keep_unk=self.keep_unk, add_start=self.add_src_start)
        imp_sx_ids, imp_attr_ids = self.sv.encode(
            sample['imp_sxs'], keep_unk=self.keep_unk, add_start=self.add_tgt_start, add_end=self.add_tgt_end)
        exp_exe_ids = [0 for i in range(len(exp_sx_ids))]
        imp_exe_ids = [0]
        imp_exe_ids.extend([0 for i in range(len(imp_sx_ids) - 2)])
        imp_exe_ids.append(0)
    
        exp_sx_ids, exp_attr_ids, exp_exe_ids, imp_sx_ids, imp_attr_ids, imp_exe_ids, label = to_tensor_vla(
            exp_sx_ids, exp_attr_ids, exp_exe_ids, imp_sx_ids, imp_attr_ids, imp_exe_ids, self.dv.encode(sample['label']),  dtype=torch.long)
        item = {
            'exp_sx_ids': exp_sx_ids,
            'exp_attr_ids': exp_attr_ids,
            'exp_exe_ids': exp_exe_ids,
            'imp_sx_ids': imp_sx_ids,
            'imp_attr_ids': imp_attr_ids,
            'imp_exe_ids': imp_exe_ids,
            'label': label,
            'id': index
        }
        return item


# language model
def lm_collater(samples):
    sx_ids = pad_sequence(
        [torch.cat([sample['exp_sx_ids'], sample['imp_sx_ids']]) for sample in samples], padding_value=0)
    attr_ids = pad_sequence(
        [torch.cat([sample['exp_attr_ids'], sample['imp_attr_ids']]) for sample in samples], padding_value=0)
    labels = torch.stack([sample['label'] for sample in samples])
    ids = [sample['id'] for sample in samples]
    items = {
        'sx_ids': sx_ids,
        'attr_ids': attr_ids,
        'labels': labels,
        'ids': ids
    }
    return items




def recursive_sum(item):
    if isinstance(item, list):
        try:
            return sum(item)
        except TypeError:
            return recursive_sum(sum(item, []))
    else:
        return item


def average(numerator, denominator):
    return 0 if recursive_sum(denominator) == 0 else recursive_sum(numerator) / recursive_sum(denominator)


def to_numpy(tensors):
    arrays = {}
    for key, tensor in tensors.items():
        arrays[key] = tensor.cpu().numpy()
    return arrays


def to_numpy_(tensor):
    return tensor.detach().cpu().numpy()


def to_list(tensor):
    return to_numpy_(tensor).tolist()


def to_numpy_vla(*tensors):
    arrays = []
    for tensor in tensors:
        arrays.append(to_numpy_(tensor))
    return arrays


def to_tensor_(array, dtype=None):
    if dtype is None:
        return torch.tensor(array, device=device)
    else:
        return torch.tensor(array, dtype=dtype, device=device)


def to_tensor_vla(*arrays, dtype=None):
    tensors = []
    for array in arrays:
        tensors.append(to_tensor_(array, dtype))
    return tensors


def extract_features(sx_ids, attr_ids, sv: SymptomVocab):
    sx_feats, attr_feats = [], []
    exe_feats = []
    for idx in range(len(sx_ids)):
        sx_feat, attr_feat, exe_feat = [sv.start_idx], [sv.SX_ATTR_PAD_IDX], [sv.SX_ATTR_PAD_IDX]
        for sx_id, attr_id in zip(sx_ids[idx], attr_ids[idx]):
            if sx_id == sv.end_idx:
                break
            if attr_id not in [sv.SX_ATTR_PAD_IDX, sv.SX_ATTR_NM_IDX]:
                sx_feat.append(sx_id)
                attr_feat.append(attr_id)
                
        sx_feats.append(to_tensor_(sx_feat))
        attr_feats.append(to_tensor_(attr_feat))
        
    return sx_feats, attr_feats



def make_features_xfmr(sv: SymptomVocab, batch, si_sx_ids=None, si_attr_ids=None,  merge_act: bool = False,
                       merge_si: bool = False):
    # convert to numpy
    assert merge_act or merge_si
    sx_feats, attr_feats = [], []
    exe_feats = []
    
    si_sx_ids, si_attr_ids = to_numpy_vla(si_sx_ids, si_attr_ids)
    si_sx_feats, si_attr_feats = extract_features(si_sx_ids, si_attr_ids, sv)
    sx_feats += si_sx_feats
    attr_feats += si_attr_feats
        
    sx_feats = pad_sequence(sx_feats, padding_value=sv.pad_idx).long()
    attr_feats = pad_sequence(attr_feats, padding_value=sv.SX_ATTR_PAD_IDX).long()
    
    return sx_feats, attr_feats

class HumanAgent:

    def __init__(self, notrain_ds_loader, test_ds_loader):
        for i in notrain_ds_loader:
            tr_true = i['labels']
            # tr_pred = simulate_human_pred_diff_dx(tr_true, num_dx_s, num_dx)
            tr_pred = simulate_human_pred(tr_true, num_dx_s, num_dx)
        for i in test_ds_loader:
            te_true = i['labels']
            te_pred = simulate_human_pred(te_true, num_dx_s, num_dx)
            # te_pred = simulate_human_pred_diff_dx(te_true, num_dx_s, num_dx)
        self.n_cls = num_dx
        self.tr_pred = tr_pred
        self.te_pred = te_pred
        alpha, beta = get_dirichlet_params(0.75, 1., self.n_cls)
        prior_matr = np.eye(self.n_cls) * alpha + (np.ones(self.n_cls) - np.eye(self.n_cls)) * beta
        posterior_matr = 1. * confusion_matrix(tr_true.cpu().numpy(), tr_pred.cpu().numpy(), labels=np.arange(self.n_cls))
        
        self.acc_per_class = to_tensor_(posterior_matr.diagonal() / posterior_matr.sum(axis=1))
        # print(self.acc_per_class)
        # assert 0
        posterior_matr += prior_matr        
        posterior_matr = posterior_matr.T
        posterior_matr = (posterior_matr) / (np.sum(posterior_matr, axis=0, keepdims=True))
        
        self.posterior_matr = to_tensor_(posterior_matr)
        self.threshold_per_class = self.acc_per_class + (lambda_a - lambda_r)/ (correct + error)

    def combine_hm(self, machine_probs, h_preds):
        n_samples = machine_probs.shape[0]
        h_conf = torch.empty((n_samples, self.n_cls), device=device)
        for i in range(n_samples):
            h_conf[i] = self.posterior_matr[h_preds[i]]
        y_comb = machine_probs * h_conf 
        y_comb /= torch.sum(y_comb, axis=1, keepdims=True)
        return y_comb

    def print_acc_per_class(self, true, pred):
        conf_matrix = confusion_matrix(np.array(true), np.array(pred), labels=np.arange(self.n_cls))
        error_counts = np.sum(conf_matrix, axis=1) - np.diag(conf_matrix)
        error_per_class = error_counts / (conf_matrix.sum(axis=1) + 1e-5 )
        return error_per_class, np.sum(conf_matrix, axis=1)

    def print_error_per_class(self, true, pred, h_pred):
        conf_matrix = confusion_matrix(to_numpy_(true), to_numpy_(pred), labels=np.arange(self.n_cls))
        conf_matrix_h = confusion_matrix(to_numpy_(true), to_numpy_(h_pred), labels=np.arange(self.n_cls))
        error_counts = np.sum(conf_matrix, axis=1) - np.diag(conf_matrix)
        error_per_class = error_counts / (conf_matrix.sum(axis=1) + 1e-5 )
        error_counts_h = np.sum(conf_matrix_h, axis=1) - np.diag(conf_matrix_h)
        error_per_class_h = error_counts_h / (conf_matrix_h.sum(axis=1) + 1e-5 )
        return error_per_class_h, error_per_class

    def compute_expected_utility(self, h_preds, probs, labels):
        n_samples = h_preds.shape[0]
         
        expected_utilitys = torch.empty((n_samples), device=device)
        for i in range(n_samples):
            threshold = self.threshold_per_class[h_preds[i]]
            # no advice
            if h_preds[i] == probs[i].argmax(dim=-1):
                expected_utilitys[i] = probs[i][h_preds[i]] * probs[i][labels[i]] * (correct + error) - error
            else:
                # accept advice (= m)
                if torch.max(probs[i]) > threshold:
                    expected_utilitys[i] = probs[i][probs[i].argmax(dim=-1)] * probs[i][labels[i]]  * (correct + error) - error -lambda_a
                # rejcet advice (= h)
                else:
                    expected_utilitys[i] = (1-probs[i][h_preds[i]]) *  self.acc_per_class[h_preds[i]]  * (correct + error) - error -lambda_r
        return  expected_utilitys
    def compute_dispersed_utility(self, h_preds, probs, labels):
        n_samples = h_preds.shape[0]
        dispersed_utilitys = []
        final_preds = torch.empty((n_samples), device=device)
        # Accept Reject True False
        nums_A_R_T_F = [0, 0, 0, 0]
        advice_all = []
        advice_label = []
        for i in range(n_samples):
            # no advice
            if h_preds[i] == probs[i].argmax(dim=-1):
                final_preds[i] = h_preds[i]
                if final_preds[i] == labels[i]:
                    dispersed_utility = 1
                    nums_A_R_T_F[2] += 1
                else:
                    dispersed_utility = -error
            else:
                threshold = self.threshold_per_class[h_preds[i]]
                # threshold = self.threshold_per_class[probs[i].argmax(dim=-1)]
                # accept advice (= m)
                advice_all.append(probs[i].argmax(dim=-1).item())
                advice_label.append(labels[i].item())
                if h_preds[i] == labels[i]:
                    nums_A_R_T_F[3] += -1
                else:
                    nums_A_R_T_F[3] += error
                if torch.max(probs[i]) > threshold:
                    nums_A_R_T_F[0] += - lambda_a
                    final_preds[i] = probs[i].argmax(dim=-1)
                    if final_preds[i] == labels[i]:
                        dispersed_utility = 1 - lambda_a
                        nums_A_R_T_F[2] += 1
                        nums_A_R_T_F[3] += 1
                    else:
                        dispersed_utility = -error - lambda_a
                        nums_A_R_T_F[3] += -error
                    
                # rejcet advice (= h)
                else:
                    nums_A_R_T_F[1] += - lambda_r
                    final_preds[i] = h_preds[i]
                    if final_preds[i] == labels[i]:
                        dispersed_utility = 1 - lambda_r
                        nums_A_R_T_F[2] += 1
                        nums_A_R_T_F[3] += 1
                    else:
                        dispersed_utility = -error - lambda_r
                        nums_A_R_T_F[3] += -error
                
            dispersed_utilitys.append(dispersed_utility)
        nums_A_R_T_F = [i/n_samples for i in nums_A_R_T_F]
        return  sum(dispersed_utilitys)/n_samples, final_preds, nums_A_R_T_F, advice_all, advice_label
    
    def init_sx_ids(self, bsz):
        return 0

def simulate_human_pred(true_labels, id_s, id_e):
    predicted_labels = torch.randint(0, num_dx, (len(true_labels),), device='cuda:0')
    for i in range(id_s):
        indices = (true_labels == i).nonzero(as_tuple=True)[0]
        correct_predictions = int(len(indices) * human_acc)
        incorrect_predictions = len(indices) - correct_predictions
        correct_indices = np.random.choice(indices.cpu().numpy(), correct_predictions, replace=False)
        incorrect_indices = list(set(indices.cpu().numpy()) - set(correct_indices))
        predicted_labels[correct_indices] = i
        for idx in incorrect_indices:
            predicted_labels[idx] = (i + np.random.randint(1, num_dx-1)) % num_dx
    for i in range(id_s, id_e):
        indices = (true_labels == i).nonzero(as_tuple=True)[0]
        correct_predictions = int(len(indices) * human_weak_acc)
        incorrect_predictions = len(indices) - correct_predictions
        correct_indices = np.random.choice(indices.cpu().numpy(), correct_predictions, replace=False)
        incorrect_indices = list(set(indices.cpu().numpy()) - set(correct_indices))
        predicted_labels[correct_indices] = i
        for idx in incorrect_indices:
            predicted_labels[idx] = (i + np.random.randint(1, num_dx-1)) % num_dx
    return predicted_labels

def simulate_human_pred_diff_dx(true_labels, id_s, id_e):
    predicted_labels = torch.randint(0, num_dx, (len(true_labels),), device='cuda:0')
    for i in range(id_e):
        indices = (true_labels == i).nonzero(as_tuple=True)[0]
        if i == id_s:
            correct_predictions = int(len(indices) * human_weak_acc)
        else:
            correct_predictions = int(len(indices) * human_acc)
        incorrect_predictions = len(indices) - correct_predictions
        correct_indices = np.random.choice(indices.cpu().numpy(), correct_predictions, replace=False)
        incorrect_indices = list(set(indices.cpu().numpy()) - set(correct_indices))
        predicted_labels[correct_indices] = i
        for idx in incorrect_indices:
            predicted_labels[idx] = (i + np.random.randint(1, num_dx-1)) % num_dx
    return predicted_labels

def get_dirichlet_params(acc, strength, n_cls):
    beta = 0.1
    alpha = beta * (n_cls - 1) * acc / (1. - acc)

    alpha *= strength
    beta *= strength

    alpha += 1
    beta += 1

    return alpha, beta



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True