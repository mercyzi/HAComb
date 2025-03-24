# These are all neural networks.
import os
import torch
import torch.nn as nn
from torch.nn import functional
from torch.distributions import Categorical
from tqdm import tqdm
import math
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import SymptomVocab, device, to_numpy_
from conf import *

weight_cus = 0

class SymptomEncoderXFMR(nn.Module):

    def __init__(self, num_sxs, num_dis):
        super().__init__()

        self.num_dis = num_dis
        self.sx_embedding = nn.Embedding(num_sxs, enc_emb_dim, padding_idx=0)
        self.attr_embedding = nn.Embedding(num_attrs, enc_emb_dim, padding_idx=0)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=enc_emb_dim,
                nhead=enc_num_heads,
                dim_feedforward=enc_num_layers,
                dropout=enc_dropout,
                batch_first=False,
                activation='gelu'),
            num_layers=enc_num_layers)

        self.dis_fc = nn.Linear(enc_emb_dim, num_dis, bias=True)

    def forward(self, sx_ids, attr_ids, mask=None, src_key_padding_mask=None):
        inputs = self.sx_embedding(sx_ids) + self.attr_embedding(attr_ids) 
        outputs = self.encoder(inputs, mask, src_key_padding_mask)
        return outputs

    # mean pooling feature
    def get_mp_features(self, sx_ids, attr_ids, pad_idx):
        src_key_padding_mask = sx_ids.eq(pad_idx).transpose(1, 0).contiguous()
        outputs = self.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
        seq_len, batch_size, emb_dim = outputs.shape
        mp_mask = (1 - sx_ids.eq(pad_idx).int())
        
        mp_mask_ = mp_mask.unsqueeze(-1).expand(seq_len, batch_size, emb_dim)
        
        avg_outputs = torch.sum(outputs * mp_mask_, dim=0) / torch.sum(mp_mask, dim=0).unsqueeze(-1)
        
        features = self.dis_fc(avg_outputs)
        
        return features

    def predict(self, sx_ids, attr_ids, pad_idx):
        outputs = self.get_mp_features(sx_ids, attr_ids, pad_idx)
        labels = outputs.argmax(dim=-1)
        return labels

    def inference(self, sx_ids, attr_ids, pad_idx):
        return self.simulate(sx_ids, attr_ids, pad_idx, inference=True)

    def compute_entropy(self, features):
        return torch.distributions.Categorical(functional.softmax(features, dim=-1)).entropy().item() / self.num_dis

    @staticmethod
    def compute_max_prob(features):
        return torch.max(functional.softmax(features, dim=-1))


class Agent(nn.Module):

    def __init__(self, num_sxs: int, num_dis: int, sv, graph=None):

        super().__init__()

        self.symptom_encoder = SymptomEncoderXFMR(
           num_sxs, num_dis
        )
        self.num_sxs = num_sxs
        self.sv = sv

    def forward(self):
        pass

        
    def load(self, path):
        if os.path.exists(path):
            state_dict = torch.load(path)
            shared_state_dict = {k: v for k, v in state_dict.items() if k in self.state_dict() and 'gnnmodel' not in k}
            self.load_state_dict(shared_state_dict, strict=False)
            if verbose:
                print('loading pre-trained parameters from {} ...'.format(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
        if verbose:
            print('saving best model to {}'.format(path))
