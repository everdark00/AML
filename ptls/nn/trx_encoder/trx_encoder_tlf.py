from collections import OrderedDict

import torch
import pandas as pd
from torch import nn as nn

from ptls.data_load.padded_batch import PaddedBatch
from ptls.preprocessing.deeptlf.src import DeepTLF


class TrxEncoderTLF(nn.Module):
    def __init__(self,  
                 encoder : DeepTLF =  None,
                 feature_names : list = None
                 ):
        super().__init__()

        self.encoder_conditions = encoder.TDE_encoder.all_conditions
        self.feature_names = feature_names
                

    def forward(self, x: PaddedBatch):
        out = []
        for condition in self.encoder_conditions:
            out.append((x.payload[condition["feature"]] < condition["threshold"]).unsqueeze(2))
        out = torch.cat(out, dim=2).float()
        return PaddedBatch(out, x.seq_lens)


    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        return len(self.encoder_conditions)
