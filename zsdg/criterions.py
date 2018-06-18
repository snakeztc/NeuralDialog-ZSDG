# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
from __future__ import print_function
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
import torch
from zsdg.utils import INT, FLOAT, LONG, cast_type
import logging


class L2Loss(_Loss):

    logger = logging.getLogger()
    def forward(self, state_a, state_b):
        if type(state_a) is tuple:
            losses = 0.0
            for s_a, s_b in zip(state_a, state_b):
                losses += torch.pow(s_a-s_b, 2)
        else:
            losses = torch.pow(state_a-state_b, 2)
        return torch.mean(losses)


class NLLEntropy(_Loss):

    logger = logging.getLogger()
    def __init__(self, padding_idx, config, rev_vocab=None, key_vocab=None):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx
        self.avg_type = config.avg_type

        if rev_vocab is None or key_vocab is None:
            self.weight = None
        else:
            self.logger.info("Use extra cost for key words")
            weight = np.ones(len(rev_vocab))
            for key in key_vocab:
                weight[rev_vocab[key]] = 10.0
            self.weight = cast_type(torch.from_numpy(weight), FLOAT,
                                    config.use_gpu)

    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        input = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)
        if self.avg_type is None:
            loss = F.nll_loss(input, target, size_average=False,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
        elif self.avg_type == 'seq':
            loss = F.nll_loss(input, target, size_average=False,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
            loss = loss / batch_size
        elif self.avg_type == 'real_word':
            loss = F.nll_loss(input, target, size_average=True,
                              ignore_index=self.padding_idx,
                              weight=self.weight, reduce=False)
            loss = loss.view(-1, net_output.size(1))
            loss = torch.sum(loss, dim=1)
            word_cnt = torch.sum(torch.sign(labels), dim=1).float()
            loss = loss/word_cnt
            loss = torch.mean(loss)
        elif self.avg_type == 'word':
            loss = F.nll_loss(input, target, size_average=True,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
        else:
            raise ValueError("Unknown avg type")

        return loss


