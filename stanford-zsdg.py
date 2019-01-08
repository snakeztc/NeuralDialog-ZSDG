# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
from __future__ import print_function

from zsdg.dataset.corpora import ZslStanfordCorpus, SYS
from zsdg.dataset import data_loaders
from zsdg.models import models
from zsdg.main import train, validate
from zsdg import hred_utils
from zsdg.utils import str2bool, prepare_dirs_loggers, get_time, process_config
from zsdg import evaluators
import argparse
import os
import torch

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, nargs='+', default=['data/stanford'])
data_arg.add_argument('--log_dir', type=str, default='logs')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--rnn_cell', type=str, default='lstm')
net_arg.add_argument('--embed_size', type=int, default=200)
net_arg.add_argument('--utt_type', type=str, default='rnn')
net_arg.add_argument('--utt_cell_size', type=int, default=256)
net_arg.add_argument('--ctx_cell_size', type=int, default=512)
net_arg.add_argument('--dec_cell_size', type=int, default=512)
net_arg.add_argument('--bi_ctx_cell', type=str2bool, default=False)
net_arg.add_argument('--max_utt_len', type=int, default=20)
net_arg.add_argument('--max_dec_len', type=int, default=40)
net_arg.add_argument('--num_layer', type=int, default=1)
net_arg.add_argument('--use_attn', type=str2bool, default=True)
net_arg.add_argument('--attn_type', type=str, default='cat')

train_arg = add_argument_group('Training')
train_arg.add_argument('--op', type=str, default='adam')
train_arg.add_argument('--backward_size', type=int, default=14)
train_arg.add_argument('--step_size', type=int, default=2)
train_arg.add_argument('--grad_clip', type=float, default=3.0)
train_arg.add_argument('--init_w', type=float, default=0.08)
train_arg.add_argument('--init_lr', type=float, default=0.001)
train_arg.add_argument('--momentum', type=float, default=0.0)
train_arg.add_argument('--lr_hold', type=int, default=1)
train_arg.add_argument('--lr_decay', type=float, default=0.6)
train_arg.add_argument('--dropout', type=float, default=0.3)
train_arg.add_argument('--improve_threshold', type=float, default=0.996)
train_arg.add_argument('--patient_increase', type=float, default=2.0)
train_arg.add_argument('--early_stop', type=str2bool, default=True)
train_arg.add_argument('--max_epoch', type=int, default=50)
train_arg.add_argument('--preview_batch_num', type=int, default=50)
train_arg.add_argument('--include_domain', type=str2bool, default=True)
train_arg.add_argument('--include_example', type=str2bool, default=False)
train_arg.add_argument('--include_state', type=str2bool, default=True)

# MISC
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--save_model', type=str2bool, default=True)
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--print_step', type=int, default=100)
misc_arg.add_argument('--ckpt_step', type=int, default=400)
misc_arg.add_argument('--batch_size', type=int, default=20)
misc_arg.add_argument('--gen_type', type=str, default='greedy')
misc_arg.add_argument('--avg_type', type=str, default='word')
misc_arg.add_argument('--beam_size', type=int, default=20)

# KEY PARAMETERS

# decide which domains are excluded from the training
train_arg.add_argument('--black_domains', type=str, nargs='*', default=['schedule'])
train_arg.add_argument('--black_ratio', type=float, default=1.0)
train_arg.add_argument('--target_example_cnt', type=int, default=150)

# Which model is used
net_arg.add_argument('--action_match', type=str2bool, default=True)
net_arg.add_argument('--use_ptr', type=str2bool, default=True)

# Where to load existing model
misc_arg.add_argument('--forward_only', type=str2bool, default=False)
misc_arg.add_argument('--load_sess', type=str, default="ENTER_YOUR_PATH_HERE")


def main(config):
    prepare_dirs_loggers(config, os.path.basename(__file__))

    corpus_client = ZslStanfordCorpus(config)
    warmup_data = corpus_client.get_seed_responses(config.target_example_cnt)
    dial_corpus = corpus_client.get_corpus()
    train_dial, valid_dial, test_dial = dial_corpus['train'],\
                                        dial_corpus['valid'],\
                                        dial_corpus['test']

    evaluator = evaluators.BleuEntEvaluator("SMD", corpus_client.ent_metas)

    # create data loader that feed the deep models
    train_feed = data_loaders.ZslSMDDialDataLoader("Train", train_dial, config, warmup_data)
    valid_feed = data_loaders.ZslSMDDialDataLoader("Valid", valid_dial, config)
    test_feed = data_loaders.ZslSMDDialDataLoader("Test", test_dial, config)
    if config.action_match:
        if config.use_ptr:
            model = models.ZeroShotPtrHRED(corpus_client, config)
        else:
            model = models.ZeroShotHRED(corpus_client, config)
    else:
        if config.use_ptr:
            model = models.PtrHRED(corpus_client, config)
        else:
            model = models.HRED(corpus_client, config)

    if config.forward_only:
        session_dir = os.path.join(config.log_dir, config.load_sess)
        test_file = os.path.join(session_dir, "{}-test-{}.txt".format(get_time(),
                                                         config.gen_type))
        model_file = os.path.join(config.log_dir, config.load_sess, "model")
    else:
        session_dir = config.session_dir
        test_file = os.path.join(config.session_dir,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        model_file = os.path.join(config.session_dir, "model")

    if config.use_gpu:
        model.cuda()

    if config.forward_only is False:

        try:
            train(model, train_feed, valid_feed, test_feed, config, evaluator, gen=hred_utils.generate)
        except KeyboardInterrupt:
            print("Training stopped by keyboard.")

    config.batch_size = 20
    model.load_state_dict(torch.load(model_file))

    # hred_utils.dump_latent(model, test_feed, config, session_dir)

    # run the model on the test dataset.
    validate(model, test_feed, config)

    with open(os.path.join(test_file), "wb") as f:
        hred_utils.generate(model, test_feed, config, evaluator, num_batch=None, dest_f=f)


if __name__ == "__main__":
    config, unparsed = get_config()
    config = process_config(config)
    main(config)
