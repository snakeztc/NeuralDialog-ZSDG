# @Time    : 2/22/18 5:27 PM
# @Author  : Tiancheng Zhao
from __future__ import print_function

from zsdg import utils
import numpy as np
from zsdg.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN, DecoderPointerGen
import logging
from zsdg.main import get_sent
import pickle
import os

logger = logging.getLogger()


def generate(model, data_feed, config, evaluator, num_batch=1, dest_f=None):
    model.eval()
    de_tknize = utils.get_dekenize()

    def write(msg):
        if msg is None or msg == '':
            return
        if dest_f is None:
            logger.info(msg)
        else:
            dest_f.write(msg + '\n')

    data_feed.epoch_init(config, shuffle=num_batch is not None, verbose=False)
    evaluator.initialize()
    logger.info("Generation: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    batch_cnt = 0
    while True:
        batch_cnt += 1
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)

        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in
                       outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        true_labels = labels.data.numpy()
        # get attention if possible
        if config.use_attn or config.use_ptr:
            pred_attns = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_ATTN_SCORE]]
            pred_attns = np.array(pred_attns, dtype=float).squeeze(2).swapaxes(0,1)
        else:
            pred_attns = None

        # get last 1 context
        ctx = batch.get('contexts')
        ctx_len = batch.get('context_lens')
        domains = batch.domains
        attn_ctx = outputs.get(DecoderPointerGen.KEY_PTR_CTX)
        sel_acts = outputs.get(DecoderPointerGen.KEY_POLICY)
        if sel_acts is not None:
            sel_acts = sel_acts.cpu().data.numpy()

        if attn_ctx is not None:
            attn_ctx = attn_ctx.cpu().data.numpy()
            attn_ctx = attn_ctx.reshape(attn_ctx.shape[0], -1)

        # logger.info the batch in String.
        for b_id in range(pred_labels.shape[0]):
            pred_str, attn = get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns, attn_ctx=attn_ctx)
            true_str, _ = get_sent(model, de_tknize, true_labels, b_id)
            prev_ctx = ""
            act_str = ""
            if ctx is not None:
                ctx_str = []
                for t_id in range(ctx_len[b_id]):
                    temp_str, _ = get_sent(model, de_tknize, ctx[:, t_id, :], b_id, stop_eos=False)
                    ctx_str.append(temp_str)
                ctx_str = '|'.join(ctx_str)[-200::]
                prev_ctx = "Source: {}".format(ctx_str)

            if sel_acts is not None:
                act_str, _ = get_sent(model, de_tknize, sel_acts, b_id, stop_eos=False, stop_pad=False)
                act_str = "Acts: {}".format(act_str)

            domain = domains[b_id]
            evaluator.add_example(true_str, pred_str, domain)

            if num_batch is None or batch_cnt < 2:
                write(prev_ctx)
                write(act_str)
                write("{}:: True: {} ||| Pred: {}".format(domain, true_str, pred_str))
                if attn:
                    write("[[{}]]".format(attn))
                write("-")

    write(evaluator.get_report(include_error=dest_f is not None))
    logger.info("Generation Done")


def dump_latent(model, data_feed, config, log_dir):
    model.eval()
    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, verbose=False, shuffle=False)
    logger.info("Dumping: {} batches".format(data_feed.num_batch))
    all_zs = []
    all_metas = []
    while True:
        batch = data_feed.next_batch()
        if batch is None:
            break
        results = model(batch, mode=TEACH_FORCE, return_latent=True)

        labels = batch.outputs
        domains = batch.domains
        acts = batch.get('output_actions')

        latent_acts = results.latent_actions
        if type(latent_acts) is tuple:
            latent_acts = list(latent_acts[0].cpu().data.numpy())
        else:
            latent_acts = list(latent_acts.cpu().data.numpy())

        for b_id in range(labels.shape[0]):
            true_str, _ = get_sent(model, de_tknize, labels, b_id)
            act_str, _ = get_sent(model, de_tknize, acts, b_id)
            all_metas.append({'utt': true_str, 'domain': domains[b_id], 'acts':act_str})

        all_zs.extend(latent_acts)

    pickle.dump({'z': all_zs, "metas": all_metas}, open(os.path.join(log_dir,
                                                                     "latent-{}.p".format(utils.get_time())), 'wb'))
    logger.info("Dumping Done")
