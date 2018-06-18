#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
from __future__ import print_function
import numpy as np
from zsdg.utils import Pack
from zsdg.dataset.corpora import SimDialCorpus, SYS, USR
from zsdg.dataset.dataloader_bases import DataLoader, LongDataLoader
import logging


class ZslSMDDialDataLoader(DataLoader):
    def __init__(self, name, data, config, warmup_data=None):
        super(ZslSMDDialDataLoader, self).__init__(name)
        self.max_utt_size = config.max_utt_len

        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        data_lens = [len(line.context) for line in self.data]
        if False:
            self.indexes = list(np.argsort(data_lens))[::-1]
        else:
            self.indexes = range(len(data_lens))

        # prepare indexes for warm up
        self.warmup_data = warmup_data
        if self.warmup_data is not None:
            self.warmup_size = len(self.warmup_data)
            self.warmup_indexes = range(self.warmup_size)
        self.warmup_flags = None
        self.warmup_num_batch = None

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)):
                e_id = i
                s_id = max(0, e_id - backward_size)
                response = dialog[i].copy()
                if response.speaker == USR:
                    continue
                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                # response['kb'] = [self.pad_to(self.max_utt_size, item, do_pad=True) for item in response.kb]

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)
                results.append(Pack(context=contexts, response=response))
        return results

    def epoch_init(self, config, shuffle=True, verbose=True):
        super(ZslSMDDialDataLoader, self).epoch_init(config, shuffle, verbose)
        self.warmup_flags = [False] * self.num_batch

        if self.warmup_data is None:
            return

        self.warmup_num_batch = int(self.warmup_size / config.batch_size)
        for i in range(self.warmup_num_batch):
            self.batch_indexes.append(np.random.choice(self.warmup_indexes, config.batch_size, replace=False))
            self.warmup_flags.append(True)

        if shuffle:
            temp_batch_id = range(len(self.warmup_flags))
            np.random.shuffle(temp_batch_id)
            self.batch_indexes = [self.batch_indexes[i] for i in temp_batch_id]
            self.warmup_flags = [self.warmup_flags[i] for i in temp_batch_id]

        if verbose:
            self.logger.info("%s add with %d warm up batches" % (self.name, self.warmup_num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            is_warmup = self.warmup_flags[self.ptr]
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1

            if is_warmup:
                return self._prepare_warmup_batch(selected_ids)
            else:
                return self._prepare_batch(selected_ids)
        else:
            return None

    def _prepare_batch(self, selected_index):
        # the batch index, the starting point and end point for segment
        rows = [self.data[idx] for idx in selected_index]

        cxt_lens, ctx_utts = [], []
        out_utts, out_lens = [], []
        domains, domain_metas = [], []

        for row in rows:
            in_row, out_row = row.context, row.response

            # source context
            batch_ctx = []
            #for item in out_row.kb:
            #    batch_ctx.append(item)
            for turn in in_row:
                batch_ctx.append(self.pad_to(self.max_utt_size, turn.utt))

            cxt_lens.append(len(batch_ctx))
            ctx_utts.append(batch_ctx)

            # target response
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            domains.append(out_row.domain)
            domain_metas.append(out_row.domain_id)

        domain_metas = np.array(domain_metas)
        vec_ctx_lens = np.array(cxt_lens)
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((self.batch_size, max_ctx_len, self.max_utt_size), dtype=np.int32)
        vec_ctx_confs = np.ones((self.batch_size, max_ctx_len), dtype=np.float32)

        vec_out_utts = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)

        for b_id in range(self.batch_size):
            vec_out_utts[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_ctx_utts[b_id, 0:vec_ctx_lens[b_id], :] = ctx_utts[b_id]

        return Pack(context_lens=vec_ctx_lens, contexts=vec_ctx_utts, context_confs=vec_ctx_confs,
                    output_lens=vec_out_lens, outputs=vec_out_utts,
                    domains=domains, domain_metas=domain_metas)

    def _prepare_warmup_batch(self, selected_ids):
        # the batch index, the starting point and end point for segment
        rows = [self.warmup_data[idx] for idx in selected_ids]
        out_utts, out_lens = [], []
        out_acts, out_act_lens = [], []
        domains, domain_metas = [], []

        for row in rows:
            out_utt = [t for idx, t in enumerate(row.utt)]

            # target response
            out_acts.append(row.actions)
            out_act_lens.append(len(row.actions))

            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            domains.append(row.domain)
            domain_metas.append(row.domain_id)

        vec_out_lens = np.array(out_lens)
        domain_metas = np.array(domain_metas)
        vec_out_utts = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_acts = np.zeros((self.batch_size, np.max(out_act_lens)), dtype=np.int32)

        for b_id in range(self.batch_size):
            vec_out_utts[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_out_acts[b_id, 0:out_act_lens[b_id]] = out_acts[b_id]

        return Pack(output_lens=vec_out_lens, outputs=vec_out_utts, output_actions=vec_out_acts,
                    domains=domains, domain_metas=domain_metas)


class SimDialDataLoader(LongDataLoader):
    def __init__(self, name, data, domain_meta, config, warmup_data=None):
        super(SimDialDataLoader, self).__init__(name)
        self.max_utt_size = config.max_utt_len
        self.data = data
        self.domain_meta = self.prepare_domain_meta(domain_meta)
        self.data_size = len(data)
        self.data_lens = [len(line) for line in self.data]
        self.indexes = list(np.argsort(self.data_lens))[::-1]

        # prepare indexes for warm up
        self.warmup_data = warmup_data
        if self.warmup_data is not None:
            self.warmup_size = len(self.warmup_data)
            self.warmup_indexes = range(self.warmup_size)
        self.warmup_flags = None
        self.warmup_num_batch = None

        # Pretty printing
        covered_data = np.where(np.array(self.data_lens) < config.backward_size)[0]
        coverage = len(covered_data) / float(self.data_size)
        msg = "Initialized {} Max len {} Min len {} Avg len {} Max ctx {} covers {}" \
            .format(self.name, np.max(self.data_lens), np.min(self.data_lens),
                    np.average(self.data_lens), config.backward_size, coverage)
        self.logger.info(msg)

    def prepare_domain_meta(self, domain_meta):
        # pre-compute domain meta since it's independent of dialogs
        # domain description just slot names
        # domain sys/usr templates example sys or user uttearnecs
        vec_domain_meta = {}

        for domain, meta in domain_meta.items():
            if type(meta) is not Pack:
                continue
            sys_templates = []
            sys_acts = []
            usr_templates = []
            usr_acts = []
            for template, act in zip(meta.templates, meta.acts):
                padded_template = self.pad_to(self.max_utt_size, template)
                if domain_meta.sys_id in padded_template:
                    # warmup_data.append(Pack(domain=domain, utt=template, act=act))
                    sys_templates.append(padded_template)
                    sys_acts.append(act)
                else:
                    usr_templates.append(padded_template)
                    usr_acts.append(act)

            padded_desc = self.pad_to(self.max_utt_size, meta.description)
            vec_domain_meta[domain] = Pack(sys_templates=sys_templates,
                                           sys_acts=sys_acts,
                                           usr_templates=usr_templates,
                                           usr_acts=usr_acts,
                                           description=padded_desc)

        return vec_domain_meta

    def epoch_init(self, config, shuffle=True, verbose=True):
        super(SimDialDataLoader, self).epoch_init(config, shuffle, verbose)
        self.warmup_flags = [False] * self.num_batch

        if self.warmup_data is None:
            return

        self.warmup_num_batch = int(self.warmup_size/config.batch_size)
        for i in range(self.warmup_num_batch):
            self.grid_indexes.append(np.random.choice(self.warmup_indexes, config.batch_size, replace=False))
            self.warmup_flags.append(True)

        if shuffle:
            temp_batch_id = range(len(self.warmup_flags))
            np.random.shuffle(temp_batch_id)
            self.grid_indexes = [self.grid_indexes[i] for i in temp_batch_id]
            self.warmup_flags = [self.warmup_flags[i] for i in temp_batch_id]

        if verbose:
            self.logger.info("%s add with %d warm up batches" % (self.name, self.warmup_num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            is_warmup = self.warmup_flags[self.ptr]
            current_grid = self.grid_indexes[self.ptr]

            if is_warmup:
                self.ptr += 1
                return self._prepare_warmup_batch(current_grid)
            else:
                if self.ptr > 0:
                    prev_grid = self.grid_indexes[self.ptr - 1]
                else:
                    prev_grid = None
                self.ptr += 1
                return self._prepare_batch(cur_grid=current_grid,
                                           prev_grid=prev_grid)
        else:
            return None

    def _prepare_batch(self, cur_grid, prev_grid):
        # the batch index, the starting point and end point for segment
        b_id, s_id, e_id = cur_grid

        batch_ids = self.batch_indexes[b_id]
        rows = [self.data[idx] for idx in batch_ids]
        cxt_lens, ctx_utts, ctx_confs = [], [], []
        out_utts, out_lens = [], []
        out_acts, out_act_lens = [], []
        # sys_templates, sys_acts, sys_lens = [], [], []
        # usr_templates, usr_acts, usr_lens = [], [], []
        domains, domain_metas= [], []

        for row in rows:
            if s_id < len(row) - 1:
                if s_id > 0:
                    cut_row = row[0:1] + row[s_id+1:e_id]
                else:
                    cut_row = row[s_id:e_id]

                in_row, out_row = cut_row[0:-1], cut_row[-1]
                out_utt = out_row.utt

                # source context
                cxt_lens.append(len(in_row))
                batch_ctx, batch_confs = [], []
                for turn in in_row:
                    batch_ctx.append(self.pad_to(self.max_utt_size, turn.utt))
                    batch_confs.append(turn.conf)

                ctx_utts.append(batch_ctx)
                ctx_confs.append(batch_confs)

                # target response
                out_utts.append(out_utt)
                out_lens.append(len(out_utt))

                out_acts.append(out_row.actions)
                out_act_lens.append(len(out_row.actions))

                domains.append(out_row.domain)
                domain_metas.append(self.domain_meta[out_row.domain].description)

                #sys_templates.append(self.domain_meta[out_row.domain].sys_templates)
                #sys_acts.append(self.domain_meta[out_row.domain].sys_acts)
                #sys_lens.append(len(sys_templates[-1]))

                #usr_templates.append(self.domain_meta[out_row.domain].usr_templates)
                #usr_acts.append(self.domain_meta[out_row.domain].usr_acts)
                #usr_lens.append(len(usr_templates[-1]))

            else:
                raise ValueError("s_id %d larger than row" % s_id)

        vec_ctx_lens = np.array(cxt_lens)
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((self.batch_size, max_ctx_len, self.max_utt_size), dtype=np.int32)
        vec_ctx_confs = np.zeros((self.batch_size, max_ctx_len), dtype=np.float32)

        vec_out_utts = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_acts = np.zeros((self.batch_size, np.max(out_act_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)

        #vec_sys_templates = np.zeros((self.batch_size, np.max(sys_lens), self.max_utt_size), dtype=np.int32)
        #vec_sys_acts = np.zeros((self.batch_size, np.max(sys_lens), len(sys_acts[0][0])), dtype=np.int32)

        #vec_usr_templates = np.zeros((self.batch_size, np.max(usr_lens), self.max_utt_size), dtype=np.int32)
        #vec_usr_acts = np.zeros((self.batch_size, np.max(usr_lens), len(usr_acts[0][0])), dtype=np.int32)

        vec_domain_metas = np.zeros((self.batch_size, self.max_utt_size), dtype=np.int32)

        for b_id in range(self.batch_size):
            vec_out_utts[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_out_acts[b_id, 0:out_act_lens[b_id]] = out_acts[b_id]

            vec_ctx_confs[b_id, 0:vec_ctx_lens[b_id]] = ctx_confs[b_id]
            vec_ctx_utts[b_id, 0:vec_ctx_lens[b_id], :] = ctx_utts[b_id]

            #vec_sys_templates[b_id, 0:sys_lens[b_id], :] = sys_templates[b_id]
            #vec_sys_acts[b_id, 0:sys_lens[b_id], :] = sys_acts[b_id]

            vec_domain_metas[b_id, :] = domain_metas[b_id]

            #vec_usr_templates[b_id, 0:usr_lens[b_id], :] = usr_templates[b_id]
            #vec_usr_acts[b_id, 0:usr_lens[b_id]] = usr_acts[b_id]

        return Pack(context_lens=vec_ctx_lens, contexts=vec_ctx_utts, context_confs=vec_ctx_confs,
                    output_lens=vec_out_lens, outputs=vec_out_utts, output_actions=vec_out_acts,
                    domains=domains, domain_metas=vec_domain_metas)

    def _prepare_warmup_batch(self, selected_ids):
        # the batch index, the starting point and end point for segment
        rows = [self.warmup_data[idx] for idx in selected_ids]
        out_utts, out_lens = [], []
        out_acts, out_act_lens = [], []
        domains, domain_metas = [], []

        for row in rows:
            out_utt = row.utt
            # target response
            out_acts.append(row.actions)
            out_act_lens.append(len(row.actions))

            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            domains.append(row.domain)
            domain_metas.append(self.domain_meta[row.domain].description)

        vec_out_lens = np.array(out_lens)
        vec_out_utts = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_acts = np.zeros((self.batch_size, np.max(out_act_lens)), dtype=np.int32)
        vec_domain_metas = np.zeros((self.batch_size, self.max_utt_size), dtype=np.int32)

        for b_id in range(self.batch_size):
            vec_out_utts[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_out_acts[b_id, 0:out_act_lens[b_id]] = out_acts[b_id]
            vec_domain_metas[b_id, :] = domain_metas[b_id]

        return Pack(output_lens=vec_out_lens, outputs=vec_out_utts, output_actions=vec_out_acts,
                    domains=domains, domain_metas=vec_domain_metas)

