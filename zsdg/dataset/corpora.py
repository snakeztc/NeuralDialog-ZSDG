# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
from __future__ import unicode_literals  # at top of module
from collections import Counter
import numpy as np
import json
from zsdg.utils import get_tokenize, get_chat_tokenize, missingdict, Pack
import logging
import os
import itertools
from collections import defaultdict
import copy

PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'
BOD = "<d>"
EOD = "</d>"
BOT = "<t>"
EOT = "</t>"
ME = "<me>"
OT = "<ot>"
SYS = "<sys>"
USR = "<usr>"
KB = "<kb>"
SEP = "|"
REQ = "<requestable>"
INF = "<informable>"
WILD = "%s"


class SimDialCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config, max_vocab_size=10000):
        self.config = config
        self.max_utt_len = config.max_utt_len
        self.black_domains = config.black_domains
        self.black_ratio = config.black_ratio
        self.include_domain = config.include_domain
        self.include_example = config.include_example
        self.include_state = config.include_state
        self.tokenize = get_tokenize()

        train_data, train_meta = self._read_file(self.config.train_dir, is_train=True)
        test_data, test_meta = self._read_file(self.config.test_dir, is_train=False)

        # combine train and test domain meta
        train_meta.update(test_meta)
        self.domain_meta = self._process_meta(train_meta)
        self.corpus = self._process_dialog(train_data)
        self.test_corpus = self._process_dialog(test_data)
        self.logger.info("Loaded Corpus with %d, test %d"
                         % (len(self.corpus), len(self.test_corpus)))

        # build up a vocabulary
        self.vocab, self.rev_vocab = self._build_vocab(max_vocab_size)

        self.logger.info("Done loading corpus")

    def _read_file(self, paths, is_train):
        """
        Read data from file
        :param paths: a list of path or a string path
        :return: a list of dialogs
        """
        dialogs = []
        metas = {}
        if type(paths) is not list:
            paths = [paths]
        for path in paths:
            data = json.load(open(path, 'rb'))
            meta = data['meta']
            conversations = data['dialogs']
            metas[meta['name']] = meta

            if is_train and self.config.data_cap is not None \
                    and self.config.data_cap < len(conversations):
                data_size = min(self.config.data_cap, len(conversations))
                self.logger.info("Capped {} data to {}".format(path, data_size))
                conversations = conversations[0:data_size]

            dialogs.extend(conversations)

        return dialogs, metas

    def _dict_to_str(self, name, data, keys=None):
        if keys is None:
            keys = data.keys()
        return "%s %s" % (name, " ".join(['%s: %s' % (k, data[k]) for k in keys]))

    def _process_meta(self, domain_meta):
        all_norm_meta = {}
        for domain_name, domain in domain_meta.items():
            norm_meta = Pack()

            nlg_spec = domain['nlg_spec']
            usr_slots = {"#" + s[0]: s[2] for s in domain['usr_slots']}
            sys_slots = {"#" + s[0]: s[2] for s in domain['sys_slots']}
            sys_slots['#default'] = [str(t) for t in range(domain['db_size'])]

            # save the dictionary
            norm_meta['usr_slots'] = usr_slots
            norm_meta['sys_slots'] = sys_slots
            norm_meta['usr_id2slot'] = usr_slots.keys()
            norm_meta['sys_id2slot'] = sys_slots.keys()
            norm_meta['greet'] = [BOS, SYS] + self.tokenize(domain['greet']) + [EOS]

            # add KB searches
            kb_meta = []
            for i in range(1):
                sample = {}
                for usr_key in norm_meta.usr_id2slot:
                    sample[usr_key] = np.random.choice(usr_slots[usr_key])
                sample_ret = np.random.choice(norm_meta.sys_id2slot)
                search_str = self._dict_to_str("QUERY", sample, norm_meta.usr_id2slot)
                search_str += " RET {}".format(sample_ret)
                search_tkns = [BOS, SYS] + self.tokenize(search_str) + [EOS]
                kb_meta.append(Pack(slot="#QUERY", is_usr=False,
                                    intent="query", utt=search_tkns,
                                    kb_search=sample, ret=sample_ret))

            norm_meta['#QUERY'] = kb_meta

            # a dictionary slot_id -> [1 nlg example, dialog act, sys/usr]
            for slot, examples in nlg_spec.items():
                slot = "#{}".format(slot)
                is_usr = slot in usr_slots
                vocab = usr_slots[slot] if is_usr else sys_slots[slot]
                slot_meta = []

                for intent, utts in examples.items():
                    if type(utts) is list:
                        for u in utts:
                            if intent == 'inform':
                                speaker = USR if is_usr else SYS
                                template_utt = [BOS, speaker] + self.tokenize(u) + [EOS]
                                # cap the example number up to 10
                                examples = [[BOS, speaker]+self.tokenize(u % word)+[EOS]
                                            for word in vocab[0:10]]
                                slot_meta.append(Pack(slot=slot, intent=intent,
                                                      utt=template_utt, is_usr=is_usr,
                                                      examples=examples))
                            else:
                                speaker = SYS if is_usr else USR
                                template_utt = [BOS, speaker] + self.tokenize(u) + [EOS]
                                slot_meta.append(Pack(slot=slot, intent=intent, is_usr=is_usr,
                                                      utt=template_utt))
                    elif type(utts) is dict:
                        # we cap to at most 10 YN questions
                        for expect_answer, qs in utts.items()[0:5]:
                            for q in qs:
                                template_utt = [BOS, USR, expect_answer] + self.tokenize(q) + [EOS]
                                slot_meta.append(Pack(slot=slot, intent=intent, is_usr=is_usr,
                                                      expected=expect_answer, utt=template_utt))
                    else:
                        raise ValueError("Unknown meta type")

                norm_meta[slot] = slot_meta
            all_norm_meta[domain_name] = norm_meta
        # END OF reading
        self.logger.info("Read {} domain metas".format(len(all_norm_meta)))
        return all_norm_meta

    def _process_dialog(self, data):
        """
        For the return dialog corpus, each uttearnce is is represented by:
        (speaker, conf, utterance)
        
        :param data: a list of list of utterances 
        :return: a dialog coprus. 
        """
        dialogs = []
        all_length = []
        all_dialog_len = []
        for raw_dialog in data:
            norm_dialog = [Pack(speaker=USR, conf=1.0, utt= [BOS, raw_dialog[0]['domain'], BOD, EOS])]
            for l in raw_dialog:
                msg = Pack.msg_from_dict(l, self.tokenize,
                                         {"SYS": SYS, "USR": USR},
                                         BOS, EOS, include_domain=self.include_domain)
                norm_dialog.append(msg)
                all_length.append(len(msg.utt))
            dialogs.append(norm_dialog)
            all_dialog_len.append(len(norm_dialog))

        max_len = np.max(all_length)
        mean_len = float(np.average(all_length))
        coverage = len(np.where(np.array(all_length) < self.max_utt_len)[0]) / float(len(all_length))
        self.logger.info("Max utt len %d, mean utt len %.2f, %d covers %.2f" %
                         (max_len, mean_len, self.max_utt_len, coverage))
        self.logger.info("Max dialog len %d, mean dialog len %.2f" %
                         (np.max(all_dialog_len), np.average(all_dialog_len)))
        return dialogs

    def _build_vocab(self, max_vocab_cnt, speaker=None):
        all_words = []
        for dialogs in self.corpus:
            for msg in dialogs:
                if speaker is None or speaker == msg.speaker:
                    all_words.extend(msg.utt)
                    if 'actions' in msg:
                        for act in msg.actions:
                            all_words.extend(self._act_to_str(msg.speaker, act))

        all_test_words = []
        for dialogs in self.test_corpus:
            for msg in dialogs:
                if speaker is None or speaker == msg.speaker:
                    all_test_words.extend(msg.utt)
                    if 'actions' in msg:
                        for act in msg.actions:
                            temp = self._act_to_str(msg.speaker, act)
                            all_words.extend(temp)

        for key, domain in self.domain_meta.items():
            all_words.append(key)
            all_words.extend(domain.usr_id2slot)
            all_words.extend(domain.sys_id2slot)
            for slot, slot_meta in domain.items():
                if "#" not in slot:
                    continue
                for example in slot_meta:
                    all_words.extend(example.utt)

        vocab_count = Counter(all_words+all_test_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        speaker_str = "both" if speaker is None else speaker
        self.logger.info("Raw vocab %d, vocab size %d, cut_off %d, train UNK rate %.4f For %s speaker"
                         % (raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                            float(discard_wc) / len(all_words), speaker_str))

        vocab = [PAD, UNK, SEP, REQ, INF] + [t for t, cnt in vocab_count]
        rev_vocab = missingdict(lambda: vocab.index(UNK))
        for idx, word in enumerate(vocab):
            rev_vocab[word] = idx

        test_vocab_count = Counter(all_test_words).most_common()
        test_unk_cnt = np.sum([c for t, c in test_vocab_count if rev_vocab[t] == vocab.index(UNK)])
        unk_ratio = float(test_unk_cnt) / len(all_test_words)
        self.logger.info("Test vocabulary UNK rate %.4f" % (unk_ratio))

        return vocab, rev_vocab

    def _act_to_str(self, speaker, act):
        paras = act['parameters']
        intent = act['act']
        str_paras = []
        if speaker == SYS and intent == 'inform':
            for k, v in paras[0].items():
                str_paras.extend([k, v])
        elif speaker == SYS and intent == 'query':
            for k, v in paras[0].items():
                str_paras.extend([k, v])
        elif intent == 'kb_return':
            para_id = 0 if speaker == SYS else 1
            for k in paras[para_id].keys():
                str_paras.append(k)
        else:
            for k, v in paras:
                if k and v:
                    str_paras.extend([k, v])
                elif v:
                    str_paras.append(v)
                elif k:
                    str_paras.append(k)
        str_paras = map(str, str_paras)
        return [intent] + str_paras

    def _to_id_corpus(self, name, data, use_black_list):
        results = []
        kick_cnt = 0
        for dialog in data:
            temp = []
            is_kicked = False
            should_filter = np.random.rand() < self.black_ratio
            # convert utterance and feature into numeric numbers
            for msg in dialog:
                domain = msg.get("domain")
                if use_black_list and self.black_domains \
                        and domain in self.black_domains \
                        and should_filter:
                    is_kicked = True
                    break

                copy_msg = msg.copy()
                copy_msg['utt'] = [self.rev_vocab[t] for t in msg.utt]

                # make state become Ids
                if self.include_state and 'state' in copy_msg.keys()\
                        and copy_msg.state is not None:
                    id_state = []
                    state = copy_msg.state
                    sys_goals = state['sys_goals']
                    usr_slots = state['usr_slots']

                    for slot in usr_slots:
                        # name, expected, value, max_val
                        one_hot_s = self._get_id_slot(slot, domain)
                        # is is_usr, delivered, conf, max_conf, just_kb
                        float_s = [1.0, 0.0, 0.0, slot['max_conf'], 0.0]
                        id_state.append(Pack(cat=one_hot_s, real=float_s))

                    for goal in sys_goals:
                        # name, expected, value, max_val
                        one_hot_s = self._get_id_slot(goal, domain)
                        # is is_usr, delivered, conf, max_conf, just_kb
                        float_s = [0.0, float(goal['delivered']), goal['conf'], 0.0, float(state['kb_update'])]
                        id_state.append(Pack(cat=one_hot_s, real=float_s))

                    copy_msg['state'] = id_state

                # make action become Ids
                if 'actions' in copy_msg.keys():
                    str_acts = []
                    for act in msg.actions:
                        tokens = self._act_to_str(msg.speaker, act)
                        str_acts.append(' '.join(tokens))

                    str_acts = self.tokenize("|".join(str_acts))

                    if self.include_domain and False:
                        str_acts = [msg.speaker, domain] + str_acts
                    else:
                        str_acts = [msg.speaker] + str_acts

                    tkn_acts = [self.rev_vocab[t] for t in str_acts]

                    copy_msg['actions'] = tkn_acts

                temp.append(copy_msg)

            if not is_kicked:
                results.append(temp)
            else:
                kick_cnt += 1
        self.logger.info("Filter {} samples from {}".format(kick_cnt, name))

        return results

    def _get_id_slot(self, slot, domain):
        def tokenize(t):
            if t is None:
                return [UNK]
            else:
                return self.tokenize(t)

        domain_meta = self.domain_meta[domain]
        domain = tokenize(domain)
        name = tokenize(slot.get('name'))
        expected = tokenize(slot.get('expected'))
        value = tokenize(slot.get('value'))
        max_val = tokenize(slot.get('max_val'))
        cat = name + [SEP] + expected + [SEP] + value + [SEP] + max_val + [SEP]
        if self.include_domain:
            cat = domain + [SEP] + cat

        if self.include_example:
            slot_key = slot.get('name')
            is_usr = slot_key in domain_meta.usr_id2slot
            if is_usr:
                # find request
                all_requests = [msg.utt for msg in domain_meta[slot_key] if msg.intent == 'request']
                if all_requests:
                    cat += ['request'] + all_requests[0] + [SEP]
            else:
                # find inform
                all_informs = [msg.utt for msg in domain_meta[slot_key] if msg.intent == 'inform']
                if all_informs:
                    cat += ['inform'] + all_informs[0] + [SEP]

        id_cat = [self.rev_vocab[t] for t in cat]
        return id_cat

    def get_dialog_corpus(self):
        """
        Convert utt into word IDs and return the corpus in dictionary.
        :return: {train: [d1, d2, ...], valid: [d1, d2, ...], test: [d1, d2, ...]} 
        """
        # get equal amount of valid data from each domains
        id2domains = defaultdict(list)
        for d_id, d in enumerate(self.corpus):
            id2domains[d[1].domain].append(d_id)
        domain_valid_size = int(min(1000, int(len(self.corpus) * 0.1))/len(id2domains))
        train_ids, valid_ids = [], []
        for ids in id2domains.values():
            train_ids.extend(ids[domain_valid_size:])
            valid_ids.extend(ids[0:domain_valid_size])

        self.logger.info("Loaded Corpus with train %d, valid %d, test %d"
                         % (len(train_ids), len(valid_ids), len(self.test_corpus)))

        train_corpus = [self.corpus[i] for i in train_ids]
        valid_corpus = [self.corpus[i] for i in valid_ids]

        id_train = self._to_id_corpus('train', train_corpus, use_black_list=True)
        id_valid = self._to_id_corpus('valid', valid_corpus, use_black_list=True)
        id_test = self._to_id_corpus('test', self.test_corpus, use_black_list=False)
        return Pack(train=id_train, valid=id_valid, test=id_test)

    def get_domain_meta(self):
        # get domain_meta
        id_domain_meta = Pack()
        id_domain_meta['sys_id'] = self.rev_vocab[SYS]
        id_domain_meta['usr_id'] = self.rev_vocab[USR]
        for domain, meta in self.domain_meta.items():
            description = [domain, SEP] + meta.usr_id2slot + [SEP] + meta.sys_id2slot
            description = [self.rev_vocab[t] for t in description]
            templates, acts = [], []
            templates.append([self.rev_vocab[t] for t in meta.greet])
            acts.append([self.rev_vocab[t] for t in [REQ, '0', 'greet', 'greet']])

            for s, value in meta.items():
                if '#' not in s:
                    continue
                for example in value:
                    if s == '#QUERY':
                        s_id = 0
                    else:
                        s_id = meta.usr_id2slot.index(s) if example.is_usr else meta.sys_id2slot.index(s)
                    type = INF if example.is_usr else REQ
                    acts.append([self.rev_vocab[t] for t in [type, str(s_id), example.intent, example.slot]])
                    templates.append([self.rev_vocab[t] for t in example.utt])

            id_domain_meta[domain] = Pack(description=description,
                                          templates=templates,
                                          acts=acts)
            self.logger.info("{} templates for {}".format(len(templates), domain))

        return id_domain_meta

    def get_seed_responses(self, utt_cnt=100, domains=None, speakers=None):
        if utt_cnt == 0 or self.config.action_match is False:
            return []

        # estimate how many dialogs we need.
        dialog_cnt = utt_cnt/10

        # find all domains IDs
        id2domains = defaultdict(list)
        for d_id, d in enumerate(self.corpus):
            id2domains[d[1].domain].append(d_id)

        black_sys_utts = []
        all_domains = []

        for domain, ids in id2domains.items():
            selected_ids = []

            if domains is None or domain in domains:
                selected_ids.extend(ids[0:dialog_cnt])

            data = [self.corpus[idx] for idx in selected_ids]
            id_selected_data = self._to_id_corpus('Extra', data, use_black_list=False)

            domain_responses = {}
            # prepare the container
            for dialog in id_selected_data:
                for msg in dialog:
                    if speakers is None or msg.speaker in speakers:
                        if 'actions' not in msg:
                            continue

                        if len(domain_responses) >= utt_cnt:
                            break
                        domain = msg.get('domain')
                        utt = " ".join(map(str, msg.utt))
                        if utt not in domain_responses:
                            domain_responses[utt] = (msg, domain)

                        #all_domains.append(domain)
                        #black_sys_utts.append(msg)
                        #domain_cnt +=1
                if len(domain_responses) > utt_cnt:
                    break

            for msg, domain in domain_responses.values():
                black_sys_utts.append(msg)
                all_domains.append(domain)


        self.logger.info("Collected {} extra samples".format(len(black_sys_utts)))
        self.logger.info(Counter(all_domains).most_common())
        return black_sys_utts

    def get_turn_corpus(self, speaker):
        """
        :return: all system utterances -> actions 
        """
        data = self.corpus + self.test_corpus
        utt2act = []
        for dialog in data:
            for msg in dialog:
                if msg.speaker == speaker:
                    utt2act.append(Pack(utt=msg.utt, actions=msg.actions, domain=msg.domain))
        return utt2act


class ZslStanfordCorpus(object):
    logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_tokenize()
        self.black_domains = config.black_domains
        self.black_ratio = config.black_ratio
        self.train_corpus = self._read_file(os.path.join(self._path, 'kvret_train_public.json'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'kvret_dev_public.json'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'kvret_test_public.json'))
        with open(os.path.join(self._path, 'kvret_entities.json'), 'rb') as f:
            self.ent_metas = json.load(f)
        self.domain_descriptions = self._read_domain_descriptions(self._path)
        self._build_vocab()
        print("Done loading corpus")

    def _read_domain_descriptions(self, path):
        # read all domains
        seed_responses = []
        speaker_map = {'assistant': SYS, 'driver': USR}

        def _read_file(domain):
            with open(os.path.join(path, 'domain_descriptions/{}.tsv'.format(domain)), 'rb') as f:
                lines = f.readlines()
                for l in lines[1:]:
                    tokens = l.split('\t')
                    if tokens[2] == "":
                        break
                    utt = tokens[1]
                    speaker = tokens[0]
                    action = tokens[3]
                    if self.config.include_domain:
                        utt = [BOS, speaker_map[speaker], domain] + self.tokenize(utt) + [EOS]
                        action = [BOS, speaker_map[speaker], domain] + self.tokenize(action) + [EOS]
                    else:
                        utt = [BOS, speaker_map[speaker]] + self.tokenize(utt) + [EOS]
                        action = [BOS, speaker_map[speaker]] + self.tokenize(action) + [EOS]

                    seed_responses.append(Pack(domain=domain, speaker=speaker,
                                               utt=utt, actions=action))

        _read_file('navigate')
        _read_file('schedule')
        _read_file('weather')
        return seed_responses

    def _read_file(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)

        return self._process_dialog(data)

    def _process_dialog(self, data):
        new_dialog = []
        all_lens = []
        all_dialog_lens = []
        speaker_map = {'assistant': SYS, 'driver': USR}
        for raw_dialog in data:
            domain = raw_dialog['scenario']['task']['intent']
            kb_items = []
            if raw_dialog['scenario']['kb']['items'] is not None:
                for item in raw_dialog['scenario']['kb']['items']:
                    kb_items.append([KB]+self.tokenize(" ".join(["{} {}".format(k, v) for k, v in item.items()])))

            dialog = [Pack(utt=[BOS, domain, BOD, EOS], speaker=USR, slots=None, domain=domain)]
            for turn in raw_dialog['dialogue']:
                utt = turn['data']['utterance']
                slots = turn['data'].get('slots')
                speaker = speaker_map[turn['turn']]
                if self.config.include_domain:
                    utt = [BOS, speaker, domain] + self.tokenize(utt) + [EOS]
                else:
                    utt = [BOS, speaker] + self.tokenize(utt) + [EOS]

                all_lens.append(len(utt))
                if speaker == SYS:
                    dialog.append(Pack(utt=utt, speaker=speaker, slots=slots, domain=domain, kb=kb_items))
                else:
                    dialog.append(Pack(utt=utt, speaker=speaker, slots=slots, domain=domain))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)
                #for item in turn.get('kb', []):
                #    all_words.extend(item)

        for resp in self.domain_descriptions:
            all_words.extend(resp.actions)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count])

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, name, data, use_black_list):
        results = []
        kick_cnt = 0
        domain_cnt = []
        for dialog in data:
            if len(dialog) < 1:
                continue
            domain = dialog[0].domain
            should_filter = np.random.rand() < self.black_ratio
            if use_black_list and self.black_domains \
                    and domain in self.black_domains \
                    and should_filter:
                kick_cnt += 1
                continue
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               domain=turn.domain,
                               domain_id=self.rev_vocab[domain])
                               #kb=[self._sent2id(item) for item in turn.get('kb', [])])
                temp.append(id_turn)

            results.append(temp)
            domain_cnt.append(domain)
        self.logger.info("Filter {} samples from {}".format(kick_cnt, name))
        self.logger.info(Counter(domain_cnt).most_common())
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus("Train", self.train_corpus, use_black_list=True)
        id_valid = self._to_id_corpus("Valid", self.valid_corpus, use_black_list=False)
        id_test = self._to_id_corpus("Test", self.test_corpus, use_black_list=False)
        return Pack(train=id_train, valid=id_valid, test=id_test)

    def get_seed_responses(self, utt_cnt=100):
        domain_seeds = defaultdict(list)
        all_domains = []
        if utt_cnt == 0 or self.config.action_match is False:
            return []

        for resp in self.domain_descriptions:
            resp_copy = resp.copy()
            resp_copy['utt'] = self._sent2id(resp.utt)
            resp_copy['actions'] = self._sent2id(resp.actions)
            resp_copy['domain_id'] = self.rev_vocab[resp.domain]
            if len(domain_seeds[resp.domain]) >= utt_cnt:
                continue

            domain_seeds[resp.domain].append(resp_copy)
            all_domains.append(resp.domain)

        seed_responses = []
        for v in domain_seeds.values():
            seed_responses.extend(v)

        self.logger.info("Collected {} extra samples".format(len(seed_responses)))
        self.logger.info(Counter(all_domains).most_common())
        return seed_responses

