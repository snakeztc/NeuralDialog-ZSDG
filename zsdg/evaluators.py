# @Time    : 9/25/17 3:54 PM
# @Author  : Tiancheng Zhao
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from zsdg.utils import get_dekenize, get_tokenize
from scipy.stats import gmean
import logging
from zsdg.dataset.corpora import EOS, BOS
from collections import defaultdict


class EvaluatorBase(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp, domain='default'):
        raise NotImplementedError

    def get_report(self, include_error=False):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class TurnEvaluator(EvaluatorBase):
    """
    Use string matching to find the F-1 score of slots
    Use logistic regression to find F-1 score of acts
    Use string matching to find F-1 score of KB_SEARCH
    """
    CLF = "clf"
    REPRESENTATION = "rep"
    ID2TAG = "id2tag"
    TAG2ID = "tag2id"
    logger = logging.getLogger()

    def __init__(self, data_name, turn_corpus, domain_meta):
        self.data_name = data_name
        # train a dialog act classifier
        domain2ids = defaultdict(list)
        for d_id, d in enumerate(turn_corpus):
            domain2ids[d.domain].append(d_id)
        selected_ids = [v[0:1000] for v in domain2ids.values()]
        corpus = [turn_corpus[idx] for idxs in selected_ids for idx in idxs]

        self.model = self.get_intent_tagger(corpus)

        # get entity value vocabulary
        self.domain_id2ent = self.get_entity_dict_from_meta(domain_meta)

        # Initialize containers
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

    def get_entity_dict_from_meta(self, domain_meta):
        # get entity value vocabulary
        domain_id2ent = defaultdict(set)
        for domain, meta in domain_meta.items():
            domain_id2ent[domain].add("QUERY")
            domain_id2ent[domain].add("GOALS")
            for slot, vocab in meta.sys_slots.items():
                domain_id2ent[domain].add(slot)
                for v in vocab:
                    domain_id2ent[domain].add(v)

            for slot, vocab in meta.usr_slots.items():
                domain_id2ent[domain].add(slot)
                for v in vocab:
                    domain_id2ent[domain].add(v)

        domain_id2ent = {k: list(v) for k, v in domain_id2ent.items()}
        return domain_id2ent

    def get_entity_dict(self, turn_corpus):
        utt2act = {}
        for msg in turn_corpus:
            utt2act[" ".join(msg.utt[1:-1])] = msg

        dekenize = get_dekenize()
        utt2act = {dekenize(k.split()): v for k, v in utt2act.items()}
        self.logger.info("Compress utt2act from {}->{}".format(len(turn_corpus), len(utt2act)))

        # get entity value vocabulary
        domain_id2ent = defaultdict(set)
        for utt, msg in utt2act.items():
            for act in msg.actions:
                paras = act['parameters']
                intent = act['act']
                if intent == 'inform':
                    for v in paras[0].values():
                        domain_id2ent[msg.domain].add(str(v))
                elif intent == 'query':
                    for v in paras[0].values():
                        domain_id2ent[msg.domain].add(v)
                else:
                    for k, v in paras:
                        if v:
                            domain_id2ent[msg.domain].add(v)
        domain_id2ent = {k: list(v) for k, v in domain_id2ent.items()}
        return domain_id2ent

    def get_intent_tagger(self, corpus):
        """
        :return: train a dialog act tagger for system utterances 
        """
        self.logger.info("Train a new intent tagger")
        all_tags, utts, tags = [], [], []
        de_tknize = get_dekenize()
        for msg in corpus:
            utts.append(de_tknize(msg.utt[1:-1]))
            tags.append([a['act'] for a in msg.actions])
            all_tags.extend([a['act'] for a in msg.actions])

        most_common = Counter(all_tags).most_common()
        self.logger.info(most_common)
        tag_set = [t for t, c, in most_common]
        rev_tag_set = {t: i for i, t in enumerate(tag_set)}

        # create train and test set:
        data_size = len(corpus)
        train_size = int(data_size * 0.7)
        train_utts = utts[0:train_size]
        test_utts = utts[train_size:]

        # create y:
        sparse_y = np.zeros([data_size, len(tag_set)])
        for idx, utt_tags in enumerate(tags):
            for tag in utt_tags:
                sparse_y[idx, rev_tag_set[tag]] = 1
        train_y = sparse_y[0:train_size, :]
        test_y = sparse_y[train_size:, :]

        # train classifier
        representation = CountVectorizer(ngram_range=[1, 2]).fit(train_utts)
        train_x = representation.transform(train_utts)
        test_x = representation.transform(test_utts)

        clf = OneVsRestClassifier(SGDClassifier(loss='hinge', n_iter=10)).fit(train_x, train_y)
        pred_test_y = clf.predict(test_x)

        def print_report(score_name, scores, names):
            for s, n in zip(scores, names):
                self.logger.info("%s: %s -> %f" % (score_name, n, s))

        print_report('F1', metrics.f1_score(test_y, pred_test_y, average=None),
                     tag_set)

        x = representation.transform(utts)
        clf = OneVsRestClassifier(SGDClassifier(loss='hinge', n_iter=20)) \
            .fit(x, sparse_y)

        model_dump = {self.CLF: clf, self.REPRESENTATION: representation,
                      self.ID2TAG: tag_set,
                      self.TAG2ID: rev_tag_set}
        # pkl.dump(model_dump, open("{}.pkl".format(self.data_name), "wb"))
        return model_dump

    def pred_ents(self, sentence, tokenize, domain):
        pred_ents = []
        padded_hyp = "/{}/".format("/".join(tokenize(sentence)))
        for e in self.domain_id2ent[domain]:
            count = padded_hyp.count("/{}/".format(e))
            if domain =='movie' and e == 'I':
                continue
            pred_ents.extend([e] * count)
        return pred_ents

    def pred_acts(self, utts):
        test_x = self.model[self.REPRESENTATION].transform(utts)
        pred_test_y = self.model[self.CLF].predict(test_x)
        pred_tags = []
        for ys in pred_test_y:
            temp = []
            for i in range(len(ys)):
                if ys[i] == 1:
                    temp.append(self.model[self.ID2TAG][i])
            pred_tags.append(temp)
        return pred_tags

    """
    Public Functions
    """
    def initialize(self):
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

    def add_example(self, ref, hyp, domain='default'):
        self.domain_labels[domain].append(ref)
        self.domain_hyps[domain].append(hyp)

    def get_report(self, include_error=False):
        reports = []
        errors = []

        for domain, labels in self.domain_labels.items():
            intent2refs = defaultdict(list)
            intent2hyps = defaultdict(list)

            predictions = self.domain_hyps[domain]
            self.logger.info("Generate report for {} for {} samples".format(domain, len(predictions)))

            # find entity precision, recall and f1
            tp, fp, fn = 0.0, 0.0, 0.0

            # find intent precision recall f1
            itp, ifp, ifn = 0.0, 0.0, 0.0

            # backend accuracy
            btp, bfp, bfn = 0.0, 0.0, 0.0

            # BLEU score
            refs, hyps = [], []

            pred_intents = self.pred_acts(predictions)
            label_intents = self.pred_acts(labels)

            tokenize = get_tokenize()
            bad_predictions = []

            for label, hyp, label_ints, pred_ints in zip(labels, predictions, label_intents, pred_intents):
                refs.append([label.split()])
                hyps.append(hyp.split())

                label_ents = self.pred_ents(label, tokenize, domain)
                pred_ents = self.pred_ents(hyp, tokenize, domain)

                for intent in label_ints:
                    intent2refs[intent].append([label.split()])
                    intent2hyps[intent].append(hyp.split())

                # update the intent
                ttpp, ffpp, ffnn = self._get_tp_fp_fn(label_ints, pred_ints)
                itp += ttpp
                ifp += ffpp
                ifn += ffnn

                # entity or KB search
                ttpp, ffpp, ffnn = self._get_tp_fp_fn(label_ents, pred_ents)
                if ffpp > 0 or ffnn > 0:
                    bad_predictions.append((label, hyp))

                if "query" in label_ints:
                    btp += ttpp
                    bfp += ffpp
                    bfn += ffnn
                else:
                    tp += ttpp
                    fp += ffpp
                    fn += ffnn

            # compute corpus level scores
            bleu = bleu_score.corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
            ent_precision, ent_recall, ent_f1 = self._get_prec_recall(tp, fp, fn)
            int_precision, int_recall, int_f1 = self._get_prec_recall(itp, ifp, ifn)
            back_precision, back_recall, back_f1 = self._get_prec_recall(btp, bfp, bfn)

            # compute BLEU w.r.t intents
            intent_report = []
            for intent in intent2refs.keys():
                i_bleu = bleu_score.corpus_bleu(intent2refs[intent], intent2hyps[intent],
                                                smoothing_function=SmoothingFunction().method1)
                intent_report.append("{}: {}".format(intent, i_bleu))

            intent_report = "\n".join(intent_report)

            # create bad cases
            error = ''
            if include_error:
                error = '\nDomain {} errors\n'.format(domain)
                error += "\n".join(['True: {} ||| Pred: {}'.format(r, h)
                                    for r, h in bad_predictions])
            report = "\nDomain: %s\n" \
                     "Entity precision %f recall %f and f1 %f\n" \
                     "Intent precision %f recall %f and f1 %f\n" \
                     "KB precision %f recall %f and f1 %f\n" \
                     "BLEU %f BEAK %f\n\n%s\n" \
                     % (domain,
                        ent_precision, ent_recall, ent_f1,
                        int_precision, int_recall, int_f1,
                        back_precision, back_recall, back_f1,
                        bleu, gmean([ent_f1, int_f1, back_f1, bleu]),
                        intent_report)
            reports.append(report)
            errors.append(error)

        if include_error:
            return "\n==== REPORT===={error}\n========\n {report}".format(error="========".join(errors),
                                                                          report="========".join(reports))
        else:
            return "\n==== REPORT===={report}".format(report="========".join(reports))


class BleuEntEvaluator(EvaluatorBase):
    """
    Use string matching to find the F-1 score of slots
    Use logistic regression to find F-1 score of acts
    Use string matching to find F-1 score of KB_SEARCH
    """
    logger = logging.getLogger(__name__)

    def __init__(self, data_name, entity_metas):
        self.data_name = data_name
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

        # get entity value vocabulary
        self.domain_id2ent = self.get_entity_dict_from_meta(entity_metas)

    def get_entity_dict_from_meta(self, domain_meta):
        # get entity value vocabulary
        domain_id2ent = defaultdict(set)
        domain = None
        for slot, vocab in domain_meta.items():
            for value in vocab:
                if type(value) is dict:
                    for k, v in value.items():
                        domain_id2ent[domain].add(str(v).lower())
                else:
                    domain_id2ent[domain].add(str(value).lower())

        domain_id2ent = {k: list(v) for k, v in domain_id2ent.items()}
        return domain_id2ent

    def pred_ents(self, sentence, tokenize, domain):
        pred_ents = []
        sentence = sentence.lower()
        padded_hyp = "/{}/".format("/".join(tokenize(sentence)))
        for e in self.domain_id2ent[domain]:
            count = padded_hyp.count("/{}/".format(e))
            pred_ents.extend([e] * count)
        return pred_ents

    def initialize(self):
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

    def add_example(self, ref, hyp, domain='default'):
        self.domain_labels[domain].append(ref)
        self.domain_hyps[domain].append(hyp)

    def get_report(self, include_error=False):
        reports = []
        tokenize = get_tokenize()

        for domain, labels in self.domain_labels.items():
            predictions = self.domain_hyps[domain]
            self.logger.info("Generate report for {} for {} samples".format(domain, len(predictions)))
            refs, hyps = [], []

            # find entity precision, recall and f1
            tp, fp, fn = 0.0, 0.0, 0.0

            for label, hyp in zip(labels, predictions):
                label = label.replace(EOS, '').replace(BOS, '')
                hyp = hyp.replace(EOS, '').replace(BOS, '')
                ref_tokens = tokenize(label)[2:]
                hyp_tokens = tokenize(hyp)[2:]

                refs.append([ref_tokens])
                hyps.append(hyp_tokens)

                label_ents = self.pred_ents(label, tokenize, None)
                hyp_ents = self.pred_ents(hyp, tokenize, None)
                # hyp_ents = list(set(hyp_ents))

                ttpp, ffpp, ffnn = self._get_tp_fp_fn(label_ents, hyp_ents)
                tp += ttpp
                fp += ffpp
                fn += ffnn

            ent_precision, ent_recall, ent_f1 = self._get_prec_recall(tp, fp, fn)

            # compute corpus level scores
            bleu = bleu_score.corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
            report = "\nDomain: %s BLEU %f\n Entity precision %f recall %f and f1 %f\n" \
                     % (domain, bleu, ent_precision, ent_recall, ent_f1)
            reports.append(report)

        return "\n==== REPORT===={report}".format(report="========".join(reports))

