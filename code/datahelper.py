#!/usr/bin/env python
import numpy as np
import codecs
__author__ = 'arenduchintala'

class DataHelper(object):
    def __init__(self, event2feats_path, feat2id_path, actions_path):
        self.f2id = {}
        self.e2id = {}
        self.id2f = {}
        self.id2e = {}
        self.feat2id = {}
        self.id2feat = {}
        self.event2feats = {}
        self.all_e_ids = None
        self.examples = []
        self.actions = []
        self.action_vectors = []
        self.E_SIZE = 0
        self.F_SIZE = 0
        self.FEAT_SIZE = 0
        self.true_f2e = {}
        self.true_e2f = {}
        self.load_event2feats(event2feats_path)
        self.load_feats(feat2id_path)
        self.load_actions(actions_path)

    def load_actions(self, actions_path):
        self.actions = []
        self.examples = []
        for line in codecs.open(actions_path, 'r', 'utf8').readlines():
            items = [i.strip() for i in line.split('|||')]
            items[0] = int(items[0])
            self.actions.append(tuple(items))
            if len(items) == 4:
                self.examples.append((items[2], items[3]))
                self.true_f2e[items[2]] = items[3]
                self.true_e2f[items[3]] = items[2]
            else:
                pass
        return True

    def display_action(self, a_idx):
        return ' '.join([(i).encode('utf-8') for i in self.actions[a_idx][1:]])

    def load_feats(self, feat2id_path):
        for line in codecs.open(feat2id_path, 'r', 'utf8').readlines():
            items = line.split()
            f_id = int(items[0])
            f = items[1].strip()
            self.id2feat[f_id] = f
            self.feat2id[f] = f_id
        self.FEAT_SIZE = len(self.feat2id)
        return True

    def load_event2feats(self, event2feats_path):
        for line in codecs.open(event2feats_path, 'r', 'utf8').readlines():
            f, e, f2v = line.split('|||')
            f = f.strip()
            e = e.strip()
            self.f2id[f] = self.f2id.get(f, len(self.f2id))
            self.e2id[e] = self.e2id.get(e, len(self.e2id))
            f_id = self.f2id[f]
            e_id = self.e2id[e]
            self.id2f[f_id] = f
            self.id2e[self.e2id[e]] = e
            f2v = f2v.strip()
            feature_ids = [int(i) for idx, i in enumerate(f2v.split()) if idx % 2 == 0]  
            values = [float(i) for idx, i in enumerate(f2v.split()) if idx % 2 != 0]  
            self.event2feats[f_id, e_id] = (feature_ids, values)
        self.all_e_ids = np.array(range(len(self.e2id)))
        self.E_SIZE = len(self.e2id)
        self.F_SIZE = len(self.f2id)
        return True
