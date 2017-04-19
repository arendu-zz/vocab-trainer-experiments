#!/usr/bin/env python
import codecs
import argparse
#from ed import edratio
__author__ = 'arenduchintala'

def tup2str(t):
    return '__'.join(t)

def make_unique(feats,values):
    fv_set = {}
    for f, v in zip(feats, values):
        fv_set_vals = fv_set.get(f, [])
        fv_set_vals.append(v)
        fv_set[f] = fv_set_vals
    unique_feats = []
    unique_values = []
    for f in sorted(fv_set.keys()):
        v_list = fv_set[f]
        unique_feats.append(f)
        unique_values.append(sum(v_list) / float(len(v_list)))
    assert len(set(unique_feats)) == len(unique_feats)
    return unique_feats, unique_values


def get_nonbinary_feats(s1, s2):
    f = []
    v = []
    best = get_best_word_editdist(s1,s2)
    phrase = get_phrase_editdist(s1,s2)
    f += best[0]
    f += phrase[0]
    v += best[1]
    v += phrase[1]
    return [tup2str(i) for i in f], v


def get_phrase_editdist(s1, s2):
    raise Exception("unsupported")
    #f =[('PHRASE_EDIT_DIST',), ('PHRASE_EDIT_DIST', '_'.join(s1), '_'.join(s2))]
    #chars1 = ' '.join(s1)
    #chars2 = ' '.join(s2)
    #e_phr = edratio(chars1, chars2)
    #return f, [e_phr] * len(f)


def get_best_word_editdist(s1, s2):
    raise Exception("unsupported")
    #f = [('BEST_WORD_EDIT_DIST',), ('BEST_WORD_EDIT_DIST', '_'.join(s1), '_'.join(s2))]
    #e = 0.0
    #for w1 in s1:
    #    for w2 in s2:
    #        er = edratio(w1, w2)
    #        if er > e:
    #            e = er
    #return f, [e] * len(f)


def get_suffix(w1, w2, s2_cat=[], fine_grained=True):
    f = []
    max_l = max(len(w1), len(w2))
    lim_l = max_l if max_l < 4 else 4
    for l in range(1, lim_l + 1): #[1,2,3,4]:
        if fine_grained:
            n = 'SUF' + str(l) + '_TO_' + 'SUF' +str(l)
            f += [(n,w1[:-l], w2[:-l])]
        nc = 'SUF' + str(l) + '_TO_CAT'
        f += [(nc,w1[:-l], cat) for cat in s2_cat]
        #if w1[:-l] == w2[:-l]:
        #    f += [(n + '_SAME',)]
    return f

def get_prefix(w1, w2, s2_cat=[], fine_grained= True):
    f = []
    max_l = max(len(w1), len(w2))
    lim_l = max_l if max_l < 4 else 4
    for l in range(1, lim_l + 1): #[1,2,3,4]:
        if fine_grained:
            n = 'PREF' + str(l) + '_TO_' + 'PREF' +str(l)
            f += [(n,w1[:l], w2[:l])]
        nc = 'PREF' + str(l) + '_TO_CAT'
        f += [(nc,w1[:l], cat) for cat in s2_cat]
        #if w1[:l] == w2[:l]:
        #    f += [(n + '_SAME',)]
    return f

def get_word(w1, w2, s2_cat=[]):
    f = []
    f += [('WORD_TO_WORD',w1,w2)]
    f += [('WORD_TO_CAT', w1, cat) for cat in s2_cat]
    return f

def get_lcstring(w1, w2):
    lc = 0
    table = {}
    lcs = {}
    max_lcs = ''
    for i,c1 in enumerate(w1):
        for j, c2 in enumerate(w2):
            if c1 == c2:
                table[i,j] = table.get((i - 1, j - 1), 0) + 1
                lcs[i,j] = lcs.get((i - 1, j - 1), '') + c1
                max_lcs = lcs[i,j] if max_lcs < lcs[i,j] else max_lcs
                lc = table[i,j] if lc < table[i,j] else lc
            else:
                pass
    return lc, max_lcs

def get_lcsubsequence(w1, w2):
    lc = 0
    table = {}
    lcs = {}
    max_lcs = ''
    for i,c1 in enumerate(w1):
        for j, c2 in enumerate(w2):
            if c1 == c2:
                table[i,j] = table.get((i - 1, j - 1), 0) + 1
                lcs[i,j] = lcs.get((i - 1, j - 1), '') + c1
                max_lcs = lcs[i,j] if max_lcs < lcs[i,j] else max_lcs
                lc = table[i,j] if lc < table[i,j] else lc
            else:
                table[i,j] = max(table.get((i - 1, j), 0), table.get((i, j - 1), 0))
                lcs[i,j] = max(lcs.get((i - 1, j), ''), lcs.get((i, j - 1), ''))
    return lc, max_lcs 

def get_phrase(s1, s2, s2_cat=[]):
    #f = [('PHRASE_TO_PHRASE',), ('PHRASE_TO_CAT',)]
    f = [] 
    f += [('PHRASE_TO_PHRASE', '_'.join(s1), '_'.join(s2))]
    for cat in s2_cat:
        f += [('PHRASE_TO_CAT', '_'.join(s1), cat)]
    return f

def get_lcs_feats(w1, w2):
    f = []
    l_str,s_str = get_lcstring(w1, w2)
    l_sub,s_sub = get_lcsubsequence(w1, w2)
    f += [('LCSTR_LEN_' + str(l_str),), ('LCSTR_' + str(s_str),)] if l_str > 0 else []
    f += [('LCSUB_LEN_' + str(l_sub),), ('LCSUB_' + str(s_sub),)] if l_sub > 0 else []
    return f

def get_binary_feats(s1, s2, s2_cat):
    f = [('BIAS',)]
    s2_cat = s2_cat if options.cat_feats else []
    f += get_phrase(s1, s2, s2_cat) if options.phrase_feats else []
    for w1_pos, w1 in enumerate(s1):
        for w2_pos, w2 in enumerate(s2):
            f += get_word(w1, w2, s2_cat) if options.word_feats else []
            f += get_prefix(w1, w2, s2_cat, options.fine_grained_feats) if options.prefix_feats else []
            f += get_suffix(w1, w2,s2_cat, options.fine_grained_feats) if options.suffix_feats else []
            f += get_lcs_feats(w1, w2) if options.lcs_feats else []
    v = [1.0] * len(f)
    return [tup2str(i) for i in f], v


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="accepts a csv file and generates log linear features")
    #insert options here
    opt.add_argument('-v', action='store', dest='vocab_file', default='example', required = True)
    opt.add_argument('-f', action='store', dest='feat_file', default='example', required = True)
    opt.add_argument('-p', action='store_true',dest='phrase_feats', default=True)
    opt.add_argument('-w', action='store_true',dest='word_feats', default=True)
    opt.add_argument('-l', action='store_true',dest='lcs_feats', default=False)
    opt.add_argument('--pre', action='store_true',dest='prefix_feats', default=True)
    opt.add_argument('--suf', action='store_true',dest='suffix_feats', default=True)
    opt.add_argument('-c', action='store_true',dest='cat_feats', default=False)
    opt.add_argument('--fine', action='store_true',dest='fine_grained_feats', default=True)
    options = opt.parse_args()
    data = codecs.open(options.vocab_file, 'r', 'utf8').readlines()
    data = [d.strip() for d in data]
    data = [d.split(',') for d in data]
    categories = data[0]
    en2categories = {}
    all_en = set([])
    all_es = set([])
    for line in data[1:]:
        for idx, phrase_pairs in enumerate(line):
            phrase_pairs = phrase_pairs.split('/')
            es_str = phrase_pairs[0].strip()
            es_phrase= phrase_pairs[0].strip().split()
            en_str = phrase_pairs[1].strip() 
            en_phrase = phrase_pairs[1].strip().split()
            en_categories = categories[idx].split()
            en2categories[en_str] = en_categories
            all_en.add(en_str)
            all_es.add(es_str)

    feat2val= {}
    featid2val = {}
    feat2id = {}
    id2feat = {}
    name_extn = '.'.join([i for i,o in [('p',options.phrase_feats), ('w', options.word_feats), ('l', options.lcs_feats), ('pre', options.prefix_feats),('suf', options.suffix_feats), ('c', options.cat_feats)] if o])

    w = codecs.open(options.feat_file + '.' + name_extn + '.event2feats', 'w', 'utf8')
    for es_str in all_es:
        for en_str in all_en:
            es_phrase= es_str.split()
            en_phrase = en_str.split()
            #fn, fv = get_binary_feats(es_phrase, en_phrase, en2categories[en_str])
            fn, fv = get_binary_feats(es_phrase, en_phrase, en_categories)
            fn, fv = make_unique(fn, fv)

            for f, val in zip(fn, fv):
                feat2id[f] = feat2id.get(f, len(feat2id))
                id2feat[feat2id[f]] = f
                feat2val[f] = feat2val.get(f, val)
                f_id = feat2id[f]
                featid2val[f_id] = featid2val.get(f_id, val)

            f2v_str = ' '.join([str(feat2id[f]) + ' ' + str(v) for f,v in zip(fn, fv)])
            w.write(es_str + '|||' + en_str + '|||' + f2v_str + '\n')
    w.flush()
    w.close()

    w = codecs.open(options.feat_file + '.' + name_extn + '.feat2id', 'w', 'utf8')
    for f_id in sorted(id2feat.keys()):
        w.write(str(f_id) + '\t' + id2feat[f_id] + '\n')
