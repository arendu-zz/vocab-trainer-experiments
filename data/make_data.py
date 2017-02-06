#!/usr/bin/env python
import sys
import pdb
import numpy as np
import codecs
import json
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
__author__ = 'arenduchintala'
global user2data_lines, en2fr, user2test_q
user2data_lines = {}
user2test_data_lines = {}
en2fr = {}
user2test_q = {}

def add_test_data_line(u, dl, ts):
    global user2test_data_lines
    dls = user2test_data_lines.get(u, [])
    dls.append((ts, dl))
    user2test_data_lines[u] = dls
    return True

def add_data_line(u, dl, ts):
    global user2data_lines
    dls = user2data_lines.get(u, [])
    dls.append((ts, dl))
    user2data_lines[u] = dls
    #print 'added dl', dl
    return True

def fix_complex_obs(co):
    fixed_complex_obs = {}
    if co.strip() == "" or co.strip() == "{}":
        return {"1": "NO_ANSWER_MADE"}
    else:
        json_complex_obs = json.loads(co)
        vals = set([])
        for k,v in json_complex_obs.iteritems():
            if v[u'guess'] not in vals:
                vals.add(v[u'guess'])
                fixed_complex_obs[k] = v[u'guess']
        return fixed_complex_obs

def load_user2test():
    global user2test_data_lines, en2fr
    for line in codecs.open("./content/fixed_vocab_training_user_table.csv","r", "utf8").readlines()[1:]:
        items = line.strip().split('\t')
        user = items[1].strip()
        if items[3].strip() != "" and int(items[2]) == 35 and items[3].strip() != 'NULL':
            test_result = json.loads(items[3])
        t_step = 36.0
        for k,v in test_result.iteritems():
            if k == u'test_correct_num' or k == u'test_total_num':
                pass
            else:
                en_true = str(v[u'reference'])
                fr_str = en2fr[en_true]
                en_selected = v[u'user_answer']
                if en_selected != "NO_ANSWER_MADE":
                    test_data_line = '\t'.join([user, str(test_result[u'test_correct_num']), "TP", str(t_step), 'XX', fr_str, en_true, 'ALL', en_selected, "nofeedback"])
                    add_test_data_line(user, test_data_line, t_step)
                    t_step += 0.1
                else:
                    pass
    return True


if __name__ == '__main__':
    content = codecs.open('./content/fake-en-medium.vocab', 'r', 'utf8').readlines()
    user2test_q = {}
    en2fr = {}
    for line in content[1:]:
        items = line.strip().split(',')
        fr,en = zip(*[tuple(i.strip().split('/')) for i in items])
        for f,e, in zip(fr, en):
            en2fr[e] = f
    good_users = {}
    good_users_test_qa = {}
    for i in codecs.open('new-good.users', 'r', 'utf8').readlines():
        good_users[i.strip()] = None
    
    for line in codecs.open('./content/vocab_training_user_table.csv', 'r', 'utf8').readlines()[1:]:
        if line.strip() != "":
            items = line.split('\t')
            if items[3].strip() != "" and items[3].strip() != "NULL":
                test_result = json.loads(items[3])
            user = items[1].strip()
            if user in good_users:
                good_users[user] = test_result[u'test_correct_num']
    '''
    list_s= []
    test_score_hist = {}
    for g,s in good_users.iteritems():
        print g, s
        list_s.append(int(s))
        t = test_score_hist.get(int(s), 0)
        test_score_hist[int(s)] = t + 1
        assert s is not None
    ls = np.array(list_s)
    print 'mean', np.mean(ls)
    print 'sd', np.std(ls)
    print 'max', np.max(ls)
    print 'score hist'
    sum_v = 0
    for k,v in sorted(test_score_hist.iteritems()):
        print k, v
        sum_v += v
    print sum_v, 'total scores'
    '''
    fixed_records = codecs.open('./content/fixed_vocab_training_user_records.csv', 'r', 'utf8').readlines()
    c = 0
    bad_c = 0
    #0:username  1:training_step 2:prompt_type 3:current_action  4:current_action_id 5:current_observation 6:complex_observation
    user2data_lines = {}
    for line in fixed_records[1:]:
        items= [i.strip() for i in line.split('\t')]
        if items[0] in good_users:
            user = items[0].strip()
            training_step = float(items[1].strip())
            prompt_type = items[2].strip()
            user_test_score = str(good_users[user])
            current_action = json.loads(items[3])
            fr_str = current_action[2].strip()
            current_action_id = items[4].strip()
            current_obs = items[5].strip()
            complex_obs = items[6].strip()

            if current_obs.strip() == "":
                current_obs = "NO_ANSWER_MADE"

            if current_obs == "NO_ANSWER_MADE": 
                #print 'good user bad answer', items
                bad_c += 1
            if prompt_type == "MCR" and (complex_obs.strip() == "{}" or complex_obs.strip() == ""):
                #print 'good user bad answer', items
                bad_c += 1
            c += 1

            if prompt_type == "EX":
                en_selected = current_action[3].strip()
                en_true = en_selected
                en_options = "ALL"
                data_line = '\t'.join([user, user_test_score, prompt_type, str(training_step), current_action_id, fr_str, en_true, en_options, en_selected, "revealed"])
                add_data_line(user, data_line, training_step)
            elif prompt_type == "TP":
                #will only give indicative feedback i.e. does not show correct answer
                en_selected = current_obs
                en_true = current_action[3].strip()
                en_options = "ALL"
                feedback = en_selected == en_true
                fb_str = "correct" if feedback else "incorrect"
                data_line = '\t'.join([user, user_test_score, prompt_type, str(training_step), current_action_id, fr_str, en_true, en_options, en_selected, fb_str])
                add_data_line(user, data_line, training_step)
            elif prompt_type == "TPR":
                #will indicate and give correct answer
                #2 rows.. 
                en_selected = current_obs
                en_true = current_action[3].strip()
                en_options = "ALL"
                feedback = en_selected == en_true
                fb_str = "correct" if feedback else "incorrect"
                if feedback:
                    #if correct stop with one data line
                    data_line = '\t'.join([user, user_test_score, prompt_type, str(training_step), current_action_id, fr_str, en_true, en_options, en_selected, fb_str])
                    add_data_line(user, data_line, training_step)
                else:
                    #if wrong answer then 2 data lines
                    data_line = '\t'.join([user, user_test_score, prompt_type, str(training_step), current_action_id, fr_str, en_true, en_options, en_selected, fb_str])
                    add_data_line(user, data_line, training_step)
                    data_line = '\t'.join([user, user_test_score, prompt_type, str(training_step), current_action_id, fr_str, en_true, en_options, en_true, "revealed"])
                    add_data_line(user, data_line, training_step + 0.1)
                    pass
                pass
            elif prompt_type == "MC":
                #will indicate ONLY
                en_options = ','.join(current_action[3:])
                en_selected = current_obs
                en_true = current_action[3].strip()
                feedback = en_true == en_selected
                fb_str = "correct" if feedback else "incorrect"
                data_line = '\t'.join([user, user_test_score, prompt_type, str(training_step), current_action_id, fr_str, en_true, en_options, en_selected, fb_str])
                add_data_line(user, data_line, training_step)
                pass
            elif prompt_type == "MCR":
                # will let you retry but no correct answer.. i.e. lots of indicative feedback only...
                # as many rows as guesses..
                all_en_options = set(current_action[3:])
                fixed_complex_obs = fix_complex_obs(complex_obs)
                en_true = current_action[3].strip()
                #print items
                point = 0 
                prev_en_selected = None
                for k,v in sorted(fixed_complex_obs.iteritems()): 
                    en_selected = v 
                    if en_selected != prev_en_selected:
                        feedback = en_true == en_selected
                        fb_str = "correct" if feedback else "incorrect"
                        en_options = ','.join(all_en_options)
                        t_step = training_step + (point * 0.1)
                        data_line = '\t'.join([user, user_test_score, prompt_type, str(t_step), current_action_id, fr_str, en_true, en_options, en_selected, fb_str])
                        add_data_line(user, data_line, t_step)
                        #print data_line
                        if en_selected == "NO_ANSWER_MADE" or en_selected == "":
                            pass
                        else:
                            #print 'trying to remove', en_selected
                            all_en_options.remove(en_selected)
                        point += 1
                        prev_en_selected = en_selected
                    else:
                        #print "this is weird", items
                        pdb.set_trace()
                pass
            else:
                raise Exception("unknown prompt type")
        else:
            pass

    good_users_list = [k for k in good_users]
    user_groups = []
    user_groups_stats = []
    
    for g in xrange(7):
        if len(good_users_list) > 35:
            ug = np.random.choice(good_users_list, 20, False)
        else:
            ug = [_ug for _ug in good_users_list] #remaining users...
        ug_ts= []
        for _ug in ug:
            ug_ts.append(good_users[_ug])
            good_users_list.remove(_ug)
        user_groups.append(ug)
        m = round(np.mean(ug_ts),2)
        sd = round(np.std(ug_ts), 2)
        user_groups_stats.append((m,sd))

    load_user2test() 

    for idx, (ug, ug_stats) in enumerate(zip(user_groups, user_groups_stats)):
        w = codecs.open('./data_splits/group' + str(idx) + '.data', 'w', 'utf8')
        print 'group:', idx, ',test_mean:', str(ug_stats[0]),',test_std:', str(ug_stats[1]),',num users:', len(ug)
        for u in ug:
            for ts, dll in sorted(user2data_lines[u]):
                w.write(dll + '\n')
            if u in user2test_data_lines:
                for ts, dll in sorted(user2test_data_lines[u]):
                    w.write(dll + '\n')
            else:
                print u,'not in user test'
        w.flush()
        w.close()

    #print bad_c, c
