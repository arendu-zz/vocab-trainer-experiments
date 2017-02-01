#!/usr/bin/env python
import sys
import codecs
import json
from ed import edsimple as ED
__author__ = 'arenduchintala'
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def get_nearest(incomplete_answer, y_sigma):
    dist = {}
    for y in y_sigma:
        d = ED(incomplete_answer, y)
        l = dist.get(d, [])
        l.append(y)
        dist[d] = l
    return dist


if __name__ == '__main__':
    user_records = codecs.open('fixed_vocab_training_user_records.csv', 'r', 'utf8').readlines()
    user_table = codecs.open('vocab_training_user_table.csv', 'r', 'utf8').readlines()
    content = codecs.open('fake-en-medium.vocab', 'r', 'utf8').readlines()
    x_sigma = set([])
    y_sigma = set([])
    for line in content[1:]:
        items = line.strip().split(',')
        fr,en = zip(*[tuple(i.lower().strip().split('/')) for i in items])
        x_sigma.update(list(fr))
        y_sigma.update(list(en))

    completed_users = {}
    for line in user_table[1:]:
        items = line.strip().split('\t')
        if int(items[2]) == 35 and items[3] != 'NULL':
            completed_users[items[1].strip()] = (int(items[0]), int(items[2]), items[3])

    user2no_answer_count = {}
    for line in user_records[1:]:
        items = line.split('\t')
        user = items[0].strip()
        if user in completed_users:
            if items[5].strip() == "" or items[5].strip() == "NO_ANSWER_MADE" or items[6].strip() == "" or items[6].strip() == "{}":
                count = user2no_answer_count.get(user, 0)
                user2no_answer_count[user] = count + 1
            elif items[5].strip().lower() in y_sigma:
                pass
            else:
                pass
        else:
            pass
    for k,v in user2no_answer_count.iteritems():
        print k, v, completed_users[k]

    good ={}
    bad ={}
    good_users = codecs.open('good.users', 'w', 'utf8')

    for u in completed_users.iterkeys():
        if user2no_answer_count.get(u,0) > 1: 
            bad[u] = user2no_answer_count.get(u, 0) 
        else:
            good[u] = 0
    print '***************good**************'
    for k in good:
        print k
        good_users.write(k + '\n')
    print len(good), 'good users'
    print '***************bad***************'
    for v,k in sorted([(v,k) for k,v in bad.iteritems()]):
        test_result = json.loads(completed_users[k][-1])
        print k,v, test_result[u'test_correct_num']
    print len(bad), 'bad users'
    good_users.flush()
    good_users.close()
