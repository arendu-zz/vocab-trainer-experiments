#!/usr/bin/env python
import sys
import codecs
from ed import edsimple as ED
__author__ = 'arenduchintala'
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def get_nearest(incomplete_answer, y_sigma):
    dist = {}
    min_d = 10000
    if incomplete_answer in ["watch", "enter", "accept"]:
        return ["to " + incomplete_answer], 0
    else:
        for y in y_sigma:
            d = ED(incomplete_answer.lower(), y)[0]
            l = dist.get(d, [])
            l.append(y)
            dist[d] = l
            min_d = d if d < min_d else min_d
        return dist[min_d], min_d


if __name__ == '__main__':
    user_records = codecs.open('./content/vocab_training_user_records.csv', 'r', 'utf8').readlines()
    fixed_user_records = codecs.open('./content/fixed_vocab_training_user_records.csv', 'w', 'utf8')
    fixed_user_records.write(user_records[0])
    user_table = codecs.open('./content/vocab_training_user_table.csv', 'r', 'utf8').readlines()
    content = codecs.open('./content/fake-en-medium.vocab', 'r', 'utf8').readlines()
    x_sigma = set([])
    y_sigma = set([])
    y_sigma_lower = set([])
    user2unfixable_count = {}
    for line in content[1:]:
        items = line.strip().split(',')
        fr,en = zip(*[tuple(i.strip().split('/')) for i in items])
        x_sigma.update(list(fr))
        y_sigma.update(list(en))

    for line in user_records[1:]:
        items = line.split('\t')
        user = items[0].strip()
        current_obs = items[5].strip()
        if current_obs.lower() == "NO_ANSWER_MADE".lower() or current_obs.strip() == "":
            pass
        elif current_obs not in y_sigma:
            list_nearest_current_obs, near_dist = get_nearest(current_obs, y_sigma)
            if len(list_nearest_current_obs) == 1 and near_dist < 4:
                nearest_current_obs = list_nearest_current_obs[0]
                print current_obs, nearest_current_obs
                items[5] = nearest_current_obs
            else:
                print current_obs, 'UNFIXABLE', user
                uc = user2unfixable_count.get(user, 0)
                user2unfixable_count[user] = uc + 1
                items[5] = "NO_ANSWER_MADE"
        else:
            pass
        new_line = '\t'.join(items)
        fixed_user_records.write(new_line + '\n')
    fixed_user_records.flush()
    fixed_user_records.close()
    print "unfixable users"
    for u,uf in user2unfixable_count.iteritems():
        print u, uf
