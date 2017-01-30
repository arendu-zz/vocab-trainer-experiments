#!/usr/bin/env python
import sys
import codecs
import uuid
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
__author__ = 'arenduchintala'

if __name__ == '__main__':
    user_table = codecs.open('vocab_training_user_table.csv', 'r', 'utf8').readlines()
    user2uuid = {}
    anon_user_table = codecs.open('anon_vocab_training_user_table.csv', 'w', 'utf8')
    anon_user_table.write(user_table[0].strip() + '\n')
    for line in user_table[1:]:
        items = line.strip().split('\t')
        items[1] = items[1].strip()
        if items[1] in user2uuid:
            pass    
        else:
            user2uuid[items[1]] = str(uuid.uuid4())
        items[1]= user2uuid[items[1]]
        anon_items = '\t'.join(items)
        anon_user_table.write(anon_items + '\n')
    anon_user_table.flush()
    anon_user_table.close()
    user_records = codecs.open('vocab_training_user_records.csv', 'r', 'utf8').readlines()
    anon_user_records = codecs.open('anon_vocab_training_user_records.csv', 'w', 'utf8')
    for line in user_records[1:]:
        items = line.strip().split('\t')
        items[0] = user2uuid[items[0]]
        anon_items = '\t'.join(items)
        anon_user_records.write(anon_items + '\n')
    anon_user_records.flush() 
    anon_user_records.close()
