#!/usr/bin/env python
import glob
import re
__author__ = 'arenduchintala'
def scan_file(fs):
    min_loss = (1000000, None, None) 
    max_p_r = (0, None, None)
    max_p_u = (0, None, None)
    max_p_c = (0, None, None)
    max_p_ic = (0, None, None)
    for f in glob.glob(fs):
        content = open(f, 'r').readlines()
        content = [c for c in content if c.startswith('ave')]
        dev_content = [c for idx,c in enumerate(content) if idx % 2 ==0]
        #train_content = [c for idx,c in enumerate(content) if idx % 2 ==0]
        for idx,dc in enumerate(dev_content):
            items = re.split(r'(\s+|:)', dc)
            items = [i for i in items if i.strip() != '' and i.strip() !=':']
            loss = float(items[3])
            if loss < min_loss[0]:
                min_loss = (loss, idx, f)
            p_r = float(items[5])
            if p_r > max_p_r[0]:
                max_p_r = (p_r, idx, f)
            p_u = float(items[7])
            if p_u > max_p_u[0]:
                max_p_u = (p_u, idx, f)
            p_c = float(items[9])
            if p_c > max_p_c[0]:
                max_p_c = (p_c, idx, f)
            p_ic = float(items[11])
            if p_ic > max_p_ic[0]:
                max_p_ic = (p_ic, idx, f)
    return min_loss, max_p_r, max_p_u, max_p_c, max_p_ic


if __name__ == '__main__':
    fs = "./simple.scalar.*.log"
    min_loss, max_p_r, max_p_u, max_p_c, max_p_ic = scan_file(fs)
    print 'best scalar'
    print 'min_loss:', min_loss
    print 'max_p_r:', max_p_r 
    print 'max_p_u:', max_p_u
    print 'max_p_c:', max_p_c
    print 'max_p_ic:', max_p_ic 
    fs = "./simple.adapt.*.log"
    min_loss, max_p_r, max_p_u, max_p_c, max_p_ic = scan_file(fs)
    print 'best adapt'
    print 'min_loss:', min_loss
    print 'max_p_r:', max_p_r 
    print 'max_p_u:', max_p_u
    print 'max_p_c:', max_p_c
    print 'max_p_ic:', max_p_ic 
