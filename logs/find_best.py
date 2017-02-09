#!/usr/bin/env python
import glob
import re
__author__ = 'arenduchintala'
"""
ave total loss:174.348 p_u:0.303,0.401 p_c:0.504,0.426 p_ic:0.135,0.285 p_ict:0.340,0.399 params:4.334,4.332--5.001,4.999--0.471,0.463--5.004,4.996
"""
def scan_file(fs):
    min_loss = (1000000, None, None) 
    max_p_u = (0, None, None)
    max_p_c = (0, None, None)
    max_p_ic = (0, None, None)
    min_p_ict = (1000000, None, None)
    max_p_c_ict = (0, None, None)
    for f in glob.glob(fs):
        content = open(f, 'r').readlines()
        content = [c for c in content if c.startswith('ave')]
        dev_content = [c for idx,c in enumerate(content) if idx % 2 ==0]
        #train_content = [c for idx,c in enumerate(content) if idx % 2 ==0]
        for idx,dc in enumerate(dev_content):
            items = re.split(r'(\s+|:)', dc)
            items = [i for i in items if i.strip() != '' and i.strip() !=':']
            items = [i.split(',')[0] for i in items]
            loss = float(items[3])
            if loss < min_loss[0]:
                min_loss = (loss, idx, f)
            p_u = float(items[5])
            if p_u > max_p_u[0]:
                max_p_u = (p_u, idx, f)
            p_c = float(items[7])
            if p_c > max_p_c[0]:
                max_p_c = (p_c, idx, f)
            p_ic = float(items[9])
            if p_ic > max_p_ic[0]:
                max_p_ic = (p_ic, idx, f)
            p_ict = float(items[11])
            if p_ict < min_p_ict[0]:
                min_p_ict = (p_ict, idx, f)
            p_c_ict = p_c - p_ict
            if p_c_ict > max_p_c_ict[0]:
                max_p_c_ict = (p_c_ict, idx, f)
    print 'min_loss:'.ljust(20), "%.4f" % min_loss[0], min_loss[1], min_loss[2]
    print 'max_p_u:'.ljust(20), "%.4f" % max_p_u[0], max_p_u[1], max_p_u[2]
    print 'max_p_c:'.ljust(20), "%.4f" % max_p_c[0], max_p_c[1], max_p_c[2]
    print 'max_p_ic:'.ljust(20), "%.4f" % max_p_ic[0], max_p_ic[1], max_p_ic[2]
    print 'max_p_c_diff_ict:'.ljust(20), "%.4f" % max_p_c_ict[0], max_p_c_ict[1], max_p_c_ict[2]
    print 'min_p_ictt:'.ljust(20), "%.4f" % min_p_ict[0], min_p_ict[1], min_p_ict[2]
    return min_loss, max_p_u, max_p_c, max_p_ic, min_p_ict, max_p_c_ict


if __name__ == '__main__':
    fs = "./simple.scalar.*1.log"
    print 'best scalar'
    min_loss, max_p_u, max_p_c, max_p_ic, min_p_ict, max_p_c_ict = scan_file(fs)
    fs = "./simple.adapt.*1.log"
    print '\nbest adapt'
    min_loss, max_p_u, max_p_c, max_p_ic, min_p_ict, max_p_c_ict = scan_file(fs)
