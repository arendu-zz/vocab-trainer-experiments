#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)10s %(name)s %(asctime)s: %(message)s')
import codecs
import argparse
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('-e', action='store' , dest='example_option', default='example default', required = True)
    opt.add_argument('-a', action='store_true' ,dest='example_option2', default=False, required = True)
    options = opt.parse_args()
    logging.debug(str(options))

