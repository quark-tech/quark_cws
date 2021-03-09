#! /usr/local/bin/python2.7
# -*- coding: utf-8 -*-
import sys
import os.path
import random
reload(sys)
sys.setdefaultencoding('utf-8')
def load_voc_list(filename):
    voc = []
    with open(filename, 'r') as fd:
        for line in fd:
            voc.append(line.strip())
    return voc
if __name__ == "__main__" :
    f_name = sys.argv[1]
    count = 0
    with open(f_name, 'r') as fd:
        for line in fd:
            sents = line.strip().replace(',,,', '，').split("。")
            for sent in sents:
                q = sent.strip().replace(" ", "")
                l = sent.strip().replace("  ", "||")
                if len(q.decode('utf-8')) <2:
                    continue
                if len(q.decode('utf-8'))>128:
                    count = count + 1
                print q+"\t"+l
        print count

