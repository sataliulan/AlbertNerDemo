# _*_ coding:utf-8 _*_
"""
@author：hecl
@date：2020年6月9日
@desc：工具类，用于公共方法复用
"""

import re

def brace_remove(sent):
    # 去除[2-3]等干扰项
    brace_compile = re.compile('\[.*?\]')
    brace_list = brace_compile.findall(sent)
    for b in brace_list:
        sent = sent.replace(b, '')
    # 去除（原单位xx）等干扰项
    brace_compile1 = re.compile('（原.*?）|（\d）')
    brace_list1 = brace_compile1.findall(sent)
    for b in brace_list1:
        sent = sent.replace(b, '')
    # sent = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[~@#￥%……&*（）]", "", sent)
    # sent = sent.translate(str.maketrans('', '', string.punctuation))
    sent = sent.replace('(', '')
    sent = sent.replace(')', '')
    sent = sent.replace('（', '')
    sent = sent.replace('）', '')
    sent = sent.replace('*', '')
    sent = sent.replace('+', '')
    sent = sent.replace('^', '')
    sent = sent.replace('?', '')
    sent = sent.replace('$', '')
    # sent = sent.replace(' ', '')
    sent=sent.replace('\u3000','')
    sent = sent.replace('．', '.').replace('·', '')
    sent=sent.replace('\\','|')

    print('去除干扰项的句子：', sent)
    return sent