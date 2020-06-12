# _*_ coding:utf-8 _*_
"""
@time: 2020年5月11日
@author: hecl
@desc: Named Entity Recognition based on LTP
"""

# Albert_NER_CONF
model_name='albert_small'
weight_path=r'''../albert_small/albert_small_lstm_crf_ner6layers.weights'''
model_path=r'..//albert_small'
config_path = model_path + r'/albert_config_small_google.json'
checkpoint_path = model_path + '/albert_model.ckpt'
dict_path = model_path + '/vocab.txt'
layers=6
unshared=False
albert_ner_path='./albert_small_ner.h5'
