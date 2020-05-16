# coding=utf-8

# 引入外部库
import json

# 引入内部库
from Sampling.DataAugmentation.Model.StochasticDrop import *
from Sampling.DataAugmentation.Model.BackTranslation import *


data_path = 'C:\\Users\\Dell\\Desktop\\0.txt'
sen_list = []
SIZE = 30000
LEN = 512
with open(data_path, 'r', encoding='utf-8') as fo:
	i = 0
	for line in fo:
		if len(line) <= LEN:
			sen_list.append(line.rstrip('\n'))
		if len(sen_list) == SIZE:
			break

# StochasticDrop
print('begin StochasticDrop')
res_list = stochastic_drop(sen_list)
faq_list = []
for i in range(len(sen_list)):
	faq_dict = {
		'index': i,
		'问题': sen_list[i],
		'答案': res_list[i]
	}
	faq_list.append(faq_dict)
with open('./StochasticDrop_faq.json', 'w', encoding='utf-8') as fo:
	json.dump(faq_list, fo, ensure_ascii=False, indent=2)

# BackTranslation
print('begin BackTranslation')
res_list = back_translation_youdao(sen_list)
faq_list = []
for i in range(len(sen_list)):
	faq_dict = {
		'index': i,
		'问题': sen_list[i],
		'答案': res_list[i]
	}
	faq_list.append(faq_dict)
with open('./BackTranslation_faq.json', 'w', encoding='utf-8') as fo:
	json.dump(faq_list, fo, ensure_ascii=False, indent=2)
