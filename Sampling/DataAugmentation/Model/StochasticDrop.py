# coding=utf-8


import random
import math
from tqdm import tqdm


def stochastic_drop(sen_list: list, rate=0.15) -> list:
	"""
	随机丢弃句子中的若干字符
	:param sen_list:
	:param rate:
	:return:
	"""
	res_list = []
	for sen in tqdm(sen_list):
		s_l = len(sen)
		idx_list = [i for i in range(s_l)]
		drop_n = math.ceil(s_l * rate)
		drop_list = random.sample(idx_list, drop_n)
		new_sen = ''
		for idx in idx_list:
			if idx not in drop_list:
				new_sen += sen[idx]
		res_list.append(new_sen)

	return res_list
