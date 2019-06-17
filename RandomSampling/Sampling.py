# coding=utf-8

"""
author: 王黎成
function: 采样函数库
"""

# 引入外部库
import random


# 引入内部库


def simple_sampling (data_mat: list, num: int) -> list:
	"""
	简单随机采样
	:param data_mat:
	:param num:
	:return:
	"""
	try:
		samples = random.sample(data_mat, num)
		return samples
	except:
		print('sample larger than population')


def systematic_sampling (data_mat: list, num: int) -> list:
	"""
	系统采样
	:param data_mat:
	:param num:
	:return:
	"""
	k = int(len(data_mat) / num)
	samples = [random.sample(data_mat[i * k:(i + 1) * k], 1) for i in range(num)]
	return samples
