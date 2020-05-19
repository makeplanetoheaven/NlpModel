# coding=utf-8


def back_translation_google (sen_list, lang='en', iter_num=1) -> list:
	"""
	google反向翻译
	:param sen_list:
	:param lang:
	:param iter_num:
	:return:
	"""
	from Sampling.DataAugmentation.Model.GoogleTranslator import GoogleTranslator
	from tqdm import tqdm
	fd_translator = GoogleTranslator(src='zh-CN', dest=lang)
	bd_translator = GoogleTranslator(src=lang, dest='zh-CN')

	res_list = sen_list

	# forward
	for _ in range(iter_num):
		temp_list = []
		for sen in tqdm(res_list):
			temp_list.append(fd_translator.translate(sen))

		# backward
		res_list = []
		for temp in tqdm(temp_list):
			res_list.append(bd_translator.translate(temp))

	return res_list
