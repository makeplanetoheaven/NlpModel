# coding=utf-8


def back_translation_google (sen_list, lang='en') -> list:
	"""
	google反向翻译
	:param sen_list:
	:param lang:
	:return:
	"""
	from Sampling.DataAugmentation.Model.GoogleTranslator import GoogleTranslator
	from tqdm import tqdm

	# forward
	temp_list = []
	fd_translator = GoogleTranslator(src='zh-CN', dest=lang)
	for sen in tqdm(sen_list):
		temp_list.append(fd_translator.translate(sen))

	# backward
	res_list = []
	bd_translator = GoogleTranslator(src=lang, dest='zh-CN')
	for temp in tqdm(temp_list):
		res_list.append(bd_translator.translate(temp))

	return res_list
