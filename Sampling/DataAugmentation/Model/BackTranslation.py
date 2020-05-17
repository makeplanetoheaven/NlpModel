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
	import googletrans
	google_translator = googletrans.Translator(service_urls=['translate.google.cn'])

	# forward
	temp_list = []
	fd_translator = GoogleTranslator(src='zh-CN', dest=lang)
	for sen in tqdm(sen_list):
		temp = fd_translator.translate(sen)
		if temp == '':
			temp_list.append(google_translator.translate(sen, src='zh-CN', dest=lang).text)
		else:
			temp_list.append(temp)

	# backward
	res_list = []
	bd_translator = GoogleTranslator(src=lang, dest='zh-CN')
	for temp in tqdm(temp_list):
		res = bd_translator.translate(temp)
		if res == '':
			temp_list.append(google_translator.translate(temp, src=lang, dest='zh-CN').text)
		else:
			res_list.append(res)

	return res_list
