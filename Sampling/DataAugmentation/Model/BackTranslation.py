# coding=utf-8


import grequests
from tqdm import tqdm
from googletrans import Translator
from googletrans.utils import format_json

translator = Translator(service_urls=['translate.google.cn'])


def get_trans_res (urls: list) -> list:
	res_list = []
	reqs = (grequests.get(u, verify=True, allow_redirects=True, timeout=4) for u in urls)
	res = grequests.map(reqs, size=20)
	for r in res:
		if hasattr(r, 'status_code'):
			if r.status_code == 200:
				try:
					a = format_json(r.text)
					target = ''.join([d[0] if d[0] else '' for d in a[0]])
					source = ''.join([d[1] if d[1] else '' for d in a[0]])
				except Exception as e:
					source = ''
					target = ''
				if len(source) != 0 and len(target) != 0:
					res_list.append(target)
				else:
					res_list.append('')
			else:
				res_list.append('')

	return res_list


def sentence_translate (line, src='zh-cn', dest='en'):
	"""

	:param line:
	:param src:
	:param dest:
	:return:
	"""
	line = line.strip()
	text = translator.translate(line, src=src, dest=dest).text

	return text


def total_translate (sen_list: list, src='zh-cn', dest='en') -> list:
	"""

	:param sen_list:
	:param src:
	:param dest:
	:return:
	"""
	res_list = []

	urls = []
	num = 0
	for sen in tqdm(sen_list):
		num += 1
		token = translator.token_acquirer.do(sen)
		url = "https://translate.google.cn/translate_a/single?client=t&sl={0}&tl={1}&hl={1}&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&ie=UTF-8&oe=UTF-8&otf=1&ssel=3&tsel=0&kc=1&tk={2}&q={3}".format(
			src, dest, token, sen)
		urls.append(url)

		if len(urls) >= 50:
			res_list += get_trans_res(urls)
			urls = []

	res_list += get_trans_res(urls)

	return res_list


def complete_translate (sen_list: list, res_list: list, src='zh-cn', dest='en'):
	"""

	:param sen_list:
	:param sen_list:
	:param src:
	:param dest:
	:return:
	"""
	for i in range(len(sen_list)):
		src_sen = sen_list[i]
		dest_sen = res_list[i]
		if dest_sen == '':
			res_list[i] = sentence_translate(src_sen, src, dest)


def back_translation (sen_list, lang='en') -> list:
	"""
	反向翻译
	:param sen_list:
	:param lang:
	:return:
	"""
	# forward
	temp_list = total_translate(sen_list, src='zh-cn', dest=lang)
	complete_translate(sen_list, temp_list, src='zh-cn', dest=lang)

	# backward
	res_list = total_translate(temp_list, src=lang, dest='zh-cn')
	complete_translate(temp_list, res_list, src=lang, dest='zh-cn')

	return res_list
