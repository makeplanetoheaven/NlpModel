# coding=utf-8


import json
from tqdm import tqdm
from urllib import request, parse


def youdao_translte (content, src='zh-cn', dest='en'):
	Request_URL = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'
	Form_Data = {}
	Form_Data['from'] = src
	Form_Data['to'] = dest
	Form_Data['doctype'] = 'json'  # can't delete,return type
	Form_Data['i'] = content  # can't delete,what you want to translate
	data = parse.urlencode(Form_Data).encode('utf-8')
	response = request.urlopen(Request_URL, data)
	html = response.read().decode('utf-8')
	translate_results = json.loads(html)
	translate_results = translate_results['translateResult'][0][0]['tgt']

	return translate_results


def back_translation (sen_list, lang='en') -> list:
	"""
	反向翻译
	:param sen_list:
	:param lang:
	:return:
	"""
	# forward
	temp_list = []
	for sen in tqdm(sen_list):
		temp_list.append(youdao_translte(sen, src='zh-cn', dest=lang))

	# backward
	res_list = []
	for temp in tqdm(temp_list):
		res_list.append(youdao_translte(temp, src=lang, dest='zh-cn'))

	return res_list
