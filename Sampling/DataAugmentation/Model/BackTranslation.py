# coding=utf-8


def back_translation_google (sen_list, lang='en') -> list:
	"""
	google反向翻译
	:param sen_list:
	:param lang:
	:return:
	"""
	from tqdm import tqdm
	from lxml import html
	from selenium import webdriver
	from selenium.webdriver.common.by import By
	from selenium.webdriver.support.ui import WebDriverWait
	from selenium.webdriver.support import expected_conditions as EC
	from selenium.webdriver.chrome.options import Options
	from retry import retry

	# 隐藏浏览器界面
	chrome_options = Options()
	chrome_options.add_argument('--headless')
	browser = webdriver.Chrome("C:\\Users\\Dell\\Desktop\\chromedriver.exe", options=chrome_options)

	@retry(tries=3, delay=1)
	def translate (sen, src='zh-CN', dest='en'):
		base_url = 'https://translate.google.cn/#view=home&op=translate&sl=%s&tl=%s' % (src, dest)

		if browser.current_url != base_url:
			browser.get(base_url)

		submit = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="source"]')))
		submit.clear()
		submit.send_keys(sen)
		WebDriverWait(browser, 10).until(
			EC.presence_of_element_located((By.XPATH, '//span[@class="tlid-translation translation"]')))
		source = html.etree.HTML(browser.page_source)
		result = source.xpath('//span[@class="tlid-translation translation"]//text()')[0]

		return result

	# forward
	temp_list = []
	for sen in tqdm(sen_list):
		temp_list.append(translate(sen, src='zh-CN', dest=lang))
	print(temp_list)

	# backward
	res_list = []
	for temp in tqdm(temp_list):
		res_list.append(translate(temp, src=lang, dest='zh-cn'))

	return res_list


def back_translation_youdao (sen_list, lang='en') -> list:
	"""
	youdao反向翻译
	:param sen_list:
	:param lang:
	:return:
	"""
	import time
	import random
	import hashlib
	from tqdm import tqdm
	import urllib.request
	import urllib.parse

	def translate (sen, src='zh', dest='en'):
		url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&sessionFrom=https://www.google.com/'
		data = {}
		u = 'fanyideskweb'
		d = sen
		f = str(int(time.time() * 1000) + random.randint(1, 10))
		c = 'rY0D^0\'nM0}g5Mm1z%1G4'
		sign = hashlib.md5((u + d + f + c).encode('utf-8')).hexdigest()
		data['i'] = sen
		data['from'] = src
		data['to'] = dest
		data['smartresult'] = 'dict'
		data['client'] = 'fanyideskweb'
		data['salt'] = f
		data['sign'] = sign
		data['doctype'] = 'json'
		data['version'] = '2.1'
		data['keyfrom'] = 'fanyi.web'
		data['action'] = 'FY_BY_CL1CKBUTTON'
		data['typoResult'] = 'true'
		data = urllib.parse.urlencode(data).encode('utf-8')
		request = urllib.request.Request(url=url, data=data, method='POST')
		response = urllib.request.urlopen(request)

		return response.read().decode('utf-8')

	# forward
	temp_list = []
	for sen in tqdm(sen_list):
		temp_list.append(translate(sen, src='zh-CN', dest=lang))
	print(temp_list)

	# backward
	res_list = []
	for temp in tqdm(temp_list):
		res_list.append(translate(temp, src=lang, dest='zh-cn'))

	return res_list
