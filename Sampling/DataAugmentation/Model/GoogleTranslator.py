# coding=utf-8

import requests
import re
import json
import time


class GoogleTranslator():
	_host = 'translate.google.cn'
	_headers = {
		'Host': _host,
		'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Mobile Safari/537.36',
		'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
		'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
		'Accept-Encoding': 'gzip, deflate, br',
		'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8',
		'Referer': 'https://' + _host,
		'Connection': 'keep-alive',
		'Cache-Control': 'max-age=0'
	}
	_language = {
		'afrikaans': 'af',
		'arabic': 'ar',
		'belarusian': 'be',
		'bulgarian': 'bg',
		'catalan': 'ca',
		'czech': 'cs',
		'welsh': 'cy',
		'danish': 'da',
		'german': 'de',
		'greek': 'el',
		'english': 'en',
		'esperanto': 'eo',
		'spanish': 'es',
		'estonian': 'et',
		'persian': 'fa',
		'finnish': 'fi',
		'french': 'fr',
		'irish': 'ga',
		'galician': 'gl',
		'hindi': 'hi',
		'croatian': 'hr',
		'hungarian': 'hu',
		'indonesian': 'id',
		'icelandic': 'is',
		'italian': 'it',
		'hebrew': 'iw',
		'japanese': 'ja',
		'korean': 'ko',
		'latin': 'la',
		'lithuanian': 'lt',
		'latvian': 'lv',
		'macedonian': 'mk',
		'malay': 'ms',
		'maltese': 'mt',
		'dutch': 'nl',
		'norwegian': 'no',
		'polish': 'pl',
		'portuguese': 'pt',
		'romanian': 'ro',
		'russian': 'ru',
		'slovak': 'sk',
		'slovenian': 'sl',
		'albanian': 'sq',
		'serbian': 'sr',
		'swedish': 'sv',
		'swahili': 'sw',
		'thai': 'th',
		'filipino': 'tl',
		'turkish': 'tr',
		'ukrainian': 'uk',
		'vietnamese': 'vi',
		'yiddish': 'yi',
		'chinese_simplified': 'zh-CN',
		'chinese_traditional': 'zh-TW',
		'auto': 'auto'
	}
	_url = 'https://' + _host + '/translate_a/single'
	_params = {
		'client': 'webapp',
		'sl': 'en',
		'tl':  'zh-CN',
		'hl':  'zh-CN',
		'dt': 'at',
		'dt': 'bd',
		'dt': 'ex',
		'dt': 'ld',
		'dt': 'md',
		'dt': 'qca',
		'dt': 'rw',
		'dt': 'rm',
		'dt': 'ss',
		'dt': 't',
		'otf': '1',
		'ssel': '0',
		'tsel': '0',
		'kc': '1'
	}
	__cookies = None
	__googleTokenKey = '376032.257956'
	__googleTokenKeyUpdataTime = 600.0
	__googleTokenKeyRetireTime = time.time() + 600.0

	def __init__ (self, src='en', dest='zh-CN', tkkUpdataTime=600.0):
		if src not in self._language and src not in self._language.values():
			src = 'auto'
		if dest not in self._language and dest not in self._language.values():
			dest = 'auto'
		self._params['sl'] = src
		self._params['tl'] = dest
		self.googleTokenKeyUpdataTime = tkkUpdataTime
		self.__updateGoogleTokenKey()

	def __updateGoogleTokenKey (self):
		self.__googleTokenKey = self.__getGoogleTokenKey()
		self.__googleTokenKeyRetireTime = time.time() + self.__googleTokenKeyUpdataTime

	def __getGoogleTokenKey (self):
		"""Get the Google TKK from https://translate.google.cn"""
		# TKK example: '435075.3634891900'
		result = ''
		try:
			res = requests.get('https://' + self._host, timeout=3)
			res.raise_for_status()
			self.__cookies = res.cookies
			result = re.search(r'tkk\:\'(\d+\.\d+)?\'', res.text).group(1)
		except requests.exceptions.ReadTimeout as ex:
			print('ERROR: ' + str(ex))
			time.sleep(1)
		return result

	def __getGoogleToken (self, a, TKK):
		"""Calculate Google tk from TKK """

		# https://www.cnblogs.com/chicsky/p/7443830.html
		# if text = 'Tablet Developer' and TKK = '435102.3120524463', then tk = '315066.159012'

		def RL (a, b):
			for d in range(0, len(b) - 2, 3):
				c = b[d + 2]
				c = ord(c[0]) - 87 if 'a' <= c else int(c)
				c = a >> c if '+' == b[d + 1] else a << c
				a = a + c & 4294967295 if '+' == b[d] else a ^ c
			return a

		g = []
		f = 0
		while f < len(a):
			c = ord(a[f])
			if 128 > c:
				g.append(c)
			else:
				if 2048 > c:
					g.append((c >> 6) | 192)
				else:
					if (55296 == (c & 64512)) and (f + 1 < len(a)) and (56320 == (ord(a[f + 1]) & 64512)):
						f += 1
						c = 65536 + ((c & 1023) << 10) + (ord(a[f]) & 1023)
						g.append((c >> 18) | 240)
						g.append((c >> 12) & 63 | 128)
					else:
						g.append((c >> 12) | 224)
						g.append((c >> 6) & 63 | 128)
				g.append((c & 63) | 128)
			f += 1

		e = TKK.split('.')
		h = int(e[0]) or 0
		t = h
		for item in g:
			t += item
			t = RL(t, '+-a^+6')
		t = RL(t, '+-3^+b+-f')
		t ^= int(e[1]) or 0
		if 0 > t:
			t = (t & 2147483647) + 2147483648
		result = t % 1000000
		return str(result) + '.' + str(result ^ h)

	def translate (self, text):
		if time.time() > self.__googleTokenKeyRetireTime:
			self.__updateGoogleTokenKey()
		data = {'q': text}
		self._params['tk'] = self.__getGoogleToken(text, self.__googleTokenKey)
		result = ''
		try:
			res = requests.post(self._url, headers=self._headers, cookies=self.__cookies, data=data,
			                    params=self._params, timeout=6)
			res.raise_for_status()
			jsonText = res.text
			if len(jsonText) > 0:
				jsonResult = json.loads(jsonText)
				if len(jsonResult[0]) > 0:
					for item in jsonResult[0]:
						result += item[0]
			return result
		except Exception as ex:
			print('ERROR: ' + str(ex))
			return ''
