# coding=utf-8

# 引入外部库
import os, sys
import json

# 引入内部库
from LanguageModel.xlnet.Model.xlnet import Lm, lm_hparams
from TexCategorization.InductionNet.Model.induction_net import InductionNet, induction_net_hparams


def induction_net_model_train (data_path, lm_save_path, save_path):
	# 0.准备训练所需语料------------------------------
	with open('../../LanguageModel/xlnet/Lang/char_dict.json', 'r', encoding='utf-8') as fo:
		char_dict = json.load(fo)

	# 1.语言模型构建-----------------------------------
	lm_args = lm_hparams()
	lm_args.vocab_dict = char_dict
	lm_args.dropout = 0.
	lm_args.d_embed = 768
	lm_args.d_model = 768
	lm_args.n_layers = 12
	lm_args.n_head = 12
	lm_args.d_head = 64
	lm_args.init_range = 1
	lm_args.init_std = 0
	lm_args.is_training = True
	lm_args.task_name = 'xlnet'
	lm_args.save_path = lm_save_path
	lm = Lm(lm_args)

	# 2.归纳网络模型训练-----------------------------------
	args = induction_net_hparams()
	args.data_path = data_path
	args.vocab_dict = char_dict
	args.lm = lm
	args.is_training_lm = True
	args.episode = 12029
	args.max_step = 100
	args.lr = 1e-3
	args.dropout = 0.
	args.d_input = 768
	args.n_head = 12
	args.d_head = 64
	args.c_way = 5
	args.k_shot = 10
	args.n_query = 8
	args.n_route = 3
	args.d_tensor = 128
	args.init_range = 1
	args.init_std = 0
	args.is_training = True
	args.gpu_index = [0, 1, 2, 3]
	args.task_name = 'intention' # 可自选
	args.save_path = save_path
	os.makedirs(args.save_path, exist_ok=True)
	induction_net = InductionNet(args)

	induction_net.train_gpu()


def induction_net_model_predict (q_list, label_dict, lm_save_path, save_path):
	# .语料加载-----------------------------------
	with open('../../LanguageModel/xlnet/Lang/char_dict.json', 'r', encoding='utf-8') as fo:
		char_dict = json.load(fo)

	# 1.语言模型构建-----------------------------------
	lm_args = lm_hparams()
	lm_args.vocab_dict = char_dict
	lm_args.dropout = 0.
	lm_args.d_embed = 768
	lm_args.d_model = 768
	lm_args.n_layers = 12
	lm_args.n_head = 12
	lm_args.d_head = 64
	lm_args.init_range = 1
	lm_args.init_std = 0
	lm_args.is_training = False
	lm_args.task_name = 'xlnet'
	lm_args.save_path = lm_save_path
	lm = Lm(lm_args)

	# 2.归纳网络模型构建-----------------------------------
	args = induction_net_hparams()
	args.vocab_dict = char_dict
	args.lm = lm
	args.d_input = 768
	args.n_head = 12
	args.d_head = 64
	args.n_route = 3
	args.d_tensor = 128
	args.init_range = 1
	args.init_std = 0
	args.is_training = False
	args.task_name = 'intention'  # 可自选
	args.save_path = save_path
	induction_net = InductionNet(args)
	induction_net.start_session(label_dict)

	# 3. 开始预测-----------------------------------------
	outputs = induction_net.predict(q_list)

	return outputs
