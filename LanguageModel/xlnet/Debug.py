# coding=utf-8

# 引入外部库
import json

# 引入内部库
from LanguageModel.xlnet.Model.xlnet import Lm, lm_hparams


def xlnet_model_train (data_path, save_path):
	# 0.准备训练所需语料------------------------------
	with open('./lang/char_dict.json', 'r', encoding='utf-8') as fo:
		char_dict = json.load(fo)

	# 1.语言模型训练-----------------------------------
	lm_args = lm_hparams()
	lm_args.data_path = data_path
	lm_args.vocab_dict = char_dict
	lm_args.bsz = 512
	lm_args.seq_len = 32
	lm_args.epoch = 1
	lm_args.max_step = 10
	lm_args.lr = 1e-3
	lm_args.dropout = 0.5
	lm_args.d_embed = 768
	lm_args.d_model = 768
	lm_args.n_layers = 8
	lm_args.n_head = 8
	lm_args.d_head = 768
	lm_args.init_range = 1
	lm_args.init_std = 0
	lm_args.is_training = True
	lm_args.task_name = 'xlnet'
	lm_args.save_path = save_path
	lm = Lm(lm_args)

	lm.generate_data_set()
	lm.train_gpu(4)
