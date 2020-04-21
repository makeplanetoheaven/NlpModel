# coding=utf-8

# 引入外部库
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm

# 引入内部库
from SpeechRecognition.AcousticModel.dfsmn_v1.utils import get_data, data_hparams, get_online_data, decode_ctc, GetEditDistance
from SpeechRecognition.AcousticModel.dfsmn_v1.Model.cnn_dfsmn_ctc import Am, am_hparams


def dfsmn_model_train (train_data_path, label_data_path):
	# 0.准备训练所需数据------------------------------
	data_args = data_hparams()
	data_args.data_type = 'train'
	data_args.data_path = train_data_path
	data_args.label_data_path = label_data_path
	data_args.thchs30 = True
	data_args.aishell = True
	data_args.batch_size = 16
	# data_args.data_length = 100
	data_args.data_length = None
	data_args.shuffle = True
	train_data = get_data(data_args)

	# 0.准备验证所需数据------------------------------
	data_args = data_hparams()
	data_args.data_type = 'dev'
	data_args.data_path = train_data_path
	data_args.label_data_path = label_data_path
	data_args.thchs30 = True
	data_args.aishell = True
	data_args.batch_size = 16
	# data_args.data_length = 16
	data_args.data_length = None
	data_args.shuffle = True
	dev_data = get_data(data_args)

	# 1.声学模型训练-----------------------------------
	from .Model.cnn_dfsmn_ctc import Am, am_hparams
	am_args = am_hparams()
	am_args.vocab_size = len(train_data.am_vocab)
	am_args.gpu_nums = 1
	am_args.lr = 0.001
	am_args.is_training = True
	am = Am(am_args)

	if os.path.exists('ModelMemory/cnn_dfsmn_ctc_model.h5'):
		print('load acoustic model...')
		am.ctc_model.load_weights('ModelMemory/cnn_dfsmn_ctc_model.h5')

	batch_num = len(train_data.wav_lst) // train_data.batch_size

	# checkpoint
	# ckpt = "model_{epoch:02d}.hdf5"
	# checkpoint = ModelCheckpoint(os.path.join('./logs_am/checkpoint', ckpt), monitor='val_loss', save_weights_only=False, verbose=1, save_best_only=True)


	batch = train_data.get_am_batch()
	dev_batch = dev_data.get_am_batch()
	epochs = 150
	# am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, callbacks=[checkpoint], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)
	am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, workers=1, use_multiprocessing=False,
	                           validation_data=dev_batch, validation_steps=500)
	am.ctc_model.save_weights('ModelMemory/cnn_dfsmn_ctc_model.h5')


def dfsmn_model_decode (wav_file_path):
	# 1.语料加载-----------------------------------
	print('loading Lang...')
	pinyin_list = []
	with open('Lang/pinyin.txt', 'r', encoding='utf-8') as file_object:
		for pinyin in tqdm(file_object):
			pinyin_list.append(pinyin.rstrip('\n'))
	pinyin_list.append('_')

	# 2.声学模型加载-----------------------------------
	print('loading acoustic model...')
	am_args = am_hparams()
	am_args.vocab_size = len(pinyin_list)
	am_args.is_training = False
	am = Am(am_args)
	am.ctc_model.load_weights('ModelMemory/cnn_dfsmn_ctc_model.h5')

	# 3. 启动在线识别-------------------------------------------
	print('start online decode...')
	x = get_online_data(wav_file_path)
	pinyin_id = am.model.predict(x, steps=1)
	_, pinyin = decode_ctc(pinyin_id, pinyin_list)
