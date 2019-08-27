# -*- coding: utf-8 -*-

import keras
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, GRU  # Merge
from keras.layers import Reshape, Dense, Dropout, Lambda  # Flatten
from keras_layer_normalization import LayerNormalization
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow as tf


def am_hparams ():
	params = tf.contrib.training.HParams(  # vocab
		vocab_size=50, lr=0.0008, gpu_nums=4, is_training=True)
	return params


# =============================搭建模型====================================
class Am():
	"""docstring for Amodel."""

	def __init__ (self, args):
		self.vocab_size = args.vocab_size
		self.gpu_nums = args.gpu_nums
		self.lr = args.lr
		self.dropout_r = 0.25
		self.is_training = args.is_training
		self._model_init()
		self._ctc_init()
		if self.is_training:
			self.opt_init()

	def _model_init (self):
		self.inputs = Input(name='the_inputs', shape=(None, 200, 1))

		# CNN-layers
		self.h1 = cnn_cell(32, self.inputs)
		if self.is_training:
			self.h1 = Dropout(self.dropout_r)(self.h1)
		self.h2 = cnn_cell(64, self.h1)
		if self.is_training:
			self.h2 = Dropout(self.dropout_r)(self.h2)
		self.h3 = cnn_cell(128, self.h2)
		if self.is_training:
			self.h3 = Dropout(self.dropout_r)(self.h3)
		self.h4 = cnn_cell(128, self.h3, pool=False)
		if self.is_training:
			self.h4 = Dropout(self.dropout_r)(self.h4)
		# (200 / 8) * 128 = 3200
		self.h5 = Reshape((-1, 3200))(self.h4)

		# linerTransform-layers
		# self.h5 = Dropout(0.2)(self.h5)
		self.h6 = dense(512, activation='relu')(self.h5)
		if self.is_training:
			self.h6 = Dropout(self.dropout_r * 2)(self.h6)

		# FSMN-layers
		with tf.variable_scope('cfsmn'):
			cfsmn = cfsmn_cell('cfsmn-cell', 512, 512, 1024, 40, 40)
			cfsmn_o, cfsmn_p_hatt = Lambda(cfsmn)(self.h6)
			cfsmn_o = LayerNormalization()(cfsmn_o)
			cfsmn_o = Lambda(tf.nn.swish)(cfsmn_o)
			if self.is_training:
				cfsmn_o = Dropout(self.dropout_r * 2)(cfsmn_o)

		with tf.variable_scope('dfsmn1'):
			dfsmn1 = dfsmn_cell('dfsmn1-cell', 512, 512, 1024, 40, 40)
			dfsmn1_o, dfsmn1_p_hatt = Lambda(dfsmn1)([cfsmn_o, cfsmn_p_hatt])
			dfsmn1_o = LayerNormalization()(dfsmn1_o)
			dfsmn1_o = Lambda(tf.nn.swish)(dfsmn1_o)
			if self.is_training:
				dfsmn1_o = Dropout(self.dropout_r * 2)(dfsmn1_o)

		with tf.variable_scope('dfsmn2'):
			dfsmn2 = dfsmn_cell('dfsmn2-cell', 512, 512, 1024, 40, 40)
			dfsmn2_o, dfsmn2_p_hatt = Lambda(dfsmn2)([dfsmn1_o, dfsmn1_p_hatt])
			dfsmn2_o = LayerNormalization()(dfsmn2_o)
			dfsmn2_o = Lambda(tf.nn.swish)(dfsmn2_o)
			if self.is_training:
				dfsmn2_o = Dropout(self.dropout_r * 2)(dfsmn2_o)

		with tf.variable_scope('dfsmn3'):
			dfsmn3 = dfsmn_cell('dfsmn3-cell', 512, 512, 1024, 40, 40)
			dfsmn3_o, dfsmn3_p_hatt = Lambda(dfsmn3)([dfsmn2_o, dfsmn2_p_hatt])
			dfsmn3_o = LayerNormalization()(dfsmn3_o)
			dfsmn3_o = Lambda(tf.nn.swish)(dfsmn3_o)
			if self.is_training:
				dfsmn3_o = Dropout(self.dropout_r * 2)(dfsmn3_o)

		with tf.variable_scope('dfsmn4'):
			dfsmn4 = dfsmn_cell('dfsmn4-cell', 512, 512, 1024, 40, 40)
			dfsmn4_o, dfsmn4_p_hatt = Lambda(dfsmn4)([dfsmn3_o, dfsmn3_p_hatt])
			dfsmn4_o = LayerNormalization()(dfsmn4_o)
			dfsmn4_o = Lambda(tf.nn.swish)(dfsmn4_o)
			if self.is_training:
				dfsmn4_o = Dropout(self.dropout_r * 2)(dfsmn4_o)

		with tf.variable_scope('dfsmn5'):
			dfsmn5 = dfsmn_cell('dfsmn5-cell', 512, 512, 1024, 40, 40)
			dfsmn5_o, dfsmn5_p_hatt = Lambda(dfsmn5)([dfsmn4_o, dfsmn4_p_hatt])
			dfsmn5_o = LayerNormalization()(dfsmn5_o)
			dfsmn5_o = Lambda(tf.nn.swish)(dfsmn5_o)
			if self.is_training:
				dfsmn5_o = Dropout(self.dropout_r * 2)(dfsmn5_o)

		# softmax-layers
		self.h8 = dense(384, activation='relu')(dfsmn5_o)
		self.outputs = dense(self.vocab_size, activation='softmax')(self.h8)

		self.model = Model(inputs=self.inputs, outputs=self.outputs)
		self.model.summary()

	def _ctc_init (self):
		self.labels = Input(name='the_labels', shape=[None], dtype='float32')
		self.input_length = Input(name='input_length', shape=[1], dtype='int64')
		self.label_length = Input(name='label_length', shape=[1], dtype='int64')
		self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')(
			[self.labels, self.outputs, self.input_length, self.label_length])
		self.ctc_model = Model(inputs=[self.labels, self.inputs, self.input_length, self.label_length],
		                       outputs=self.loss_out)

	def opt_init (self):
		opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0004, epsilon=10e-8)
		# opt = SGD(lr=self.lr, momentum=0.0, decay=0.00004, nesterov=False)
		if self.gpu_nums > 1:
			self.ctc_model = multi_gpu_model(self.ctc_model, gpus=self.gpu_nums)
		self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)


# ============================模型组件=================================
def conv2d (size):
	return Conv2D(size, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')


def norm (x):
	return BatchNormalization(axis=-1)(x)


def maxpool (x):
	return MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)


def dense (units, activation="relu"):
	return Dense(units, activation=activation, use_bias=True, kernel_initializer='he_normal')


# x.shape=(none, none, none)
# output.shape = (1/2, 1/2, 1/2)
def cnn_cell (size, x, pool=True):
	x = norm(conv2d(size)(x))
	x = norm(conv2d(size)(x))
	if pool:
		x = maxpool(x)
	return x


def ctc_lambda (args):
	labels, y_pred, input_length, label_length = args
	y_pred = y_pred[:, :, :]
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class sfsmn_cell(object):
	def __init__ (self, name, input_size, output_size, memory_size):
		super(sfsmn_cell, self).__init__()
		self.__name__ = name
		self._memory_size = memory_size
		self._output_size = output_size
		self._input_size = input_size

		self._build_graph()

	def _build_graph (self):
		self._W1 = tf.get_variable("fsmn_w1", [self._input_size, self._output_size],
		                           initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
		self._W2 = tf.get_variable("fsmn_w2", [self._input_size, self._output_size],
		                           initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
		self._bias = tf.get_variable("fsmn_bias", [self._output_size],
		                             initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
		self._memory_weights = tf.get_variable("memory_weights", [self._memory_size],
		                                       initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))

	def __call__ (self, inputs):
		def for_cond (step, memory_matrix):
			return step < num_steps

		def for_body (step, memory_matrix):
			left_pad_num = tf.maximum(0, step + 1 - self._memory_size)
			right_pad_num = num_steps - step - 1
			mem = self._memory_weights[tf.minimum(step, self._memory_size)::-1]
			d_batch = tf.pad(mem, [[left_pad_num, right_pad_num]])

			return step + 1, memory_matrix.write(step, d_batch)

		# memory build
		num_steps = tf.shape(inputs)[1]
		memory_matrix = tf.TensorArray(dtype=tf.float32, size=num_steps)
		_, memory_matrix = tf.while_loop(for_cond, for_body, [0, memory_matrix])
		memory_matrix = memory_matrix.stack()

		# memory block
		h_hatt = tf.transpose(tf.tensordot(tf.transpose(inputs, [0, 2, 1]), tf.transpose(memory_matrix), axes=1),
		                      [0, 2, 1])

		h = tf.tensordot(inputs, self._W1, axes=1)

		# liner transform
		h += tf.tensordot(h_hatt, self._W2, axes=1) + self._bias

		return h


class vfsmn_cell(object):
	def __init__ (self, name, input_size, output_size, l_memory_size, r_memory_size):
		super(vfsmn_cell, self).__init__()
		self.__name__ = name
		self._l_memory_size = l_memory_size
		self._r_memory_size = r_memory_size
		self._memory_size = l_memory_size + r_memory_size
		self._output_size = output_size
		self._input_size = input_size

		self._build_graph()

	def _build_graph (self):
		self._W1 = tf.get_variable("fsmn_w1", [self._input_size, self._output_size],
		                           initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
		self._W2 = tf.get_variable("fsmn_w2", [self._input_size, self._output_size],
		                           initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
		self._bias = tf.get_variable("fsmn_bias", [self._output_size],
		                             initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
		self._memory_weights = tf.get_variable("memory_weights", [self._memory_size],
		                                       initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))

	def __call__ (self, inputs):
		def for_cond (step, memory_matrix):
			return step < num_steps

		def for_body (step, memory_matrix):
			left_pad_num = tf.maximum(0, step + 1 - self._l_memory_size)
			right_pad_num = tf.maximum(0, num_steps - step - 1 - self._r_memory_size)
			l_mem = self._memory_weights[tf.minimum(step, self._l_memory_size - 1)::-1]
			r_mem = self._memory_weights[
			        self._l_memory_size:self._l_memory_size + tf.minimum(num_steps - step - 1, self._r_memory_size)]
			mem = tf.concat([l_mem, r_mem], 0)
			d_batch = tf.pad(mem, [[left_pad_num, right_pad_num]])

			return step + 1, memory_matrix.write(step, d_batch)

		# memory build
		num_steps = tf.shape(inputs)[1]
		memory_matrix = tf.TensorArray(dtype=tf.float32, size=num_steps)
		_, memory_matrix = tf.while_loop(for_cond, for_body, [0, memory_matrix])
		memory_matrix = memory_matrix.stack()

		# memory block
		h_hatt = tf.transpose(tf.tensordot(tf.transpose(inputs, [0, 2, 1]), tf.transpose(memory_matrix), axes=1),
		                      [0, 2, 1])

		h = tf.tensordot(inputs, self._W1, axes=1)

		# liner transform
		h += tf.tensordot(h_hatt, self._W2, axes=1) + self._bias

		return h


class cfsmn_cell(object):
	def __init__ (self, name, input_size, output_size, hidden_size, l_memory_size, r_memory_size):
		super(cfsmn_cell, self).__init__()
		self.__name__ = name
		self._input_size = input_size
		self._output_size = output_size
		self._hidden_size = hidden_size
		self._l_memory_size = l_memory_size
		self._r_memory_size = r_memory_size
		self._memory_size = l_memory_size + r_memory_size

		self._build_graph()

	def _build_graph (self):
		self.lay_norm = tf_LayerNormalization('cfsmn_laynor', self._hidden_size)
		self._V = tf.get_variable("cfsmn_V", [self._input_size, self._hidden_size],
		                          initializer=tf.contrib.layers.xavier_initializer())
		self._bias_V = tf.get_variable("cfsmn_V_bias", [self._hidden_size],
		                               initializer=tf.contrib.layers.xavier_initializer())
		self._U = tf.get_variable("cfsmn_U", [self._hidden_size, self._output_size],
		                          initializer=tf.contrib.layers.xavier_initializer())
		self._bias_U = tf.get_variable("cfsmn_U_bias", [self._output_size],
		                               initializer=tf.contrib.layers.xavier_initializer())
		self._memory_weights = tf.get_variable("memory_weights", [self._memory_size],
		                                       initializer=tf.contrib.layers.xavier_initializer())

	def __call__ (self, inputs):
		def for_cond (step, memory_matrix):
			return step < num_steps

		def for_body (step, memory_matrix):
			left_pad_num = tf.maximum(0, step + 1 - self._l_memory_size)
			right_pad_num = tf.maximum(0, num_steps - step - 1 - self._r_memory_size)
			l_mem = self._memory_weights[tf.minimum(step, self._l_memory_size - 1)::-1]
			r_mem = self._memory_weights[
			        self._l_memory_size:self._l_memory_size + tf.minimum(num_steps - step - 1, self._r_memory_size)]
			mem = tf.concat([l_mem, r_mem], 0)
			d_batch = tf.pad(mem, [[left_pad_num, right_pad_num]])

			return step + 1, memory_matrix.write(step, d_batch)

		# memory build
		num_steps = tf.shape(inputs)[1]
		memory_matrix = tf.TensorArray(dtype=tf.float32, size=num_steps)
		_, memory_matrix = tf.while_loop(for_cond, for_body, [0, memory_matrix])
		memory_matrix = memory_matrix.stack()

		p = tf.tensordot(inputs, self._V, axes=1) + self._bias_V

		# memory block
		p_hatt = tf.transpose(tf.tensordot(tf.transpose(p, [0, 2, 1]), tf.transpose(memory_matrix), axes=1), [0, 2, 1])
		p_hatt = self.lay_norm(p + p_hatt)

		# liner transform
		h = tf.tensordot(p_hatt, self._U, axes=1) + self._bias_U

		return [h, p_hatt]


class dfsmn_cell(object):
	def __init__ (self, name, input_size, output_size, hidden_size, l_memory_size, r_memory_size):
		super(dfsmn_cell, self).__init__()
		self.__name__ = name
		self._input_size = input_size
		self._output_size = output_size
		self._hidden_size = hidden_size
		self._l_memory_size = l_memory_size
		self._r_memory_size = r_memory_size
		self._memory_size = l_memory_size + r_memory_size

		self._build_graph()

	def _build_graph (self):
		self.lay_norm = tf_LayerNormalization('dfsmn_laynor', self._hidden_size)
		self._V = tf.get_variable("dfsmn_V", [self._input_size, self._hidden_size],
		                          initializer=tf.contrib.layers.xavier_initializer())
		self._bias_V = tf.get_variable("dfsmn_V_bias", [self._hidden_size],
		                               initializer=tf.contrib.layers.xavier_initializer())
		self._U = tf.get_variable("dfsmn_U", [self._hidden_size, self._output_size],
		                          initializer=tf.contrib.layers.xavier_initializer())
		self._bias_U = tf.get_variable("dfsmn_U_bias", [self._output_size],
		                               initializer=tf.contrib.layers.xavier_initializer())
		self._memory_weights = tf.get_variable("memory_weights", [self._memory_size],
		                                       initializer=tf.contrib.layers.xavier_initializer())

	def __call__ (self, args):
		inputs = args[0]
		last_p_hatt = args[1]

		def for_cond (step, memory_matrix):
			return step < num_steps

		def for_body (step, memory_matrix):
			left_pad_num = tf.maximum(0, step + 1 - self._l_memory_size)
			right_pad_num = tf.maximum(0, num_steps - step - 1 - self._r_memory_size)
			l_mem = self._memory_weights[tf.minimum(step, self._l_memory_size - 1)::-1]
			r_mem = self._memory_weights[
			        self._l_memory_size:self._l_memory_size + tf.minimum(num_steps - step - 1, self._r_memory_size)]
			mem = tf.concat([l_mem, r_mem], 0)
			d_batch = tf.pad(mem, [[left_pad_num, right_pad_num]])

			return step + 1, memory_matrix.write(step, d_batch)

		# memory build
		num_steps = tf.shape(inputs)[1]
		memory_matrix = tf.TensorArray(dtype=tf.float32, size=num_steps)
		_, memory_matrix = tf.while_loop(for_cond, for_body, [0, memory_matrix])
		memory_matrix = memory_matrix.stack()

		p = tf.tensordot(inputs, self._V, axes=1) + self._bias_V

		# memory block
		p_hatt = tf.transpose(tf.tensordot(tf.transpose(p, [0, 2, 1]), tf.transpose(memory_matrix), axes=1), [0, 2, 1])
		p_hatt = self.lay_norm(last_p_hatt + p + p_hatt)

		# liner transform
		h = tf.tensordot(p_hatt, self._U, axes=1) + self._bias_U

		return [h, p_hatt]


class tf_LayerNormalization(object):
	def __init__ (self, name, hidden_size):
		super(tf_LayerNormalization, self).__init__()
		self.__name__ = name
		self.hidden_size = hidden_size
		self.build()

	def build (self):
		self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size], initializer=tf.ones_initializer())
		self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size], initializer=tf.zeros_initializer())
		self.built = True

	def __call__ (self, x, epsilon=1e-6):
		mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
		variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
		norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
		return norm_x * self.scale + self.bias
