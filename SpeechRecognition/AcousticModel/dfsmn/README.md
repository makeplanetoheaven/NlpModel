# 语音识别中基于CNN+DFSMN的声学模型实现及代码开源
本模型是在传统CNN模型的基础上，引入2018年阿里提出的声学模型DFSMN，论文地址：https://arxiv.org/pdf/1803.05030.pdf。

该声学模型使用的输入是经过fbank特征提取以后的具有16ms采样率，单声道音频数据。模型整体的语音识别框架使用的是Github：https://github.com/audier/DeepSpeechRecognition。

在该模块中，主要包含了以下4个部分内容：
* [模型实现代码](#模型实现代码)
  * [1.卷积层](##1.卷积层)
  * [2.DFSMN层](##2.DFSMN层)
  * [3.梯度更新部分](#4.梯度更新部分)
* [模型调用方式](#模型调用方式)
* [模型训练数据](#模型训练数据)
* [已训练模型库](#已训练模型库)

## 模型实现代码
模型的实现代码位于目录：`/NlpModel/SpeechRecognition/AcousticModel/dfsmn/Model/cnn_dfsmn_ctc.py`，其实现顺序从`Am`类开始。

首先，通过调用`_model_init`对模型整个数据流图进行构建。在构建数据流图的过程中，需要依次去定义模型以下几个部分：

#### 1.卷积层
在卷积层中，根据输入的音频数据（一个4维矩阵[batch, data_len, feature_len, 1]）,对其进行卷积操作，整个卷积共分为4层，其中最后一层不使用pooling操作:
```
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
```
在卷积层的最后，通过Reshape操作，将提取出的特征转换成一个三维矩阵[batch, data_len, 3200]，以作为DFSMN层的输入。

其中，每个卷积层的实现函数是`_cnn_cell`，其代码如下，其中`norm`和`conv2d`分别为BatchNorm和卷积操作的两个函数：
```
def cnn_cell (size, x, pool=True):
	x = norm(conv2d(size)(x))
	x = norm(conv2d(size)(x))
	if pool:
		x = maxpool(x)
	return x
	
def norm (x):
	return BatchNormalization(axis=-1)(x)
	
def conv2d (size):
	return Conv2D(size, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')
```
#### 2.DFSMN层
DFSMN层的实现在代码如下，在该模型中，总共使用了6层dfsmn，其中每层dfsmn的hidden_num设置为1024，前后步长设置为40，并将dfsmn的输出经过layNorm以后，再带入激活函数swish中，最终得到下一层的输出，在每层dfsmn中，DropOut的丢失率设置为0.5。
```
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
```
其中每层dfsmn的实现代码用一个类`dfsmn_cell`进行封装，其模型所需参数和实现流程如下所示：
```
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
```
另外，除了Dfsmn以外，文件中还提供sfsmn、vfsmn、cfsmn代码实现。
#### 3.梯度更新部分
cnn_dfsmn模型在`_model_init`的数据流图构建完毕以后，首先通过调用`_ctc_init`函数，以CTC作为模型的损失函数，然后再调用`opt_init`函数选择相应优化器进行模型训练。
```
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
```
## 模型调用方式
模型的调用代码位于目录：`/NlpModel/SpeechRecognition/AcousticModel/dfsmn/Debug.py`，其调用方式主要分为以下两种。
#### 1.模型训练
cnn_dfsmn模型的训练通过调用文件中的函数`dfsmn_model_train`实现，该函数以一个参数作为输入:

(1)**train_data_path**，该参数是一个问答对组成的列表，列表中的每一个元素均为一个问答对字典；

#### 2.模型在线解码
cnn_dfsmn模型的在线解码通过调用文件中的函数`dfsmn_model_decode`实现，该函数以一个参数作为输入：

（1）**wav_file_path**，该参数是一系列需要去匹配的问题组成的列表，列表中的每一个元素是一个问题字符串；
## 模型训练数据
本模块提供的训练数据，是作为预训练模型的训练数据，主要分为以下两种，其中SameFAQ表示问题，答案指向同一句子，各问答对间的语义完全独立，可用于进行语义空间划分，SimFAQ中的问答对则是语义相近的，用于语义相似度训练，该训练数据位于目录：`/NlpModel/SimNet/TransformerDSSM/TrainData/`：

数据类型 | 数据量 | 格式
--- | --- | ---
SameFAQ | 38113 | json
SimFAQ | 20109 | json

## 已训练模型库
本模块提供三种类型已训练完毕的模型，新的问答对数据可在这三个预训练模型的基础上进行训练，能够达到较好效果，经过实验发现，效果最好的预训练模型为经过SimFAQ训练后的模型。模型的参数为：`hidden_num=256`，`attention_num=512`。其模型下载地址如下：

模型类型 | 下载地址 | 提取码
--- | --- | ---
SimFAQ | https://pan.baidu.com/s/1kff2aCsPdMQ_3wGgJaTcHA | 6qhr
SameFAQ | https://pan.baidu.com/s/1C_BfjRvwV9XNM3BZ5xy-pQ | eexz
SameFAQ+SimFAQ | https://pan.baidu.com/s/1fKh4h3H6uwlHPNh2et8SKQ | cvmn