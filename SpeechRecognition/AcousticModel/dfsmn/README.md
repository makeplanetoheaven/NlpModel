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
#### 2.DFSMN层
表示层的实现在函数`presentation_transformer`中。

在原Transformer中，对于其输入的三维矩阵来说，为了能够引入单词在句子中的位置信息，需要在原有单词语义向量的基础上，通过规则的方式加上每个单词在句子中的位置编码向量。在本模型中，输入数据直接通过一个双向GRU来对句子中每个字的上下文信息进行编码。
```
# 正向
fw_cell = GRUCell(num_units=self.hidden_num)
fw_drop_cell = DropoutWrapper(fw_cell, output_keep_prob=self.dropout)
# 反向
bw_cell = GRUCell(num_units=self.hidden_num)
bw_drop_cell = DropoutWrapper(bw_cell, output_keep_prob=self.dropout)

# 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出是一个元组 每一个元素也是这种形状
if self.is_train and not self.is_extract:
	output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_drop_cell, cell_bw=bw_drop_cell,
	                                            inputs=inputs, sequence_length=inputs_actual_length,
	                                            dtype=tf.float32)
else:
	output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs,
	                                            sequence_length=inputs_actual_length, dtype=tf.float32)

# hiddens的长度为2，其中每一个元素代表一个方向的隐藏状态序列，将每一时刻的输出合并成一个输出
structure_output = tf.concat(output, axis=2)
structure_output = self.layer_normalization(structure_output)
```
对输入数据进行编码以后，再将其带入到Transformer的Encoder部分，进行Self-Attention，AddNorm, Full-connect计算。其实现类依次为`SelfAttention，LayNormAdd，FeedFowardNetwork`，这三个类通过类`TransformerEncoder`进行封装。

在得到Transformer的输出以后，由于并没有得到每个句子的特征向量表示，需要在其基础上引入Global-Attention，对每个句子的最终特征向量进行计算，其代码如下。
```
w_omega = tf.get_variable(name='w_omega', shape=[self.hidden_num * 2, self.attention_num],
				           initializer=tf.random_normal_initializer())
b_omega = tf.get_variable(name='b_omega', shape=[self.attention_num],
				           initializer=tf.random_normal_initializer())
u_omega = tf.get_variable(name='u_omega', shape=[self.attention_num],
				           initializer=tf.random_normal_initializer())

v = tf.tanh(tf.tensordot(transformer_output, w_omega, axes=1) + b_omega)

vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

# tf.expand_dims用于在指定维度增加一维
global_attention_output = tf.reduce_sum(transformer_output * tf.expand_dims(alphas, -1), 1)
```
#### 3.匹配层
匹配层的实现在函数`matching_layer_training`和`matching_layer_infer`中。这是由于模型在进行Tranning时需要进行负采样，而在Infer时不需要，因此需要定义两个不同的余弦相似度计算函数。
#### 4.梯度更新部分
匹配层最终的输出是一个二维矩阵，矩阵中的每一行代表一个问题与其所对应答案（第一列），及负样本的余弦相似度值。对于这样一个矩阵，经过Softmax归一化后，截取第一列数据，采用交叉熵损失计算模型最终loss，最后使用Adam优化器对模型进行训练及梯度更新。
```
# softmax归一化并输出
prob = tf.nn.softmax(cos_sim)
with tf.name_scope('Loss'):
	# 取正样本
	hit_prob = tf.slice(prob, [0, 0], [-1, 1])
	self.loss = -tf.reduce_sum(tf.log(hit_prob)) / self.batch_size

with tf.name_scope('Accuracy'):
	output_train = tf.argmax(prob, axis=1)
	self.accuracy = tf.reduce_sum(tf.cast(tf.equal(output_train, tf.zeros_like(output_train)),
	                                      dtype=tf.float32)) / self.batch_size

# 优化并进行梯度修剪
with tf.name_scope('Train'):
	optimizer = tf.train.AdamOptimizer(self.learning_rate)
	# 分解成梯度列表和变量列表
	grads, vars = zip(*optimizer.compute_gradients(self.loss))
	# 梯度修剪
	gradients, _ = tf.clip_by_global_norm(grads, 5)  # clip gradients
	# 将每个梯度以及对应变量打包
	self.train_op = optimizer.apply_gradients(zip(gradients, vars))
```
## 模型调用方式
模型的调用代码位于目录：`/NlpModel/SimNet/TransformerDSSM/Debug.py`，其调用方式主要分为以下三种。
#### 1.模型训练
TransformerDSSM模型的训练通过调用文件中的函数`dssm_model_train`实现，该函数以两个参数作为输入:

(1)**faq_dict**，该参数是一个问答对组成的列表，列表中的每一个元素均为一个问答对字典；

(2)**embedding_dict**，该参数是一个字典，字典中的每一个key是一个字符，value是该字符对应的字向量。字向量的提供位于目录：`/NlpModel/WordEmbedding/Word2Vec/CharactersEmbedding.json`
#### 2.模型推理
TransformerDSSM模型的推理通过调用文件中的函数`dssm_model_infer`实现，该函数以五个参数作为输入，需要注意的是，模型的推理返回结果，是输入答案的位置索引：

（1）**queries**，该参数是一系列需要去匹配的问题组成的列表，列表中的每一个元素是一个问题字符串；

（2）**answer_embedding**，该参数是由一系列待匹配的答案经过表示层所提取的特征向量组成的列表，列表中的每一个元素是一个答案对应的特征向量，之所以用特征向量直接作为待匹配答案的输入，是为了减少数据经过表示层的计算时间，提高匹配效率；

（3）**embedding_dict**，该参数是一个字典，字典中的每一个key是一个字符，value是该字符对应的字向量。字向量的提供位于目录：`/NlpModel/WordEmbedding/Word2Vec/CharactersEmbedding.json`

（4）**top_k**，该参数表示当输入一个问题时，需要从待匹配的答案中返回top_k个候选答案，默认时，该参数的值为1；

（4）**threshold**，该参数通过设置语义相似度计算的阈值，当待匹配的答案其相似度低于给定阈值时，则不返回，高于则返回。
#### 3.表示层特征向量提取
TransformerDSSM模型的表示层特征向量提取通过调用文件中的函数`dssm_model_extract_t_pre`实现，该函数以两个参数作为输入:

(1)**faq_dict**，该参数是一个问答对组成的列表，列表中的每一个元素均为一个问答对字典；

(2)**embedding_dict**，该参数是一个字典，字典中的每一个key是一个字符，value是该字符对应的字向量。字向量的提供位于目录：`/NlpModel/WordEmbedding/Word2Vec/CharactersEmbedding.json`
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