# 基于Transformer的语义相似度计算模型DSSM
该模型在DSSM模型的基础上，将模型的表示层使用基于Transformer的Encoder部分来实现，匹配层将通过表示层得到问题query和答案answer的特征表示进行余弦相似度计算，由于问题i除了与答案i相匹配以外，其余答案均为问题i的负样本，因此需要对每一个问题进行负采样。

在该模块中，主要包含了以下4个部分内容：
* [模块内容](#模块内容)
  * [模型实现代码](#模型实现代码)
    * [输入层](#输入层)
    * [表示层](#表示层)
    * [匹配层](#匹配层)
    * [梯度更新部分](#梯度更新部分)
  * [模型调用方式](#模型调用方式)
  * [模型训练数据](#模型训练数据)
  * [已训练模型库](#已训练模型库)

## 模型实现代码
模型的实现代码位于目录：`/NlpModel/SimNet/TransformerDSSM/Model/TransformerDSSM.py`，其实现顺序从`TransformerDSSM`类开始。

首先，通过调用`build_graph_by_cpu`或者`build_graph_by_gpu`对模型整个数据流图进行构建，以上两种构建方式，分别对应着模型的cpu版本和gpu版本。在构建数据流图的过程中，需要依次去定义模型以下几个部分：

#### 1.输入层
在输入层中，主要将输入的问题集和答案集转换成每个字符对应的字向量，最终形成一个三维矩阵t，q:
```
# 定义词向量
embeddings = tf.constant(self.vec_set)

# 将句子中的每个字转换为字向量
if not self.is_extract:
	q_embeddings = tf.nn.embedding_lookup(embeddings, self.q_inputs)
if self.is_train:
	t_embeddings = tf.nn.embedding_lookup(embeddings, self.t_inputs)
```
#### 2.表示层
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
