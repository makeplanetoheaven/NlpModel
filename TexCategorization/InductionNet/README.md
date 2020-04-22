# 背景知识
本篇文章所实现的基于xlnet和归纳网络的小样本文本分类模型主要用于问答系统中的意图识别部分，与传统的文本分类任务相比，该模型能够实现对问答系统现有意图（即功能）的动态增加和配置，并且只需要少量数据即能实现较好效果。

xlnet语言模型和用于小样本文本分类的归纳网络的模型的结构及其实现原理可参考如下几篇博客：

1.[XLNet: Generalized Autoregressive Pretraining for Language Understanding翻译](https://blog.csdn.net/qq_28385535/article/details/93608790)

2.[语言模型|基于Transformer(不分段)的xlnet语言模型实现及代码开源](https://blog.csdn.net/qq_28385535/article/details/103111740)

3.[Induction Networks for Few-Shot Text Classification翻译](https://blog.csdn.net/qq_28385535/article/details/102947493)

# 基于xlnet和归纳网络的小样本文本分类模型实现
本模型是参考2019年论文Induction Networks for Few-Shot Text Classification实现，其中以预训练完毕的xlnet语言模型作为编码器模块，来实现问答系统中的意图识别功能。

需要注意的是，模型在训练时，训练数据中的一个类别数据是存储在同一个`.txt`文件中，文件名已标签名进行命名；模型在使用时，只需构建一个`.json`文件并传入调用函数中使用，详细方法见下面所诉，`.json`文件格式为：

```
{
	"类别1": [
		"句子1",
		"句子2",
		...
	 ],
	 ...
}
```

在模型实现中，主要包含了一下几个部分内容：
* [模型实现代码](#模型实现代码)
  * [1.编码器模块](##1.编码器模块)
  * [2.归纳模块](##2.归纳模块)
  * [3.关系模块](##3.关系模块)
  * [4.损失函数计算部分](#4.损失函数计算部分)
* [模型调用方式](#模型调用方式)
* [模型训练数据](#模型训练数据)
* [已训练模型库](#已训练模型库)
# 模型实现代码
## 1.编码器模块
编码器模块主要分为两个部分，其中一个部分通过xlnet语言模型对输入的句子进行特征提取，得到句子中每个单词对应的表示，该部分的实现代码可参见[语言模型|基于Transformer(不分段)的xlnet语言模型实现及代码开源](https://blog.csdn.net/qq_28385535/article/details/103111740)。第二部分通过global_attention模块单词向量进行编码从而得到句子级表示，其部分代码实现如下所示：

```python
 # 评分矩阵的缩放
scale = 1 / (self.d_head ** 0.5)

with tf.name_scope('SenVec_s'):
	# 评分矩阵计算
	s_k_h = tf.einsum('ibh,hnd->ibnd', s, self.k_weight) + self.k_bias
	s_v_h = tf.einsum('ibh,hnd->ibnd', s, self.v_weight)

	s_attn_score = tf.einsum('nd,ibnd->ibn', self.q_h, s_k_h) * scale
	s_attn_prob = tf.nn.softmax(s_attn_score, 0)

	# 句子向量计算
	s_attn_vec = tf.einsum('ibn,jbnd->bnd', s_attn_prob, s_v_h)
	e_s = tf.einsum('bnd,hnd->bh', s_attn_vec, self.o_weight)
	e_s = tf.layers.dropout(e_s, self.dropout, training=self.is_training)
```
问题和待匹配数据通过编码器模块后即可分别得到问题表示`e_q`和数据表示`e_s`。
## 2.归纳模块
归纳模块通过胶囊网络中的非线性压缩变换和动态路由，将同一类别的待匹配数据进行归纳表示，得到用于表示该类别特征的向量，归纳模块的部分代码实现如下：

```python
with tf.name_scope('Induction'):
	e_s_hatt = squash(tf.matmul(e_s, self.s_weight) + self.s_bias)
	e_s_hatt = tf.layers.dropout(e_s_hatt, self.dropout, training=self.is_training)

	# shape trans: [b, d] => [c, b, d]
	e_s_hatt = tf.reshape(e_s_hatt, shape=[self.c_way, self.k_shot, self.d_input])

	# coupling coefficients define
	b = tf.zeros(shape=[self.c_way, self.k_shot])

	# dynamic routing
	for i in range(self.n_route):
		d = tf.nn.softmax(b)
		c_hatt = tf.reduce_sum(e_s_hatt * tf.expand_dims(d, -1), axis=1)
		c = squash(c_hatt)  # [c_way, d_input]
		b = b + tf.einsum('csd,cd->cs', e_s_hatt, c)

		d = tf.nn.softmax(b)
		c_hatt = tf.reduce_sum(e_s_hatt * tf.expand_dims(d, -1), axis=1)
		c = squash(c_hatt)  # [c_way, d_input]
		c = tf.layers.dropout(c, self.dropout, training=self.is_training)
```
最终得到得变量`c`即为表示类别特征的二维矩阵`[c_way, d_model]`。
## 3.关系模块
关系模块通过神经张量网络以及输出维度为`1`的全连接层对问题`e_q`和类别矩阵`c`进行相似度计算，从而得到每一个问题相对于各类别的评分，其代码实现如下：

```python
with tf.name_scope('Relation'):
	v = tf.einsum('ijn,cj->icn', self.m_weight, c)
	v = self.activation[1](tf.einsum('qi,icn->qcn', e_q, v))
	r = self.ffn(v)  # [q_nums, c_way, 1]
	logits = tf.reshape(r, [tf.shape(r)[0], tf.shape(r)[1]])
```
最终得到的`logits`即为维度为`[q_nums, c_way]`的二维矩阵。
## 4.损失函数
损失函数部分的实现与论文稍有不同，论文中使用的是多标签分类损失，而代码中使用的是单标签分类损失，其代码实现如下：

```python
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
loss = tf.reduce_mean(loss)
```
其中，标签`labels`是一个维度为[q_nums, 1]的二维矩阵。
# 模型调用方式
模型的调用代码位于目录：`./TexCategorization/InductionNet/Debug.py`，其调用方式主要分为以下两种。

## 1.模型训练
Induction Net模型的训练通过调用文件中的函数`induction_net_model_train`实现，该函数以三个参数作为输入:

（1）**data_path**，该参数指定训练数据所在目录路径，其中一个文件存储一个类别数据，文件名即为类名；

（2）**lm_save_path**，该参数指定xlnet语言模型存储参数所在目录路径；

（3）**save_path**，该参数指定归纳网络模型的存储路径；

## 2.模型预测
Induction Net模型的预测通过调用文件中的函数induction_net_model_predict实现，该函数以四个参数作为输入：

（1）**q_list**，该参数存储了需要进行分类的问题列表，列表中每一个元素为一个问题字符串；

（2）**label_dict**，该参数存储了待匹配的数据字典，字典中每一个key为一个类别，每一个value为一个存储该类别数据的列表；

（1）**lm_save_path**，该参数指定xlnet语言模型存储参数所在目录路径；

（2）**save_path**，该参数指定归纳网络模型的存储路径；
# 模型训练数据
归纳网络的模型训练数据可使用大部分的文本分类数据进行训练，因此这里不再提供。