# 基于Transformer的语义相似度计算模型DSSM
该模型在DSSM模型的基础上，将模型的表示层使用基于Transformer的Encoder部分来实现，匹配层将通过表示层得到问题query和答案answer的特征表示进行余弦相似度计算，由于问题i除了与答案i相匹配以外，其余答案均为问题i的负样本，因此需要对每一个问题进行负采样。

在该模块中，主要包含了以下4个部分内容：
* [模块内容](#模块内容)
  * [模型实现代码](#模型实现代码)
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

在得到Transformer的输出以后，由于并没有得到每个句子的特征向量表示，
#### 3.匹配层

#### 4.梯度更新部分