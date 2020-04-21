# 基于Transformer(不分段)的xlnet语言模型实现
**原论文代码**：https://github.com/zihangdai/xlnet

由于论文中的xlnet是基于Transformer-XL来解决当输入较长序列（段落或文章）时，由于计算资源的限制无法进行计算的问题。但是我觉得当对文章和段落进行编码时，可以通过分层得方式来解决相应问题，并且Transformer-XL只考虑第$i$段的前向信息，无法考虑其后向信息，因此这里参考原论文的代码来实现基于Transformer的xlnet语言模型。

在该模块中，主要包含了以下4个部分内容：
* [模型实现代码](#模型实现代码)
  * [1.数据预处理部分](##1.数据预处理部分)
  * [2.双流Attention部分](##2.双流Attention部分)
  * [3.前馈层](##3.前馈层)
  * [4.损失函数计算部分](#4.损失函数计算部分)
* [模型调用方式](#模型调用方式)
* [模型训练数据](#模型训练数据)
* [已训练模型库](#已训练模型库)

## 模型实现代码
模型的实现代码位于目录：`/NlpModel/LanguageModel/xlnet/Model/xlnet.py`，其实现顺序从`Lm`类开始。

首先，通过调用`generate_data_set`函数对模型输入数据进行预处理，接着再调用`train_cpu`或者`train_gpu`对模型数据流图进行构建并训练，其实现部分代码如下：

#### 1.数据预处理部分
数据预处理部分位于`Lm`类的`generate_data_set`函数中，在开始模型训练之前需要对其进行调用，首先对所指定的数据路径解析，对当前路径下的所有数据文件进行获取，文件中的每一行作为一条数据，然后再根据指定序列长度进行滑动窗口采样，并通过TFRecord转换成模型所需要处理的数据格式，缓存到指定文件中:
```
    def generate_data_set(self):
        """
        数据预处理函数
        :return:
        """
        assert os.path.exists(self.data_path), 'path {} does not exit'.format(self.data_path)
        tf.logging.info('begin of preprocess')

        # 1.获取文件列表
        file_list = []

        def read_file_list(data_path):
            path_list = os.listdir(data_path)
            for path in path_list:
                if '__dscache__' not in path:
                    new_path = os.path.join(data_path, path)
                    if os.path.isdir(new_path):
                        read_file_list(new_path)
                    else:
                        file_list.append(new_path)

        read_file_list(self.data_path)

        # 2.数据预处理 write tfRecord，每个样本一批数据
        def write_example(inputs):
            # 创建字典
            feature_dict = {}

            # 写入数据
            feature_dict['inputs'] = tf.train.Feature(int64_list=tf.train.Int64List(value=inputs))

            # 封装数据
            tf_features = tf.train.Features(feature=feature_dict)
            tf_example = tf.train.Example(features=tf_features)

            # 序列化数据
            tf_serialized = tf_example.SerializeToString()

            # 写入数据
            writer.write(tf_serialized)

        if not os.path.exists('{}/__dscache__/'.format(self.data_path)):
            os.mkdir('{}/__dscache__/'.format(self.data_path))
        writer = tf.python_io.TFRecordWriter(self.data_cache)
        n_data = 0
        for file in file_list:
            tf.logging.info('generate {} data'.format(file))
            with open(file, 'r', encoding='utf-8') as fo:
                for line in tqdm(fo):
                    line = line.rstrip('\n')
                    # 滑动窗口采样
                    if len(line) >= self.seq_len:
                        for i in range(len(line) - self.seq_len + 1):
                            inputs = [self.vocab_dict[char] for char in line[i:i + self.seq_len]]
                            write_example(np.array(inputs))
                            n_data += 1
        writer.close()
        tf.logging.info('the data nums is %d', n_data)
        tf.logging.info('end of preprocess')
```
#### 2.双流Attention部分
双流Attention部分根据输入的模型参数，以及上一层的内容表示$h$和问题表示$g$，对当前层的$h$和$g$进行计算。在计算过程中，分别计算$h$和$g$的评分矩阵，再根据评分矩阵对每个单词的表示进行加权得到新一轮的单词表示，其Attention计算函数如下所示：
```
def two_stream_rel_attn(h, g, r, attn_mask_h, attn_mask_g, d_head, dropout, is_training, params):
    """
    使用相对位置编码的双流Attention计算
    :param h: 内容表示
    :param g: 问题表示
    :param r: 相对位置编码向量
    :param attn_mask_h: 内容表示的mask矩阵，全为0
    :param attn_mask_g: 问题表示的mask矩阵，对角线为1
    :param d_head: 每个head的隐藏维度
    :param dropout: 丢失率
    :param is_training: 是否训练
    :param params: 模型参数
    :return: output_h, output_g
    """
    # 评分矩阵的缩放
    scale = 1 / (d_head ** 0.5)

    # 内容表示的计算
    with tf.name_scope('SelfAttn/Content'):
        # content-stream query head
        q_head_h = tf.einsum('ibh,hnd->ibnd', h, params['q_weight'])

        # content-based key head
        k_head_h = tf.einsum('ibh,hnd->ibnd', h, params['k_weight'])

        # position-based key head
        k_head_r = tf.einsum('ibh,hnd->ibnd', r, params['k_weight'])

        # content-based value head
        v_head_h = tf.einsum('ibh,hnd->ibnd', h, params['v_weight'])

        # core attention ops
        # attn_vec_h = rel_attn_core(q_head_h, k_head_h, k_head_r, v_head_h, attn_mask_h,
        #                            dropout, is_training, scale, params['q_w_bias'], params['q_r_bias'])
        attn_vec_h = rel_attn_core(q_head_h, k_head_h, k_head_r, v_head_h, attn_mask_h, dropout, is_training, scale)

        # post-attention projection (back to `d_model`)
        attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec_h, params['o_weight'])
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)

        # add + lay_norm
        output_h = params['lay_norm'](attn_out + h)

    # 问题表示的计算
    with tf.name_scope('SelfAttn/Query'):
        # query-stream query head
        q_head_g = tf.einsum('ibh,hnd->ibnd', g, params['q_weight'])

        # core attention ops
        attn_vec_g = rel_attn_core(q_head_g, k_head_h, k_head_r, v_head_h, attn_mask_g, dropout, is_training, scale,
                                   params['q_w_bias'], params['q_r_bias'])
        # attn_vec_g = rel_attn_core(q_head_g, k_head_h, k_head_r, v_head_h, attn_mask_g, dropout, is_training, scale)

        # post-attention projection (back to `d_model`)
        attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec_g, params['o_weight'])
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)

        # add + lay_norm
        output_g = params['lay_norm'](attn_out + h)

    return output_h, output_g
```
其中分别对$h$和$g$的评分矩阵进行计算的函数如下所示：
```
def rel_attn_core(q_head, k_head_h, k_head_r, v_head_h, attn_mask, dropout, is_training, scale, q_w_bias=None,
                  q_r_bias=None):
    """
    self-attention的计算，包括评分矩阵的计算和每个字符最终表示的计算
    :param q_head: Q映射
    :param k_head_h: K映射（语义向量）
    :param k_head_r: K映射（位置向量）
    :param v_head_h: V映射
    :param attn_mask: 评分矩阵的mask
    :param dropout: 丢失率
    :param is_training: 是否训练
    :param scale: 评分矩阵的缩放值
    :param q_w_bias: 语义向量的偏差
    :param q_r_bias: 相对位置编码向量的偏差
    :return: attn_vec
    """
    # content based attention score
    if q_w_bias is not None:
        ac = tf.einsum('ibnd,jbnd->ijbn', q_head + q_w_bias, k_head_h)
    else:
        ac = tf.einsum('ibnd,jbnd->ijbn', q_head, k_head_h)

    # position based attention score
    if q_r_bias is not None:
        bd = tf.einsum('ibnd,jbnd->ijbn', q_head + q_r_bias, k_head_r)
    else:
        bd = tf.einsum('ibnd,jbnd->ijbn', q_head, k_head_r)
    bd = rel_shift(bd, klen=tf.shape(ac)[1])

    # merge attention scores and perform masking
    attn_score = (ac + bd) * scale
    if attn_mask is not None:
        attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = tf.nn.softmax(attn_score, 1)
    attn_prob = tf.layers.dropout(attn_prob, dropout, training=is_training)

    # attention output
    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

    return attn_vec
```
#### 3.前馈层
前馈神经网络层根据双流Attention得到的输出，分别对$h$和$g$进行计算，其函数如下：
```
def positionwise_ffn(input, dropout, params, is_training=True):
    """
    Position-wise Feed-forward Network.
    :param input:
    :param dropout:
    :param params:
    :param is_training:
    :return:
    """
    output = input
    with tf.name_scope('FeedForward'):
        output = tf.layers.dropout(params['ff_1'](output), dropout, training=is_training)
        output = tf.layers.dropout(params['ff_2'](output), dropout, training=is_training)
        output = params['lay_norm'](output + input)
    return output
```
#### 4.损失函数计算部分
最终，将最后一层的问题表示$g$先通过一个全连接层来得到每一个类别的logit，再带入到交叉熵损失函数中，其中该模型的标签即为模型的输入，其实现函数如下：
```
    def build_loss(self, inputs, labels):
        """
        模型损失函数的构建
        :return: loss
        """
        with tf.variable_scope('lm_loss'):
            softmax_w = tf.get_variable('weight', [self.n_token, self.d_model], dtype=tf.float32,
                                        initializer=self.kernel_initializer)
            softmax_b = tf.get_variable('bias', [self.n_token], dtype=tf.float32, initializer=self.bias_initializer)

            logits = tf.einsum('ibd,nd->ibn', inputs, softmax_w) + softmax_b

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)

        return loss
```
## 模型调用方式
模型的调用代码位于目录：`/NlpModel/LanguageModel/xlnet/Debug.py`，其调用方式仅包含模型训练部分的函数，由于xlnet语言模型通常需要配合其他模型进行联合建模使用，因此不提供predict函数的实现。
#### 1.模型训练
xlnet模型的训练通过调用文件中的函数`xlnet_model_train`实现，该函数以一个参数作为输入:

(1)**data_path**，该参数指定训练数据所在路径；

(2)**save_path**，该参数指定模型存储所在路径；

## 模型训练数据
本模块提供的训练数据，是作为预训练模型的训练数据，数据量大概有7000W条左右，其中每个文件的一行为一条数据，一个文件10W条数据，数据地址链接如下：

数据类型 | 格式 | 地址 | 提取码
--- | ---| ---| ---
中文预训练数据 | txt | [https://pan.baidu.com/s/1sTYdq-Id07hd1SJLBMBSVw](https://pan.baidu.com/s/1sTYdq-Id07hd1SJLBMBSVw) | sx0q

## 已训练模型库

task_name| 参数| 地址 | 提取码
--- | ---| ---| ---
xlnet | seq_len = 512,stride = 256,d_embed=768,d_model = 768,n_layers = 12,n_head = 12,d_head = 64 | [点击](https://pan.baidu.com/s/1MGkknyD7f1qvpmsYTNtuaw) | 0imo