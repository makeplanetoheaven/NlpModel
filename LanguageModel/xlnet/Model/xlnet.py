# coding=utf-8

import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)


# =============================模型超参数====================================
def lm_hparams():
    params = tf.contrib.training.HParams(data_path='', vocab_dict=None, bsz=64, seq_len=7, epoch=10,
                                         max_step=1000, lr=0.0008,
                                         dropout=0.5, d_embed=768, d_model=768, n_layers=12, n_head=12, d_head=512,
                                         init_range=1, init_std=0, is_training=True, task_name=None, save_path=None)

    return params


# =============================模型框架====================================
class Lm:
    def __init__(self, args):
        # 数据参数
        self.data_path = args.data_path
        self.data_cache = '{}/__dscache__/{}.tfrecord'.format(args.data_path,
                                                              args.task_name) if args.data_path is not None else None
        self.vocab_dict = args.vocab_dict
        self.n_token = len(self.vocab_dict)

        # 超参数
        self.bsz = args.bsz
        self.seq_len = args.seq_len
        self.epoch = args.epoch
        self.max_step = args.max_step
        self.lr = args.lr
        self.dropout = args.dropout
        self.d_embed = args.d_embed
        self.d_model = args.d_model
        self.n_layers = args.n_layers
        self.n_head = args.n_head
        self.d_head = args.d_head
        self.init_range = args.init_range
        self.init_std = args.init_std
        self.is_training = args.is_training

        # 存储参数
        self.task_name = args.task_name
        self.model_save_name = args.save_path + 'xlnet'
        self.model_save_checkpoint = args.save_path + 'checkpoint'

        # 模型参数
        self.build_initializer()
        self.build_activation()
        self.build_opt()
        self.build_parameters()

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

    def input_fn(self):
        """
        模型输入函数
        :return:
        """
        assert os.path.exists(
            self.data_cache), 'file {} does not exit, please use generate_data_set function first'.format(
            self.data_cache)

        # 数据对象构建 read tfRecord
        def parser(example):
            example_dict = {
                'inputs': tf.FixedLenFeature(shape=(self.seq_len,), dtype=tf.int64)
            }
            parsed_example = tf.parse_single_example(example, example_dict)

            return parsed_example

        file_names = [self.data_cache]
        ds = tf.data.TFRecordDataset(file_names)
        ds = ds.cache().map(parser).repeat(self.epoch)
        ds = ds.shuffle(buffer_size=self.bsz)  # example-level shuffle
        ds = ds.batch(self.bsz, drop_remainder=True)
        ds = ds.prefetch(self.bsz)

        return ds

    def build_initializer(self, kernel_name='xavier', bias_name='zeros'):
        """
        模型参数初始化器的选择
        :param kernel_name: 权重初始化名字
        :param bias_name: 偏差初始化名字
        :return: None
        """
        if kernel_name == "uniform":
            self.kernel_initializer = tf.initializers.random_uniform(minval=-self.init_range, maxval=self.init_range)
        elif kernel_name == "normal":
            self.kernel_initializer = tf.initializers.random_normal(stddev=self.init_std)
        elif kernel_name == 'xavier':
            self.kernel_initializer = tf.contrib.layers.xavier_initializer()
        elif kernel_name == 'he_uni':
            self.kernel_initializer = tf.initializers.he_uniform()
        elif kernel_name == 'he_nor':
            self.kernel_initializer = tf.initializers.he_normal()
        else:
            raise ValueError("kernel initializer {} not supported".format(kernel_name))

        if bias_name == "zeros":
            self.bias_initializer = tf.zeros_initializer()
        elif bias_name == "ones":
            self.bias_initializer = tf.ones_initializer()
        else:
            raise ValueError("bias initializer {} not supported".format(bias_name))

    def build_activation(self, name='swish'):
        """
        模型激活函数选择
        :param name: 激活函数名字
        :return: None
        """
        if name == 'relu':
            self.activation = (name, tf.nn.relu)
        elif name == 'gelu':
            self.activation = (name, tf.nn.gelu)
        elif name == 'swish':
            self.activation = (name, tf.nn.swish)
        elif name == 'sigmoid':
            self.activation = (name, tf.nn.sigmoid)
        elif name == 'tanh':
            self.activation = (name, tf.nn.tanh)
        else:
            raise ValueError("activation {} not supported".format(name))

    def build_parameters(self):
        """
        模型参数构建
        :return: None
        """
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('word_embedding', reuse=tf.AUTO_REUSE):
                self.embedding_table = tf.get_variable('embedding_table', [self.n_token, self.d_embed],
                                                       dtype=tf.float32,
                                                       initializer=self.kernel_initializer)

            with tf.variable_scope('xlnet_layers', reuse=tf.AUTO_REUSE):
                self.rel_attn = []
                self.ff = []
                for i in range(self.n_layers):
                    with tf.variable_scope('layer_{}'.format(i), reuse=tf.AUTO_REUSE):
                        with tf.variable_scope('rel_attn'):
                            layer_para = {}
                            layer_para['q_weight'] = tf.get_variable('q/kernel',
                                                                     [self.d_model, self.n_head, self.d_head],
                                                                     dtype=tf.float32,
                                                                     initializer=self.kernel_initializer)
                            layer_para['q_w_bias'] = tf.get_variable('q/w_bias', [self.n_head, self.d_head],
                                                                     dtype=tf.float32,
                                                                     initializer=self.bias_initializer)
                            layer_para['q_r_bias'] = tf.get_variable('q/r_bias', [self.n_head, self.d_head],
                                                                     dtype=tf.float32,
                                                                     initializer=self.bias_initializer)
                            layer_para['k_weight'] = tf.get_variable('k/kernel',
                                                                     [self.d_model, self.n_head, self.d_head],
                                                                     dtype=tf.float32,
                                                                     initializer=self.kernel_initializer)
                            layer_para['v_weight'] = tf.get_variable('v/kernel',
                                                                     [self.d_model, self.n_head, self.d_head],
                                                                     dtype=tf.float32,
                                                                     initializer=self.kernel_initializer)
                            layer_para['o_weight'] = tf.get_variable('o/kernel',
                                                                     [self.d_model, self.n_head, self.d_head],
                                                                     dtype=tf.float32,
                                                                     initializer=self.kernel_initializer)
                            layer_para['lay_norm'] = LayerNormalization('lay_norm', self.d_model,
                                                                        kernel_initializer=self.kernel_initializer,
                                                                        bias_initializer=self.bias_initializer)
                            self.rel_attn.append(layer_para)

                        with tf.variable_scope('ff', reuse=tf.AUTO_REUSE):
                            layer_para = {}
                            layer_para['ff_1'] = tf.layers.Dense(self.d_model, activation=self.activation[1],
                                                                 use_bias=True,
                                                                 kernel_initializer=self.kernel_initializer,
                                                                 name='ff_1')
                            layer_para['ff_2'] = tf.layers.Dense(self.d_model, activation=self.activation[1],
                                                                 use_bias=True,
                                                                 kernel_initializer=self.kernel_initializer,
                                                                 name='ff_2')
                            layer_para['lay_norm'] = LayerNormalization('lay_norm', self.d_model,
                                                                        kernel_initializer=self.kernel_initializer,
                                                                        bias_initializer=self.bias_initializer)
                            self.ff.append(layer_para)

    def build_model(self, inputs):
        """
        构建模型训练所需的数据流图
        :return: output_h, output_g
        """
        with tf.name_scope('InputLayer'):
            # 获取输入数据的长度和数目
            qlen = tf.shape(inputs)[0]
            bsz = tf.shape(inputs)[1]

            # 将输入的每个字符转成语义向量
            pad = tf.zeros(shape=(1, self.d_embed))
            temp_embedding_table = tf.concat([self.embedding_table, pad], axis=0)
            input_embedding = tf.nn.embedding_lookup(temp_embedding_table, inputs)
            output_h = tf.layers.dropout(input_embedding, self.dropout, training=self.is_training)
            output_g = tf.layers.dropout(input_embedding, self.dropout, training=self.is_training)

            # 根据输入数据长度和隐藏维度定义相对位置编码向量
            pos_emb = relative_positional_encoding(qlen, self.d_model, bsz)
            pos_emb = tf.layers.dropout(pos_emb, self.dropout, training=self.is_training)

        with tf.name_scope('AttentionMask'):
            attn_mask_g = tf.eye(qlen, dtype=tf.float32)
            attn_mask_g = attn_mask_g[:, :, None, None]

        with tf.name_scope('XLNet'):
            for i in range(self.n_layers):
                with tf.name_scope('Layer{}'.format(i)):
                    # 双流Attention层
                    output_h, output_g = two_stream_rel_attn(h=output_h, g=output_g, r=pos_emb, attn_mask_h=None,
                                                             attn_mask_g=attn_mask_g, d_head=self.d_head,
                                                             dropout=self.dropout, is_training=self.is_training,
                                                             params=self.rel_attn[i])

                    # 前馈层
                    output_h = positionwise_ffn(input=output_h, dropout=self.dropout, params=self.ff[i],
                                                is_training=self.is_training)

                    output_g = positionwise_ffn(input=output_g, dropout=self.dropout, params=self.ff[i],
                                                is_training=self.is_training)

        return output_h, output_g

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

    def build_opt(self, name='adam'):
        """
        模型优化器的选择
        :param name: 优化器名字
        :return: None
        """
        if name == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(self.lr)
        elif name == 'momentum':
            self.opt = tf.train.MomentumOptimizer(self.lr)
        elif name == 'rms':
            self.opt = tf.train.RMSPropOptimizer(self.lr)
        elif name == 'adagrad':
            self.opt = tf.train.AdagradOptimizer(self.lr)
        elif name == 'adam':
            self.opt = tf.train.AdamOptimizer(self.lr)
        else:
            raise ValueError("opt {} not supported".format(name))

    def train_cpu(self):
        """
        模型cpu训练
        :return: None
        """
        # 1.数据获取
        ele = self.input_fn().make_one_shot_iterator().get_next()
        inputs = tf.transpose(ele['inputs'], [1, 0])

        # 2.构建数据流图
        # 模型的定义
        h, g = self.build_model(inputs)

        # 损失的定义
        loss = self.build_loss(g, inputs)

        # 优化器的定义
        # 分解成梯度列表和变量列表
        grads, vars = zip(*self.opt.compute_gradients(loss))
        # 梯度修剪
        gradients, _ = tf.clip_by_global_norm(grads, 5)
        # 将每个梯度以及对应变量打包
        train_op = self.opt.apply_gradients(zip(gradients, vars))

        # 3.模型训练
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 判断模型是否存在
            if os.path.exists(self.model_save_checkpoint):
                # 恢复变量
                saver.restore(sess, self.model_save_name)
            else:
                # 初始化变量
                sess.run(tf.global_variables_initializer())

            # 开始迭代，使用Adam优化的随机梯度下降法，并将结果输出到日志文件
            tf.logging.info('begin of training')
            fetches = [train_op, loss]
            total_loss = 0.
            cur_step = 0
            while 1:
                try:
                    _, loss_np = sess.run(fetches)
                    total_loss += loss_np
                    cur_step += 1

                    if cur_step > 0 and cur_step % self.max_step == 0:
                        cur_loss = total_loss / self.max_step
                        tf.logging.info('[%s] [step %d] loss %f', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                        cur_step, cur_loss)
                        total_loss = 0.
                except tf.errors.OutOfRangeError:
                    tf.logging.info('end of training')
                    break

            # 保存模型
            tf.logging.info('save model')
            saver.save(sess, self.model_save_name)

        pass

    def train_gpu(self, gpu_nums=1):
        """
        模型gpu训练
        :param gpu_nums: gpu数目
        :return: None
        """
        # 1.数据获取
        ele = self.input_fn().make_one_shot_iterator().get_next()
        inputs = tf.transpose(ele['inputs'], [1, 0])
        bsz_per_gpu = self.bsz // gpu_nums

        # 2.构建数据流图
        # 多GPU数据流图构建
        tower_grads, tower_losses = [], []
        for i in range(gpu_nums):
            reuse = True if i > 0 else None
            with tf.device("/gpu:%d" % i), tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                # 数据的分割
                inputs_i = inputs[:, i * bsz_per_gpu:(i + 1) * bsz_per_gpu]

                # 模型的定义
                h_i, g_i = self.build_model(inputs_i)

                # 损失的定义
                loss_i = self.build_loss(g_i, inputs_i)

                # 优化器的定义
                # 分解成梯度列表和变量列表
                grads, vars = zip(*self.opt.compute_gradients(loss_i))
                # 梯度修剪
                gradients, _ = tf.clip_by_global_norm(grads, 5)

                tower_grads.append(zip(gradients, vars))
                tower_losses.append(loss_i)

        loss = tf.reduce_mean(tower_losses, 0)
        grads = average_gradients(tower_grads)
        train_op = self.opt.apply_gradients(grads)

        # 3.模型训练
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # 判断模型是否存在
            if os.path.exists(self.model_save_checkpoint):
                # 恢复变量
                saver.restore(sess, self.model_save_name)
            else:
                # 初始化变量
                sess.run(tf.global_variables_initializer())

            # 开始迭代，使用Adam优化的随机梯度下降法，并将结果输出到日志文件
            tf.logging.info('begin of training')
            fetches = [train_op, loss]
            total_loss = 0.
            cur_step = 0
            while 1:
                try:
                    _, loss_np = sess.run(fetches)
                    total_loss += loss_np
                    cur_step += 1

                    if cur_step > 0 and cur_step % self.max_step == 0:
                        cur_loss = total_loss / self.max_step
                        tf.logging.info('[%s] [step %d] loss %f', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                        cur_step, cur_loss)
                        total_loss = 0.
                except tf.errors.OutOfRangeError:
                    tf.logging.info('end of training')
                    break

            # 保存模型
            tf.logging.info('save model')
            saver.save(sess, self.model_save_name)

        pass


# =============================模型组件====================================
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
        attn_vec_h = rel_attn_core(q_head_h, k_head_h, k_head_r, v_head_h, attn_mask_h,
                                   dropout, is_training, scale, params['q_w_bias'], params['q_r_bias'])
        # attn_vec_h = rel_attn_core(q_head_h, k_head_h, k_head_r, v_head_h, attn_mask_h, dropout, is_training, scale)

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


def relative_positional_encoding(qlen, d_model, bsz=None):
    """
    create relative positional encoding.
    :param qlen:
    :param d_model:
    :param bsz:
    :return:
    """
    inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
    pos_seq = tf.range(qlen, -1, -1.0)
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    pos_emb = pos_emb[:, None, :]

    if bsz is not None:
        pos_emb = tf.tile(pos_emb, [1, bsz, 1])

    return pos_emb


def rel_shift(x, klen=-1):
    """
    perform relative shift to form the relative attention score.
    :param x:
    :param klen:
    :return:
    """
    x_size = tf.shape(x)

    x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
    x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

    return x


class LayerNormalization(object):
    def __init__(self, name, d_model, kernel_initializer, bias_initializer):
        super(LayerNormalization, self).__init__()
        self.__name__ = name
        self.d_model = d_model
        self.build(kernel_initializer, bias_initializer)

    def build(self, kernel_initializer, bias_initializer):
        self.scale = tf.get_variable("layer_norm_scale", [self.d_model], initializer=kernel_initializer)
        self.bias = tf.get_variable("layer_norm_bias", [self.d_model], initializer=bias_initializer)
        self.built = True

    def __call__(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
