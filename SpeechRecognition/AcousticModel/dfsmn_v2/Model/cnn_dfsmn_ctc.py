# coding=utf-8

import os
import time

import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
from scipy.fftpack import fft
from tensorflow.contrib.keras import backend as tfk

tf.logging.set_verbosity(tf.logging.INFO)


# =============================模型超参数====================================
def am_hparams():
    params = tf.contrib.training.HParams(data_path=None, label_path=None, thchs30=False, aishell=False, prime=False,
                                         stcmd=False, vocab_dict=None, bsz=1, epoch=1, lr=1e-4, dropout=0.5,
                                         d_input=384, d_model=1024, l_mem=20, r_mem=20, stride=2, n_init_filters=32,
                                         n_conv=2, n_cnn_layers=4, n_dfsmn_layers=6, init_range=1, init_std=0,
                                         is_training=False, save_path=None)

    return params


# =============================模型框架====================================
class Am:
    def __init__(self, args):
        # 数据参数
        self.data_path = args.data_path
        self.label_path = args.label_path
        self.thchs30 = args.thchs30
        self.aishell = args.aishell
        self.prime = args.prime
        self.stcmd = args.stcmd
        self.vocab_dict = args.vocab_dict
        self.n_vocab = len(self.vocab_dict)

        # 超参数
        self.bsz = args.bsz
        self.epoch = args.epoch
        self.lr = args.lr
        self.dropout = args.dropout
        self.d_input = args.d_input
        self.d_model = args.d_model
        self.l_mem = args.l_mem
        self.r_mem = args.r_mem
        self.stride = args.stride
        self.n_init_filters = args.n_init_filters
        self.n_conv = args.n_conv
        self.n_cnn_layers = args.n_cnn_layers
        self.n_dfsmn_layers = args.n_dfsmn_layers
        self.init_range = args.init_range
        self.init_std = args.init_std
        self.is_training = args.is_training

        # 模型参数
        self.build_initializer()
        self.build_activation()
        self.build_opt()
        self.build_parameters()

        # 存储路径
        self.model_save_name = args.save_path + 'cnn_dfsmn_ctc'
        self.model_save_checkpoint = args.save_path + 'checkpoint'

    def generate_data_set(self):
        # 1.获取文件数据和标签列表
        file_list = []
        label_list = []

        def read_file_list(data_path, label_path):
            tf.logging.info('get source list...')
            read_files = []
            if self.thchs30 == True:
                read_files.append(label_path + 'thchs_train.txt')
            if self.aishell == True:
                read_files.append(label_path + 'aishell_train.txt')
            if self.prime == True:
                read_files.append(label_path + 'prime.txt')
            if self.stcmd == True:
                read_files.append(label_path + 'stcmd.txt')

            for file in read_files:
                tf.logging.info('load %s data...', file)
                with open(file, 'r', encoding='utf8') as fo:
                    for line in fo:
                        wav_file, label, _ = line.split('\t')
                        file_list.append(data_path + wav_file)
                        label_list.append(label.split(' '))

        read_file_list(self.data_path, self.label_path)

        # batch生成器定义
        def am_batch():
            index_list = [i for i in range(len(file_list))]
            wav_data_list = []
            label_data_list = []
            while 1:
                for i in index_list:
                    # 数据特征提取
                    fbank = compute_fbank(file_list[i])

                    # 保证卷积使得长度变为1/8仍有效
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
                    pad_fbank[:fbank.shape[0], :] = fbank

                    # 标签转ID
                    label = [self.vocab_dict.get(pny, 0) for pny in label_list[i]]

                    # 判断音频数据长度经过卷积后是否大于标签数据长度
                    label_ctc_len = ctc_len(label)
                    if pad_fbank.shape[0] // 8 >= label_ctc_len:
                        wav_data_list.append(pad_fbank)
                        label_data_list.append(label)

                    # batch获取
                    if len(wav_data_list) >= self.bsz:
                        wav_data = wav_data_list[:self.bsz]
                        wav_data_list = wav_data_list[self.bsz:]
                        label_data = label_data_list[:self.bsz]
                        label_data_list = label_data_list[self.bsz:]

                        # 长度补全
                        pad_wav_data, input_length = wav_padding(wav_data)
                        pad_label_data, label_length = label_padding(label_data)

                        yield pad_wav_data, input_length, pad_label_data, label_length

                yield None

        return am_batch()

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
            raise ValueError("kernel initializer {} not supported".format(name))

        if bias_name == "zeros":
            self.bias_initializer = tf.zeros_initializer()
        elif bias_name == "ones":
            self.bias_initializer = tf.ones_initializer()
        else:
            raise ValueError("bias initializer {} not supported".format(name))

    def build_activation(self, name='relu'):
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
        elif name == 'sigmod':
            self.activation = (name, tf.nn.sigmod)
        elif name == 'tanh':
            self.activation = (name, tf.nn.tanh)
        else:
            raise ValueError("activation {} not supported".format(name))

    def build_parameters(self):
        """
        模型参数构建
        :return: None
        """
        with tf.variable_scope('cnn_layers', reuse=tf.AUTO_REUSE):
            self.cnn_cells = []
            filters = self.n_init_filters
            for i in range(self.n_cnn_layers):
                with tf.variable_scope('layer_{}'.format(i), reuse=tf.AUTO_REUSE):
                    if i < self.n_cnn_layers - 1:
                        cell = Cnn2d_cell(filters=filters, kernel_size=(3, 3), padding='same',
                                          activation=self.activation[0], use_bias=True, kernel=self.kernel_initializer,
                                          bias=self.bias_initializer, n_conv=self.n_conv, is_pool=True)
                        filters = filters * 2
                    else:
                        cell = Cnn2d_cell(filters=filters // 2, kernel_size=(3, 3), padding='same',
                                          activation=self.activation[0], use_bias=True, kernel=self.kernel_initializer,
                                          bias=self.bias_initializer, n_conv=self.n_conv, is_pool=False)

                self.cnn_cells.append(cell)

        with tf.variable_scope('liner_transformer_layers', reuse=tf.AUTO_REUSE):
            self.liner_transformer = tf.layers.Dense(self.d_input, activation=self.activation[1], use_bias=True,
                                                     kernel_initializer=self.kernel_initializer,
                                                     bias_initializer=self.bias_initializer)

        with tf.variable_scope('dfsmn_layers', reuse=tf.AUTO_REUSE):
            self.dfsmn_cells = []
            for i in range(self.n_dfsmn_layers):
                with tf.variable_scope('layer_{}'.format(i), reuse=tf.AUTO_REUSE):
                    if i < 1:
                        cell = cfsmn_cell('cfsmn-cell', self.d_input, self.d_input, self.d_model, self.l_mem,
                                          self.r_mem, self.stride, self.kernel_initializer, self.bias_initializer)
                    else:
                        cell = dfsmn_cell('dfsmn-cell', self.d_input, self.d_input, self.d_model, self.l_mem,
                                          self.r_mem, self.stride, self.kernel_initializer, self.bias_initializer)

                    self.dfsmn_cells.append(cell)

        with tf.variable_scope('softmax_layers', reuse=tf.AUTO_REUSE):
            self.softmax_layer = tf.layers.Dense(self.n_vocab, activation=tf.nn.softmax, use_bias=True,
                                                 kernel_initializer=self.kernel_initializer,
                                                 bias_initializer=self.bias_initializer)

    def build_model(self, inputs):
        """
        构建模型训练所需的数据流图
        :return: output
        """
        outputs = inputs
        bsz = tf.shape(outputs)[0]
        with tf.name_scope('CNNLayer'):
            for i in range(self.n_cnn_layers):
                with tf.name_scope('Layer{}'.format(i)):
                    outputs = self.cnn_cells[i](outputs)
                    outputs = tf.layers.dropout(outputs, self.dropout, training=self.is_training)

        with tf.name_scope('LinerTransformer'):
            outputs = tf.reshape(outputs, shape=(bsz, -1, 3200))
            outputs = self.liner_transformer(outputs)
            outputs = tf.layers.dropout(outputs, self.dropout, training=self.is_training)

        with tf.name_scope('DFSMN'):
            for i in range(self.n_dfsmn_layers):
                with tf.name_scope('Layer{}'.format(i)):
                    if i == 0:
                        outputs, p_hatt = self.dfsmn_cells[i](outputs)
                    else:
                        outputs, p_hatt = self.dfsmn_cells[i]((outputs, p_hatt))
                    outputs = self.activation[1](outputs)
                    outputs = tf.layers.dropout(outputs, self.dropout, training=self.is_training)

        with tf.name_scope('SoftmaxLayer'):
            outputs = self.softmax_layer(outputs)

        return outputs

    def build_loss(self, inputs, inputs_length, labels, labels_length):
        """
        模型损失函数的构建
        :return: loss
        """
        with tf.name_scope('AmLoss'):
            loss = tfk.ctc_batch_cost(y_true=labels, y_pred=inputs, input_length=inputs_length,
                                      label_length=labels_length)
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
        # 1.训练数据获取
        am_batch = self.generate_data_set()

        # 2.构建数据流图
        # 模型输入的定义
        inputs = tf.placeholder(name='the_inputs', shape=[None, None, 200, 1], dtype=tf.float32)
        inputs_length = tf.placeholder(name='inputs_length', shape=[None, 1], dtype=tf.int32)
        labels = tf.placeholder(name='the_labels', shape=[None, None], dtype=tf.float32)
        labels_length = tf.placeholder(name='labels_length', shape=[None, 1], dtype=tf.int32)

        # 模型的定义
        outputs = self.build_model(inputs)

        # 损失的定义
        loss = self.build_loss(outputs, inputs_length, labels, labels_length)

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
            tf.logging.info('training------')
            fetches = [train_op, loss]
            for i in range(self.epoch):
                total_loss = 0
                step = 0
                while 1:
                    batch_data = next(am_batch)
                    if batch_data is None:
                        break
                    feed_dict = {inputs: batch_data[0], inputs_length: batch_data[1], labels: batch_data[2],
                                 labels_length: batch_data[3]}
                    _, loss_np = sess.run(fetches, feed_dict=feed_dict)
                    print("setp:{:>4}, step_loss:{:.4f}".format(step, loss_np))

                    total_loss += loss_np
                    step += 1
                tf.logging.info('[%s] [epoch %d] loss %f', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i,
                                total_loss / step)

            # 保存模型
            tf.logging.info('save model------')
            saver.save(sess, self.model_save_name)

        pass

    def train_gpu(self, gpu_nums=1):
        """
        模型gpu训练
        :param gpu_nums: gpu数目
        :return: None
        """
        # 1.训练数据获取
        am_batch = self.generate_data_set()
        bsz_per_gpu = self.bsz // gpu_nums

        # 2.构建数据流图
        # 模型输入的定义
        inputs = tf.placeholder(name='the_inputs', shape=[None, None, 200, 1], dtype=tf.float32)
        inputs_length = tf.placeholder(name='inputs_length', shape=[None, 1], dtype=tf.int32)
        labels = tf.placeholder(name='the_labels', shape=[None, None], dtype=tf.float32)
        labels_length = tf.placeholder(name='labels_length', shape=[None, 1], dtype=tf.int32)

        # 多GPU数据流图构建
        tower_grads, tower_losses = [], []
        for i in range(gpu_nums):
            reuse = True if i > 0 else None
            with tf.device("/gpu:%d" % i), tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                # 数据的分割
                inputs_i = inputs[i * bsz_per_gpu:(i + 1) * bsz_per_gpu]
                inputs_length_i = inputs_length[i * bsz_per_gpu:(i + 1) * bsz_per_gpu]
                labels_i = labels[i * bsz_per_gpu:(i + 1) * bsz_per_gpu]
                labels_length_i = labels_length[i * bsz_per_gpu:(i + 1) * bsz_per_gpu]

                # 模型的定义
                outputs_i = self.build_model(inputs_i)

                # 损失的定义
                loss_i = self.build_loss(outputs_i, inputs_length_i, labels_i, labels_length_i)

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
            tf.logging.info('training------')
            fetches = [train_op, loss]
            for i in range(self.epoch):
                total_loss = 0
                step = 0
                while 1:
                    batch_data = next(am_batch)
                    if batch_data is None:
                        break
                    feed_dict = {inputs: batch_data[0], inputs_length: batch_data[1], labels: batch_data[2],
                                 labels_length: batch_data[3]}
                    _, loss_np = sess.run(fetches, feed_dict=feed_dict)
                    print("setp:{:>4}, step_loss:{:.4f}".format(step, loss_np))

                    total_loss += loss_np
                    step += 1
                tf.logging.info('[%s] [epoch %d] loss %f', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i,
                                total_loss / step)

            # 保存模型
            tf.logging.info('save model------')
            saver.save(sess, self.model_save_name)

        pass

    def start_session(self):
        """
        开启模型用于预测时的会话，并加载数据流图
        :return:
        """
        # 1.构建数据流图
        with tf.device('/cpu:0'):
            # 模型输入的定义
            self.pre_inputs = tf.placeholder(name='the_inputs', shape=[None, None, 200, 1], dtype=tf.float32)

            # 模型的定义
            self.pre_outputs = self.build_model(self.pre_inputs)

        # 2.开启会话
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, self.model_save_name)
        # self.sess.run(tf.global_variables_initializer())

        pass

    def predict(self, wav_data):
        """
        模型预测
        :param wav_file:
        :return:
        """
        # wav_data = get_online_data(wav_file)
        feed_dict = {self.pre_inputs: wav_data}
        output = self.sess.run([self.pre_outputs], feed_dict=feed_dict)

        return output


# =============================模型组件====================================
class Cnn2d_cell(object):
    def __init__(self, filters, kernel_size, padding, activation, use_bias, kernel, bias, n_conv, is_pool):
        super(Cnn2d_cell, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel
        self.bias_initializer = bias
        self.n_conv = n_conv
        self.is_pool = is_pool

        self.build()

    def build(self):
        self.conv2d = []
        self.bn = []
        for i in range(self.n_conv):
            self.conv2d.append(
                tf.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding,
                                 activation=self.activation, use_bias=self.use_bias,
                                 kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer))
            self.bn.append(tf.layers.BatchNormalization())

    def __call__(self, x):
        for i in range(self.n_conv):
            x = self.bn[i](self.conv2d[i](x))
        if self.is_pool:
            x = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        return x


class cfsmn_cell(object):
    def __init__(self, name, input_size, output_size, hidden_size, l_memory_size, r_memory_size, stride,
                 kernel_initializer, bias_initializer):
        super(cfsmn_cell, self).__init__()
        self.__name__ = name
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.l_memory_size = l_memory_size
        self.r_memory_size = r_memory_size
        self.stride = stride
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.memory_size = l_memory_size + r_memory_size + 1

        self.build()

    def build(self):
        self.lay_norm = LayerNormalization('cfsmn_laynor', self.hidden_size, self.kernel_initializer,
                                           self.bias_initializer)
        self.V = tf.get_variable("cfsmn_V", [self.input_size, self.hidden_size], initializer=self.kernel_initializer)
        self.bias_V = tf.get_variable("cfsmn_V_bias", [self.hidden_size], initializer=self.bias_initializer)
        self.U = tf.get_variable("cfsmn_U", [self.hidden_size, self.output_size], initializer=self.kernel_initializer)
        self.bias_U = tf.get_variable("cfsmn_U_bias", [self.output_size], initializer=self.bias_initializer)
        self.memory_weights = tf.get_variable("memory_weights", [self.memory_size, self.hidden_size],
                                              initializer=self.kernel_initializer)

    def __call__(self, inputs):
        # liner transformer
        p = tf.tensordot(inputs, self.V, axes=1) + self.bias_V
        height = tf.shape(p)[0]
        length = tf.shape(p)[1]
        depth = tf.shape(p)[2]

        # memory compute
        btach_p = space_to_batch(inputs=p, stride=self.stride, length=length, l_memory_size=self.l_memory_size,
                                 r_memory_size=self.r_memory_size)
        btach_p_m = tf.einsum('tbij,ij->tbij', btach_p, self.memory_weights)
        btach_p_m = tf.reduce_sum(btach_p_m, axis=2)
        p_hatt = batch_to_space(inputs=btach_p_m, height=height, length=length, depth=depth)

        # liner transformer
        p_hatt = self.lay_norm(p + p_hatt)
        h = tf.tensordot(p_hatt, self.U, axes=1) + self.bias_U

        return [h, p_hatt]


class dfsmn_cell(object):
    def __init__(self, name, input_size, output_size, hidden_size, l_memory_size, r_memory_size, stride,
                 kernel_initializer, bias_initializer):
        super(dfsmn_cell, self).__init__()
        self.__name__ = name
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.l_memory_size = l_memory_size
        self.r_memory_size = r_memory_size
        self.stride = stride
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.memory_size = l_memory_size + r_memory_size + 1

        self.build()

    def build(self):
        self.lay_norm = LayerNormalization('dfsmn_laynor', self.hidden_size, self.kernel_initializer,
                                           self.bias_initializer)
        self.V = tf.get_variable("dfsmn_V", [self.input_size, self.hidden_size], initializer=self.kernel_initializer)
        self.bias_V = tf.get_variable("dfsmn_V_bias", [self.hidden_size], initializer=self.bias_initializer)
        self.U = tf.get_variable("dfsmn_U", [self.hidden_size, self.output_size], initializer=self.kernel_initializer)
        self.bias_U = tf.get_variable("dfsmn_U_bias", [self.output_size], initializer=self.bias_initializer)
        self.memory_weights = tf.get_variable("memory_weights", [self.memory_size, self.hidden_size],
                                              initializer=self.kernel_initializer)

    def __call__(self, args):
        inputs = args[0]
        last_p_hatt = args[1]

        # liner transformer
        p = tf.tensordot(inputs, self.V, axes=1) + self.bias_V
        height = tf.shape(p)[0]
        length = tf.shape(p)[1]
        depth = tf.shape(p)[2]

        # memory compute
        btach_p = space_to_batch(inputs=p, stride=self.stride, length=length, l_memory_size=self.l_memory_size,
                                 r_memory_size=self.r_memory_size)
        btach_p_m = tf.einsum('tbij,ij->tbij', btach_p, self.memory_weights)
        btach_p_m = tf.reduce_sum(btach_p_m, axis=2)
        p_hatt = batch_to_space(inputs=btach_p_m, height=height, length=length, depth=depth)

        # liner transform + skip-connect
        p_hatt = self.lay_norm(last_p_hatt + p + p_hatt)
        h = tf.tensordot(p_hatt, self.U, axes=1) + self.bias_U

        return [h, p_hatt]


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


def space_to_batch(inputs, stride, length, l_memory_size, r_memory_size):
    """
    参照卷积和空洞卷积实现方式，对输入数据进行拆分
    :param inputs:3-D, [batch, length, depth]
    :param stride:
    :param length:
    :return:
    """

    def for_cond(step, b_inputs):
        return step < length

    def for_body(step, b_inputs):
        # 前向记忆开始位置获取
        l_mem_index = tf.maximum(step % stride, step - l_memory_size * stride)
        l_pad = l_memory_size - (step - l_mem_index) // stride

        # 后向记忆结束位置获取
        r_mem_index = tf.minimum(length, step + r_memory_size * stride + 1)
        r_pad = r_memory_size - (r_mem_index - step - 1) // stride

        # 拆分，补全
        batch = inputs[:, l_mem_index:r_mem_index:stride, :]
        batch = tf.pad(batch, [[0, 0], [l_pad, r_pad], [0, 0]])

        return step + 1, b_inputs.write(step, batch)

    # 数据拆分
    b_inputs = tf.TensorArray(dtype=tf.float32, size=length)
    _, b_inputs = tf.while_loop(for_cond, for_body, [0, b_inputs])
    b_inputs = b_inputs.stack()

    return b_inputs


def batch_to_space(inputs, height, length, depth):
    """
    数据拆分的逆运算
    :param inputs:
    :param height:
    :param length:
    :param depth:
    :return:
    """
    s_inputs = tf.reshape(inputs, shape=(length, height, depth))
    s_inputs = tf.transpose(s_inputs, [1, 0, 2])

    return s_inputs


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


def compute_fbank(file):
    """
    计算音频文件的fbank特征
    :param file: 音频文件
    :return:
    """
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    fs, wavsignal = wav.read(file)
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    wav_arr = np.array(wavsignal)
    range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1  # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype=np.float)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    # data_input = data_input[::]
    return data_input


def ctc_len(label):
    add_len = 0
    label_len = len(label)
    for i in range(label_len - 1):
        if label[i] == label[i + 1]:
            add_len += 1
    return label_len + add_len


def wav_padding(wav_data_lst):
    wav_lens = [len(data) for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_lens = np.array([[leng // 8] for leng in wav_lens])
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens


def label_padding(label_data_lst):
    label_lens = np.array([[len(label)] for label in label_data_lst])
    max_label_len = max(label_lens)[0]
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
    for i in range(len(label_data_lst)):
        new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
    return new_label_data_lst, label_lens


def get_online_data(file):
    fbank = compute_fbank(file)
    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
    pad_fbank[:fbank.shape[0], :] = fbank

    new_wav_data = np.zeros((1, len(pad_fbank), 200, 1))
    new_wav_data[0, :, :, 0] = pad_fbank

    return new_wav_data
