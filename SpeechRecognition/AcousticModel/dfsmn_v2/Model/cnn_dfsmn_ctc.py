# coding=utf-8

import json
import os
import time
from random import shuffle

import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
from scipy.fftpack import fft
from tensorflow.contrib.keras import backend as tfk
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)


# =============================模型超参数====================================
def am_hparams():
    params = tf.contrib.training.HParams(data_path=None, label_path=None, thchs30=False, aishell=False, prime=False,
                                         stcmd=False, magicdata=False, aidatatang=False, vocab_dict=None, bsz=1,
                                         epoch=1, max_step=1000, lr=1e-4, dropout=0.5, d_input=2048, d_model=512,
                                         l_mem=20, r_mem=20, stride=2, n_init_filters=256, n_conv=1, n_cnn_layers=2,
                                         n_dfsmn_layers=8, init_range=1, init_std=0, is_training=False, save_path=None)

    return params


# =============================模型框架====================================
class Am:
    def __init__(self, args):
        # 数据参数
        self.data_path = args.data_path
        self.data_cache = '{}/__dscache__/cnn_dfsmn_ctc.tfrecord'.format(
            args.data_path) if args.data_path is not None else None
        self.label_path = args.label_path
        self.thchs30 = args.thchs30
        self.aishell = args.aishell
        self.prime = args.prime
        self.stcmd = args.stcmd
        self.magicdata = args.magicdata
        self.aidatatang = args.aidatatang
        self.vocab_dict = args.vocab_dict
        self.n_vocab = len(self.vocab_dict)

        # 超参数
        self.bsz = args.bsz
        self.epoch = args.epoch
        self.max_step = args.max_step
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
        self.build_activation('swish')
        self.build_opt()
        self.build_parameters()

        # 存储路径
        self.model_save_name = args.save_path + 'cnn_dfsmn_ctc'
        self.model_save_checkpoint = args.save_path + 'checkpoint'

    def generate_data_set(self):
        """
        数据预处理函数
        :return:
        """
        assert os.path.exists(self.data_path), 'path {} does not exit'.format(self.data_path)
        assert os.path.exists(self.label_path), 'path {} does not exit'.format(self.label_path)
        tf.logging.info('begin of preprocess')

        # 1.获取数据列表和标签列表
        file_list = []
        label_list = []

        def read_file_list(data_path, label_path):
            tf.logging.info('get source list...')
            read_files = []
            if self.thchs30:
                read_files.append(label_path + 'thchs_train.txt')
                read_files.append(label_path + 'thchs_dev.txt')
                read_files.append(label_path + 'thchs_test.txt')
            if self.aishell:
                read_files.append(label_path + 'aishell_train.txt')
                read_files.append(label_path + 'aishell_dev.txt')
                read_files.append(label_path + 'aishell_test.txt')
            if self.prime:
                read_files.append(label_path + 'prime.txt')
            if self.stcmd:
                read_files.append(label_path + 'stcmd.txt')
            if self.magicdata:
                read_files.append(label_path + 'magicdata_train.txt')
                read_files.append(label_path + 'magicdata_dev.txt')
                read_files.append(label_path + 'magicdata_test.txt')
            if self.aidatatang:
                read_files.append(label_path + 'aidatatang_200zh_train.txt')
                read_files.append(label_path + 'aidatatang_200zh_dev.txt')
                read_files.append(label_path + 'aidatatang_200zh_test.txt')

            for file in read_files:
                tf.logging.info('load %s data...', file)
                with open(file, 'r', encoding='utf8') as fo:
                    for line in fo:
                        wav_file, labels, _ = line.split('\t')
                        file_list.append(data_path + wav_file)
                        label_list.append(labels.split(' '))

        read_file_list(self.data_path, self.label_path)

        # 2.数据预处理 write tfRecord，每个样本一条数据
        def write_example(inputs, labels):
            # 创建字典
            feature_dict = {}

            # 写入数据
            feature_dict['inputs'] = tf.train.Feature(float_list=tf.train.FloatList(value=inputs.reshape(-1)))
            feature_dict['inputs_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=inputs.shape))
            feature_dict['inputs_length'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=np.array([inputs.shape[0] // 4])))
            feature_dict['labels'] = tf.train.Feature(float_list=tf.train.FloatList(value=labels))
            feature_dict['labels_length'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=np.array([labels.shape[0]])))

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
        shuffle_list = [i for i in range(len(file_list))]
        shuffle(shuffle_list)
        data_nums = 0
        for i in tqdm(shuffle_list):
            try:
                # 数据特征提取
                fbank = compute_log_mel_fbank(file_list[i])

                # 保证卷积使得长度变为1/4仍有效
                pad_fbank = np.zeros((fbank.shape[0] // 4 * 4 + 4, fbank.shape[1]))
                pad_fbank[:fbank.shape[0], :] = fbank

                # 标签转ID
                label = np.array([self.vocab_dict[pny] for pny in label_list[i]])

                # 判断音频数据长度经过卷积后是否大于标签数据长度
                label_ctc_len = ctc_len(label)
                if pad_fbank.shape[0] // 4 >= label_ctc_len:
                    write_example(pad_fbank, label)
                    data_nums += 1
            except Exception as e:
                continue
        writer.close()
        tf.logging.info('the data nums is is %d', data_nums)
        tf.logging.info('end of preprocess')

    def input_fn(self):
        """
        模型输入函数
        :return:
        """
        assert os.path.exists(self.data_cache), 'file {} does not exit'.format(self.data_cache)

        # 数据对象构建 read tfRecord
        def parser(example):
            example_dict = {
                'inputs': tf.VarLenFeature(dtype=tf.float32),  # SparseTensor
                'inputs_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                'inputs_length': tf.FixedLenFeature(shape=(1,), dtype=tf.int64),
                'labels': tf.VarLenFeature(dtype=tf.float32),  # SparseTensor
                'labels_length': tf.FixedLenFeature(shape=(1,), dtype=tf.int64)
            }
            parsed_example = tf.parse_single_example(example, example_dict)
            parsed_example['inputs'] = tf.sparse_tensor_to_dense(parsed_example['inputs'])
            parsed_example['inputs'] = tf.reshape(parsed_example['inputs'], parsed_example['inputs_shape'])
            parsed_example['labels'] = tf.sparse_tensor_to_dense(parsed_example['labels'])
            parsed_example.pop('inputs_shape')

            return parsed_example

        file_names = [self.data_cache]
        ds = tf.data.TFRecordDataset(file_names)
        ds = ds.map(parser).repeat(self.epoch)
        ds = ds.shuffle(buffer_size=self.bsz)  # example-level shuffle
        padded_shapes = {
            'inputs': tf.TensorShape([None, 80]),
            'inputs_length': tf.TensorShape([1]),
            'labels': tf.TensorShape([None]),
            'labels_length': tf.TensorShape([1])
        }
        ds = ds.padded_batch(self.bsz, padded_shapes=padded_shapes, drop_remainder=True)
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
        with tf.variable_scope('cnn_layers', reuse=tf.AUTO_REUSE):
            self.cnn_cells = []
            filters = self.n_init_filters
            for i in range(self.n_cnn_layers):
                with tf.variable_scope('layer_{}'.format(i), reuse=tf.AUTO_REUSE):
                    if i < self.n_cnn_layers - 1:
                        cell = Cnn2d_cell(filters=filters, kernel_size=(3, 3), padding='same',
                                          activation=self.activation[1], use_bias=True, kernel=self.kernel_initializer,
                                          bias=self.bias_initializer, n_conv=self.n_conv, is_pooling=True)
                        filters = filters * 2
                    else:
                        cell = Cnn2d_cell(filters=filters // 2, kernel_size=(3, 3), padding='same',
                                          activation=self.activation[1], use_bias=True, kernel=self.kernel_initializer,
                                          bias=self.bias_initializer, n_conv=self.n_conv, is_pooling=False)

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
            self.dnn_layer = tf.layers.Dense(self.d_model, activation=self.activation[1], use_bias=True,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer=self.bias_initializer)
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
        outputs = tf.reshape(outputs, shape=(bsz, -1, 80, 1))
        with tf.name_scope('CNNLayer'):
            for i in range(self.n_cnn_layers):
                with tf.name_scope('Layer{}'.format(i)):
                    outputs = self.cnn_cells[i](outputs)
                    outputs = tf.layers.dropout(outputs, self.dropout, training=self.is_training)

        with tf.name_scope('LinerTransformer'):
            outputs = tf.reshape(outputs, shape=(bsz, -1, 2560))
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
            outputs = self.dnn_layer(outputs)
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
            self.opt = tf.train.MomentumOptimizer(self.lr, 0.9)
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
        # 环境设置
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

        # 1.训练数据获取
        ele = self.input_fn().make_one_shot_iterator().get_next()
        inputs = ele['inputs']
        inputs_length = ele['inputs_length']
        labels = ele['labels']
        labels_length = ele['labels_length']

        # 2.构建数据流图
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

    def train_gpu(self, gpu_index):
        """
        模型gpu训练
        :param gpu_index: gpu索引列表
        :return: None
        """
        # 环境设置
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_index))
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

        # 1.训练数据获取
        ele = self.input_fn().make_one_shot_iterator().get_next()
        inputs = ele['inputs']
        inputs_length = ele['inputs_length']
        labels = ele['labels']
        labels_length = ele['labels_length']
        bsz_per_gpu = self.bsz // len(gpu_index)

        # 2.构建数据流图
        # 多GPU数据流图构建
        tower_grads, tower_losses = [], []
        for i, core in enumerate(gpu_index):
            reuse = True if i > 0 else None
            with tf.device("/gpu:%d" % core), tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
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

    def start_session(self):
        """
        开启模型用于预测时的会话，并加载数据流图
        :return:
        """
        # 环境设置
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # 1.构建数据流图
        # 模型输入的定义
        self.pre_inputs = tf.placeholder(name='the_inputs', shape=[None, None, 80, 1], dtype=tf.float32)

        # 模型的定义
        self.pre_outputs = self.build_model(self.pre_inputs)

        # 2.开启会话
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, self.model_save_name)

        pass

    def predict(self, wav_file):
        """
        模型预测
        :param wav_file:
        :return:
        """
        wav_data = get_online_data(wav_file)
        feed_dict = {self.pre_inputs: wav_data}
        output = self.sess.run([self.pre_outputs], feed_dict=feed_dict)
        pinyin = decode_ctc(output[0], list(self.vocab_dict.keys()))

        return pinyin


# =============================模型组件====================================
class Cnn2d_cell(object):
    def __init__(self, filters, kernel_size, padding, activation, use_bias, kernel, bias, n_conv, is_pooling):
        super(Cnn2d_cell, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel
        self.bias_initializer = bias
        self.n_conv = n_conv
        self.is_pooling = is_pooling

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
        if self.is_pooling:
            x = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 非重叠max-pooling

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

        # memory compute v1
        # height = tf.shape(p)[0]
        # length = tf.shape(p)[1]
        # depth = tf.shape(p)[2]
        # btach_p = space_to_batch(inputs=p, stride=self.stride, length=length, l_memory_size=self.l_memory_size,
        #                          r_memory_size=self.r_memory_size)
        # btach_p_m = tf.einsum('tbij,ij->tbij', btach_p, self.memory_weights)
        # btach_p_m = tf.reduce_sum(btach_p_m, axis=2)
        # p_hatt = batch_to_space(inputs=btach_p_m, height=height, length=length, depth=depth)

        # memory compute v2
        p_hatt = compute_memory_block(inputs=p, stride=self.stride, memroy_weight=self.memory_weights,
                                      l_memory_size=self.l_memory_size, r_memory_size=self.r_memory_size)

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

        # memory compute v1
        # height = tf.shape(p)[0]
        # length = tf.shape(p)[1]
        # depth = tf.shape(p)[2]
        # btach_p = space_to_batch(inputs=p, stride=self.stride, length=length, l_memory_size=self.l_memory_size,
        #                          r_memory_size=self.r_memory_size)
        # btach_p_m = tf.einsum('tbij,ij->tbij', btach_p, self.memory_weights)
        # btach_p_m = tf.reduce_sum(btach_p_m, axis=2)
        # p_hatt = batch_to_space(inputs=btach_p_m, height=height, length=length, depth=depth)

        # memory compute v2
        p_hatt = compute_memory_block(inputs=p, stride=self.stride, memroy_weight=self.memory_weights,
                                      l_memory_size=self.l_memory_size, r_memory_size=self.r_memory_size)

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


def compute_memory_block(inputs, stride, memroy_weight, l_memory_size, r_memory_size):
    """
    cfsmn,dfsmn记忆单元计算
    :param inputs: 3D, [batch, length, d_hidden]
    :param stride:
    :param memroy_weight:
    :param l_memory_size:
    :param r_memory_size:
    :return:
    """
    memory_size = l_memory_size + r_memory_size + 1
    for i in range(memory_size):
        l_pad = max((l_memory_size - i) * stride, 0)
        l_index = max((i - l_memory_size) * stride, 0)
        r_pad = max((i - l_memory_size) * stride, 0)
        r_index = min((i - l_memory_size) * stride, 0)
        if r_index != 0:
            pad_inputs = tf.pad(inputs[:, l_index:r_index, :], [[0, 0], [l_pad, r_pad], [0, 0]])
        else:
            pad_inputs = tf.pad(inputs[:, l_index:, :], [[0, 0], [l_pad, r_pad], [0, 0]])
        if i == 0:
            p_hatt = tf.einsum('bld,d->bld', pad_inputs, memroy_weight[i, :])
        else:
            p_hatt += tf.einsum('bld,d->bld', pad_inputs, memroy_weight[i, :])

    return p_hatt


def space_to_batch(inputs, stride, length, l_memory_size, r_memory_size):
    """
    参照卷积和空洞卷积实现方式，对输入数据进行拆分
    :param inputs:
    :param stride:
    :param length:
    :param l_memory_size:
    :param r_memory_size:
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


def compute_log_mel_fbank(wav_file):
    """
    计算音频文件的fbank特征
    :param wav_file: 音频文件
    :return:
    """
    # 1.数据读取
    sample_rate, signal = wav.read(wav_file)
    # print('sample rate:', sample_rate, ', frame length:', len(signal))

    # 2.预增强
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # 3.分帧
    frame_size, frame_stride = 0.025, 0.01
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1,
                                                                                                                    1)
    frames = pad_signal[indices]

    # 4.加窗
    hamming = np.hamming(frame_length)
    frames *= hamming

    # 5.N点快速傅里叶变换（N-FFT）
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # 获取能量谱

    # 6.提取mel Fbank
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)

    n_filter = 80  # mel滤波器组的个数, 影响每一帧输出维度，通常取40或80个
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filter + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    fbank = np.zeros((n_filter, int(NFFT / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    for i in range(1, n_filter + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])

    # 7.提取log mel Fbank
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    return filter_banks


def ctc_len(label):
    add_len = 0
    label_len = len(label)
    for i in range(label_len - 1):
        if label[i] == label[i + 1]:
            add_len += 1
    return label_len + add_len


def get_online_data(wav_file):
    fbank = compute_log_mel_fbank(wav_file)
    pad_fbank = np.zeros((fbank.shape[0] // 4 * 4 + 4, fbank.shape[1]))
    pad_fbank[:fbank.shape[0], :] = fbank

    new_wav_data = np.zeros((1, len(pad_fbank), 80, 1))
    new_wav_data[0, :, :, 0] = pad_fbank

    return new_wav_data


def decode_ctc(num_result, num2word):
    """
    定义解码器
    :param num_result:
    :param num2word:
    :return:
    """
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype=np.int32)
    in_len[0] = result.shape[1]
    r = tfk.ctc_decode(result, in_len, greedy=True, beam_width=10, top_paths=1)
    r1 = tfk.get_value(r[0][0])
    r1 = r1[0]
    text = []
    for i in r1:
        text.append(num2word[i])

    return text
