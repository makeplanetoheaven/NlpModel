# coding=utf-8

import os
import random
import time

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# =============================模型超参数====================================
def induction_net_hparams():
    params = tf.contrib.training.HParams(data_path=None, vocab_dict=None, lm=None, is_training_lm=False, episode=1000,
                                         max_step=1000, lr=0.0008, dropout=0., d_input=768, n_head=12, d_head=64,
                                         c_way=1, k_shot=1, n_query=1, n_route=3, d_tensor=128, init_range=1,
                                         init_std=0, is_training=True, gpu_index=None, task_name=None, save_path=None)

    return params


# =============================模型框架====================================
class InductionNet:
    def __init__(self, args):
        # 数据参数
        self.data_path = args.data_path
        self.vocab_dict = args.vocab_dict

        # 外部模型参数
        self.lm = args.lm
        self.is_training_lm = args.is_training_lm

        # 超参数
        self.episode = args.episode
        self.max_step = args.max_step
        self.lr = args.lr
        self.dropout = args.dropout
        self.d_input = args.d_input
        self.n_head = args.n_head
        self.d_head = args.d_head
        self.c_way = args.c_way
        self.k_shot = args.k_shot
        self.n_query = args.n_query
        self.n_route = args.n_route
        self.d_tensor = args.d_tensor
        self.init_range = args.init_range
        self.init_std = args.init_std
        self.is_training = args.is_training
        self.gpu_index = args.gpu_index

        # 存储参数
        self.task_name = args.task_name
        self.model_save_name = args.save_path + 'induction_net'
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
        label_list = []

        def read_file_list(data_path):
            path_list = os.listdir(data_path)
            for path in path_list:
                new_path = os.path.join(data_path, path)
                if os.path.isdir(new_path):
                    read_file_list(new_path)
                else:
                    file_list.append(new_path)
                    label_list.append(path.split('.')[0])

        read_file_list(self.data_path)

        # 2.生成临时数据
        data_dict = {}
        for i, file in enumerate(file_list):
            data_dict[label_list[i]] = []
            with open(file, 'r', encoding='utf-8') as fo:
                for line in fo:
                    data_dict[label_list[i]].append([self.vocab_dict[char] for char in line.rstrip('\n')])
        tf.logging.info('end of preprocess')

        # 3.batch生成器定义
        def induction_net_gen():
            while 1:
                # 1.类别选择
                ways = random.sample(label_list, self.c_way)

                # 2.支持集选择
                s_set = []
                temp_list = []
                for i, way in enumerate(ways):
                    way_data = data_dict[way]
                    s_set += random.sample(way_data, self.k_shot)
                    temp_list += list(zip(way_data, [i for _ in range(len(way_data))]))

                # 3.问题集选择
                q_set, q_labels = list(map(list, zip(*random.sample(temp_list, self.n_query))))

                # 4.数据补全，标签one-hot转换
                s_set = padded_batch(s_set, padding_values=len(self.vocab_dict))
                q_set = padded_batch(q_set, padding_values=len(self.vocab_dict))
                q_labels = np.array(q_labels)

                yield s_set, q_set, q_labels

        return induction_net_gen

    def input_fn(self):
        """
        模型输入函数
        :return:
        """
        ds = tf.data.Dataset.from_generator(self.generate_data_set(), (tf.int32, tf.int32, tf.int32), (
            tf.TensorShape([self.c_way * self.k_shot, None]), tf.TensorShape([self.n_query, None]),
            tf.TensorShape([self.n_query])))
        if self.gpu_index is not None:
            padded_shapes = (tf.TensorShape([self.c_way * self.k_shot, None]),
                             tf.TensorShape([self.n_query, None]),
                             tf.TensorShape([self.n_query]))
            padding_values = (tf.constant(len(self.vocab_dict), dtype=tf.int32),
                              tf.constant(len(self.vocab_dict), dtype=tf.int32),
                              tf.constant(0, dtype=tf.int32))
            ds = ds.padded_batch(len(self.gpu_index), padded_shapes=padded_shapes, padding_values=padding_values)

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
            with tf.variable_scope('global_attention_layers', reuse=tf.AUTO_REUSE):
                self.q_h = tf.get_variable('q', [self.n_head, self.d_head], dtype=tf.float32,
                                           initializer=self.kernel_initializer)
                self.k_weight = tf.get_variable('k/kernel', [self.d_input, self.n_head, self.d_head], dtype=tf.float32,
                                                initializer=self.kernel_initializer)
                self.k_bias = tf.get_variable('k/bias', [self.n_head, self.d_head], dtype=tf.float32,
                                              initializer=self.bias_initializer)
                self.v_weight = tf.get_variable('v/kernel', [self.d_input, self.n_head, self.d_head], dtype=tf.float32,
                                                initializer=self.kernel_initializer)
                self.o_weight = tf.get_variable('o/kernel', [self.d_input, self.n_head, self.d_head], dtype=tf.float32,
                                                initializer=self.kernel_initializer)

            with tf.variable_scope('induction_layer', reuse=tf.AUTO_REUSE):
                self.s_weight = tf.get_variable(name='s_weight', shape=[self.d_input, self.d_input],
                                                initializer=self.kernel_initializer)
                self.s_bias = tf.get_variable(name='s_bias', shape=[self.d_input], initializer=self.bias_initializer)

            with tf.variable_scope('relation_layer', reuse=tf.AUTO_REUSE):
                self.m_weight = tf.get_variable(name='s_weight', shape=[self.d_input, self.d_input, self.d_tensor],
                                                initializer=self.kernel_initializer)
                self.ffn = Dense('ffn', d_inputs=self.d_tensor, d_model=1,
                                 kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                 activation=None, use_bias=True)

    def build_model(self, s, q):
        """
        构建模型训练所需的数据流图
        :return: outputs
        """
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

        with tf.name_scope('SenVec_q'):
            # 评分矩阵计算
            q_k_h = tf.einsum('ibh,hnd->ibnd', q, self.k_weight) + self.k_bias
            q_v_h = tf.einsum('ibh,hnd->ibnd', q, self.v_weight)

            q_attn_score = tf.einsum('nd,ibnd->ibn', self.q_h, q_k_h) * scale
            q_attn_prob = tf.nn.softmax(q_attn_score, 0)

            # 句子向量计算
            q_attn_vec = tf.einsum('ibn,jbnd->bnd', q_attn_prob, q_v_h)
            e_q = tf.einsum('bnd,hnd->bh', q_attn_vec, self.o_weight)
            e_q = tf.layers.dropout(e_q, self.dropout, training=self.is_training)

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

        with tf.name_scope('Relation'):
            v = tf.einsum('ijn,cj->icn', self.m_weight, c)
            v = self.activation[1](tf.einsum('qi,icn->qcn', e_q, v))
            r = self.ffn(v)  # [q_nums, c_way, 1]
            logits = tf.reshape(r, [tf.shape(r)[0], tf.shape(r)[1]])

        return logits

    def build_loss(self, logits, labels):
        """
        模型损失函数的构建
        :return: loss
        """
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
        s_set = tf.transpose(ele[0], [1, 0])
        q_set = tf.transpose(ele[1], [1, 0])
        labels = ele[2]

        # 2.构建数据流图
        # Lm模型的定义
        s_lm_h, _ = self.lm.build_model(s_set)
        q_lm_h, _ = self.lm.build_model(q_set)

        # InductionNet模型的定义
        logits = self.build_model(s_lm_h, q_lm_h)

        # 损失的定义
        loss = self.build_loss(logits, labels)

        # 优化器的定义
        # 分解成梯度列表和变量列表
        t_vars = tf.trainable_variables()
        lm_vars = [var for var in t_vars if var.name.startswith(self.lm.task_name)] if self.is_training_lm else []
        induction_net_vars = [var for var in t_vars if var.name.startswith(self.task_name)]
        training_vars = lm_vars + induction_net_vars
        grads, vars = zip(*self.opt.compute_gradients(loss, training_vars))
        # 梯度修剪
        gradients, _ = tf.clip_by_global_norm(grads, 5)
        # 将每个梯度以及对应变量打包
        train_op = self.opt.apply_gradients(zip(gradients, vars))

        # 3.模型训练
        lm_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.lm.task_name))
        induction_net_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.task_name))
        with tf.Session() as sess:
            # 全局初始化参数
            sess.run(tf.global_variables_initializer())

            # 加载InductionNet模型
            if os.path.exists(self.model_save_checkpoint):
                induction_net_saver.restore(sess, self.model_save_name)

            # 加载Lm模型
            if os.path.exists(self.lm.model_save_checkpoint):
                lm_saver.restore(sess, self.lm.model_save_name)

            # 开始迭代，使用Adam优化的随机梯度下降法，并将结果输出到日志文件
            tf.logging.info('begin of training')
            fetches = [train_op, loss]
            total_loss = np.zeros(shape=[self.max_step], dtype=float)
            cur_step = 0
            for _ in range(self.episode):
                _, loss_np = sess.run(fetches)
                total_loss[cur_step % self.max_step] = loss_np
                cur_step += 1

                if cur_step > 0 and cur_step % self.max_step == 0:
                    loss_mean = np.mean(total_loss)
                    loss_var = np.var(total_loss)
                    tf.logging.info('[%s] [step %d] loss %f var %f',
                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                    cur_step, loss_mean, loss_var)
            tf.logging.info('end of training')

            # 保存模型
            tf.logging.info('save model')
            if self.is_training_lm:
                lm_saver.save(sess, self.lm.model_save_name)
            induction_net_saver.save(sess, self.model_save_name)

        pass

    def train_gpu(self):
        """
        模型gpu训练
        :return: None
        """
        # 环境设置
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, self.gpu_index))
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

        # 1.数据获取
        ele = self.input_fn().make_one_shot_iterator().get_next()
        s_set = tf.transpose(ele[0], [0, 2, 1])
        q_set = tf.transpose(ele[1], [0, 2, 1])
        labels = ele[2]

        # 2.构建数据流图
        # 多GPU数据流图构建
        tower_grads, tower_losses = [], []
        for i, core in enumerate(self.gpu_index):
            reuse = True if i > 0 else None
            with tf.device("/gpu:%d" % core), tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                # 数据的分割
                s_set_i = s_set[i]
                q_set_i = q_set[i]
                labels_i = labels[i]

                # Lm模型的定义
                s_lm_h_i, _ = self.lm.build_model(s_set_i)
                q_lm_h_i, _ = self.lm.build_model(q_set_i)

                # InductionNet模型的定义
                logits_i = self.build_model(s_lm_h_i, q_lm_h_i)

                # 损失的定义
                loss_i = self.build_loss(logits_i, labels_i)

                # 优化器的定义
                # 分解成梯度列表和变量列表
                t_vars = tf.trainable_variables()
                lm_vars = [var for var in t_vars if
                           var.name.startswith(self.lm.task_name)] if self.is_training_lm else []
                induction_net_vars = [var for var in t_vars if var.name.startswith(self.task_name)]
                training_vars = lm_vars + induction_net_vars
                grads, vars = zip(*self.opt.compute_gradients(loss_i, training_vars))
                # 梯度修剪
                gradients, _ = tf.clip_by_global_norm(grads, 5)

                tower_grads.append(zip(gradients, vars))
                tower_losses.append(loss_i)

        loss = tf.reduce_mean(tower_losses, 0)
        grads = average_gradients(tower_grads)
        train_op = self.opt.apply_gradients(grads)

        # 3.模型训练
        lm_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.lm.task_name))
        induction_net_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.task_name))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # 全局初始化参数
            sess.run(tf.global_variables_initializer())

            # 加载InductionNet模型
            if os.path.exists(self.model_save_checkpoint):
                induction_net_saver.restore(sess, self.model_save_name)

            # 加载Lm模型
            if os.path.exists(self.lm.model_save_checkpoint):
                lm_saver.restore(sess, self.lm.model_save_name)

            # 开始迭代，使用Adam优化的随机梯度下降法，并将结果输出到日志文件
            tf.logging.info('begin of training')
            fetches = [train_op, loss]
            total_loss = np.zeros(shape=[self.max_step], dtype=float)
            cur_step = 0
            for _ in range(self.episode):
                _, loss_np = sess.run(fetches)
                total_loss[cur_step % self.max_step] = loss_np
                cur_step += 1

                if cur_step > 0 and cur_step % self.max_step == 0:
                    loss_mean = np.mean(total_loss)
                    loss_var = np.var(total_loss)
                    tf.logging.info('[%s] [step %d] loss %f var %f',
                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                    cur_step, loss_mean, loss_var)
            tf.logging.info('end of training')

            # 保存模型
            tf.logging.info('save model')
            if self.is_training_lm:
                lm_saver.save(sess, self.lm.model_save_name)
            induction_net_saver.save(sess, self.model_save_name)

        pass

    def start_session(self, label_dict):
        """
        开启模型用于预测时的会话，并加载数据流图
        :return:
        """
        # 环境设置
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # 数据预处理
        self.labels = list(label_dict.keys())
        self.c_way, self.k_shot, s_set = s_set_generate(label_dict, self.vocab_dict)

        # 1.构建数据流图
        # 模型输入的定义
        self.pre_inputs = tf.placeholder(name='the_inputs', shape=[None, None], dtype=tf.float32)
        s_set = tf.constant(np.transpose(s_set, axes=[1, 0]))

        # Lm模型的定义
        s_lm_h, _ = self.lm.build_model(s_set)
        q_lm_h, _ = self.lm.build_model(self.pre_inputs)

        # InductionNet模型的定义
        self.pre_outputs = self.build_model(s_lm_h, q_lm_h)

        # 2.开启会话
        lm_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.lm.task_name))
        induction_net_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.task_name))
        self.sess = tf.Session()
        lm_saver.restore(self.sess, self.lm.model_save_name)
        induction_net_saver.restore(self.sess, self.model_save_name)

        pass

    def predict(self, q_list):
        """
        模型预测
        :param q_list:
        :return:
        """
        # 数据处理
        q_set = [[self.vocab_dict[c] for c in q] for q in q_list]
        q_set = padded_batch(q_set, padding_values=len(self.vocab_dict))
        q_set = np.transpose(q_set, [1, 0])

        # 模型预测
        feed_dict = {self.pre_inputs: q_set}
        output = self.sess.run(self.pre_outputs, feed_dict=feed_dict)
        output = np.argmax(output, axis=1)
        output = [self.labels[i] for i in output]

        return output


# =============================模型组件====================================
class Dense(object):
    def __init__(self, name,
                 d_inputs,
                 d_model,
                 kernel_initializer,
                 bias_initializer=None,
                 activation=None,
                 use_bias=True):
        super(Dense, self).__init__()
        self.name = name
        self.d_inputs = d_inputs
        self.d_model = d_model
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.use_bias = use_bias
        self.build()

    def build(self):
        self.kernel = tf.get_variable(self.name + "_kernel", [self.d_inputs, self.d_model],
                                      initializer=self.kernel_initializer)
        if self.use_bias:
            self.bias = tf.get_variable(self.name + "_bias", [self.d_model], initializer=self.bias_initializer)

    def __call__(self, x):
        out = tf.tensordot(x, self.kernel, axes=1)

        if self.use_bias:
            out += self.bias

        if self.activation:
            out = self.activation(out)

        return out


def squash(x):
    """
    非线性压缩函数，类似激活函数(激活函数是对每一个标量求解)
    :param x: 函数输入
    :return:
    """
    x_norm_square = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    x_norm = tf.sqrt(x_norm_square)

    scale = x_norm_square / ((1 + x_norm_square) * x_norm)

    return x * scale


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


def padded_batch(inputs, padding_values=0):
    """
    对输入数据进行pad
    :param inputs: 2D, [batch, len]
    :param padding_values:
    :return:
    """
    data_lens = [len(data) for data in inputs]
    max_len = max(data_lens)
    batch = len(inputs)
    new_inputs = np.array([padding_values for _ in range(batch * max_len)]).reshape(batch, max_len)
    for i in range(batch):
        new_inputs[i, :len(inputs[i])] = inputs[i]

    return new_inputs


def one_hot_generate(label_list, n_class):
    """
    sparser label to one-hot
    :param label_list:
    :param n_class:
    :return:
    """
    batch = len(label_list)
    one_hot_label = np.zeros([batch, n_class])
    for i, label in enumerate(label_list):
        one_hot_label[i, label] = 1

    return one_hot_label


def s_set_generate(label_dict, vocab_dict):
    """
    生成e_s
    :param label_dict:
    :param vocab_dict:
    :return:
    """
    c_way = 0
    k_shot = 0
    for key in label_dict:
        c_way += 1
        k_shot = max(len(label_dict[key]), k_shot)

    s_set = []
    for key in label_dict:
        s_set += [[vocab_dict[c] for c in d] for d in label_dict[key]]\
                 + [[] for _ in range(k_shot - len(label_dict[key]))]

    s_set = padded_batch(s_set, padding_values=len(vocab_dict))

    return c_way, k_shot, s_set
