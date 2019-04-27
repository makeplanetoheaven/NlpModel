# coding=utf-8

# 引入外部库
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, DropoutWrapper

# 引入内部库


class DependencyGraph:
    def __init__(self,
                 data_set,  # 训练数据集
                 label_set,  # 训练标签集
                 batch_size,  # 训练批次
                 hidden_num,  # 隐藏层个数
                 classes_num,  # 类别数目
                 learning_rate,  # 学习率
                 train_num,  # 训练次数
                 model_path,  # 模型存储路径
                 ):
        # 外部参数
        self.data_set = data_set
        self.label_set = label_set
        self.batch_size = batch_size
        self.hidden_num = hidden_num
        self.classes_num = classes_num
        self.learning_rate = learning_rate
        self.train_num = train_num

        # 内部参数
        self.steps = 0
        self.data_dimension = 0
        self.data_index = 0
        self.data_actual_length = []
        self.data_max_length = 0
        self.lost_mask = None
        self.model_save_name = model_path + 'DependencyGraph'
        self.model_save_checkpoint = model_path + 'checkpoint'

        # 模型参数
        self.graph = None
        self.session = None
        self.saver = None
        self.encoder_inputs = None
        self.encoder_inputs_actual_length = None
        self.encoder_hidden_index = 0
        self.encoder_outputs = None
        self.encoder_final_state = None
        self.decoder_outputs_arr = []
        self.decoder_targets_matrix = None
        self.decoder_lost_mask = None
        self.decoder_prediction_matrix = []
        self.loss = None
        self.train_op = None

    def init_model_parameters(self):
        # 迭代次数
        self.steps = len(self.data_set) // self.batch_size
        print("There are %d steps" % self.steps)

        # 数据维度
        self.data_dimension = len(self.data_set[0][0])
        print('the dimension of data is %d' % self.data_dimension)

        # 获取序列实际长度及最大长度
        for data in self.data_set:
            self.data_actual_length.append(len(data))
        self.data_max_length = max(self.data_actual_length)

        # 数据补全
        for i in range(len(self.data_set)):
            for j in range(self.data_max_length - len(self.data_set[i])):
                self.data_set[i].append([0.0 for _ in range(self.data_dimension)])
        self.data_set = np.array(self.data_set)

        # 标签补全
        for i in range(len(self.label_set)):
            label_matrix = []
            for j in range(self.data_max_length):
                if j < len(self.label_set[i]):
                    label_matrix.append(
                        self.label_set[i][j] + [0 for _ in range(self.data_max_length - len(self.label_set[i][j]))])
                else:
                    label_matrix.append([0 for _ in range(self.data_max_length)])
            self.label_set[i] = label_matrix
        self.label_set = np.array(self.label_set)

        # 添加损失mask
        self.lost_mask = np.ndarray(shape=(self.data_max_length, len(self.data_set)), dtype=int)
        for i in range(self.data_max_length):
            for j in range(len(self.data_actual_length)):
                if i < self.data_actual_length[j]:
                    self.lost_mask[i][j] = self.data_actual_length[j]
                else:
                    self.lost_mask[i][j] = 0

        pass

    def __generate_batch(self):
        # 构建batch
        encoder_inputs = self.data_set[self.data_index:self.data_index + self.batch_size, :]
        encoder_inputs_actual_length = self.data_actual_length[self.data_index:self.data_index + self.batch_size]
        decoder_targets_matrix = self.label_set[self.data_index:self.data_index + self.batch_size, :]
        decoder_lost_mask = self.lost_mask[:, self.data_index:self.data_index + self.batch_size]

        self.data_index += self.batch_size

        return encoder_inputs, encoder_inputs_actual_length, decoder_targets_matrix, decoder_lost_mask

    def __encoder(self, inputs, reuse=None):
        with tf.variable_scope('encoder', reuse=reuse):
            # 正向
            fw_cell = LSTMCell(num_units=self.hidden_num, forget_bias=1.0)
            encoder_f_cell = DropoutWrapper(fw_cell, output_keep_prob=0.5)
            # 反向
            bw_cell = LSTMCell(num_units=self.hidden_num, forget_bias=1.0)
            encoder_b_cell = DropoutWrapper(bw_cell, output_keep_prob=0.5)

            # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出是一个元组 每一个元素也是这种形状
            output, (encoder_fw_final_state, encoder_bw_final_state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_f_cell, cell_bw=encoder_b_cell, inputs=inputs,
                sequence_length=self.encoder_inputs_actual_length, dtype=tf.float32, time_major=True)

            # hiddens的长度为2，其中每一个元素代表一个方向的隐藏状态序列，将每一时刻的输出合并成一个输出
            output = tf.concat(output, axis=2)

            encoder_final_state_c = tf.concat(
                (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

            encoder_final_state_h = tf.concat(
                (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

            encoder_final_state = LSTMStateTuple(
                c=encoder_final_state_c,
                h=encoder_final_state_h
            )

        return output, encoder_final_state

    def __decoder(self, helper, reuse=None):
        with tf.variable_scope('decoder', reuse=reuse):
            # 1.循环单元的构建,batch-major
            # memory = tf.transpose(self.encoder_outputs, [1, 0, 2])
            # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            #     num_units=self.hidden_num, memory=memory,
            #     memory_sequence_length=self.encoder_inputs_actual_length)
            cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_num * 2)
            # attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            #     cell, attention_mechanism, attention_layer_size=self.hidden_num)

            # 1.1每个单元输入到输出的线性映射
            # out_cell = tf.contrib.rnn.OutputProjectionWrapper(
            #     attn_cell, self.classes_num, reuse=reuse
            # )

            # 1.2 全连接输出层
            output_layer = tf.layers.Dense(self.classes_num,
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1))

            # 2.decoder构建
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cell, helper=helper,
                initial_state=self.encoder_final_state,
                output_layer=output_layer)

            # 3.获取decode结果
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, impute_finished=True, maximum_iterations=self.data_max_length)
        return final_outputs

    def __initial_fn(self):
        """
        获取第一个时间节点的输入
        :return:
        """
        initial_elements_finished = (0 >= self.encoder_inputs_actual_length)  # all False at the initial step
        initial_input = self.encoder_outputs[self.encoder_hidden_index]
        return initial_elements_finished, initial_input

    def __sample_fn(self, time, outputs, state):
        """
        根据每个一时刻cell的输出，获得出现概率最大的类别
        :param time: 当前循环的次数，及第t时刻
        :param outputs: 该时刻RNN的输出
        :param state: 该时刻接收的隐藏状态
        :return:
        """
        # 选择logit最大的下标作为sample
        prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
        return prediction_id

    def __next_inputs_fn(self, time, outputs, state, sample_ids):
        """
        根据上一个时刻的输出，决定当前时刻t的输入和隐藏态
        :param time: 当前循环的次数，及第t时刻
        :param outputs: 上一时刻RNN的输出
        :param state: 上一时刻的隐藏状态
        :param sample_ids: 上一时刻RNN的采样类别
        :return:
        """
        next_input = self.encoder_outputs[self.encoder_hidden_index]
        elements_finished = (
            time >= self.encoder_inputs_actual_length)  # this operation produces boolean tensor of [batch_size]
        all_finished = tf.reduce_all(elements_finished)  # boolean scalar
        pad_step_embedded = tf.zeros([self.batch_size, self.hidden_num * 2],
                                     dtype=tf.float32)
        next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
        next_state = state
        return elements_finished, next_inputs, next_state

    def build_graph(self):
        # 构建模型训练所需的数据流图
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 定义输入输出
            self.encoder_inputs_actual_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
            self.encoder_inputs = tf.placeholder(dtype=tf.float32,
                                                 shape=[self.batch_size, self.data_max_length, self.data_dimension])
            self.decoder_targets_matrix = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.data_max_length,
                                                                                self.data_max_length])
            self.decoder_lost_mask = tf.placeholder(dtype=tf.int32, shape=[self.data_max_length, self.batch_size])

            # Encoder Bi-LSTM，time-major
            encoder_inputs_time_majored = tf.transpose(self.encoder_inputs, [1, 0, 2])
            with tf.name_scope('Encoder'):
                self.encoder_outputs, self.encoder_final_state = self.__encoder(encoder_inputs_time_majored)

            # n-Decoder
            with tf.name_scope('Decoders'):
                for i in range(self.encoder_outputs.get_shape()[0].value):
                    self.encoder_hidden_index = i
                    with tf.name_scope('Decoder-' + str(i)):
                        # 自定义Decoder
                        custom_helper = tf.contrib.seq2seq.CustomHelper(self.__initial_fn, self.__sample_fn,
                                                                        self.__next_inputs_fn)
                        if i == 0:
                            self.decoder_outputs_arr.append(self.__decoder(custom_helper))
                        else:
                            self.decoder_outputs_arr.append(self.__decoder(custom_helper, reuse=True))

            # 定义损失
            with tf.name_scope('Loss'):
                decoder_targets_time_majored = tf.transpose(self.decoder_targets_matrix, [1, 0, 2])
                for i in range(len(self.decoder_outputs_arr)):
                    self.decoder_prediction_matrix.append(self.decoder_outputs_arr[i].sample_id)

                    # 获取target
                    decoder_true_target = decoder_targets_time_majored[i]

                    # 定义mask，使padding不计入loss计算
                    mask = tf.to_float(tf.sequence_mask(self.decoder_lost_mask[i], self.data_max_length))
                    # mask = tf.transpose(mask, [1, 0])

                    # 定义标注的总损失
                    if self.loss == None:
                        self.loss = tf.contrib.seq2seq.sequence_loss(
                            self.decoder_outputs_arr[i].rnn_output, decoder_true_target, weights=mask)
                    else:
                        self.loss += tf.contrib.seq2seq.sequence_loss(self.decoder_outputs_arr[i].rnn_output,
                                                                      decoder_true_target, weights=mask)

            # 优化并进行梯度修剪
            with tf.name_scope('Train'):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                # 分解成梯度列表和变量列表
                grads, vars = zip(*optimizer.compute_gradients(self.loss))

                # 梯度修剪
                gradients, _ = tf.clip_by_global_norm(grads, 5)  # clip gradients

                # 将每个梯度以及对应变量打包
                self.train_op = optimizer.apply_gradients(zip(gradients, vars))

            # 设置模型存储所需参数
            self.saver = tf.train.Saver()

    def train(self):
        with tf.Session(graph=self.graph) as self.session:
            # 判断模型是否存在
            if os.path.exists(self.model_save_checkpoint):
                # 恢复变量
                print('restoring------')
                self.saver.restore(self.session, self.model_save_name)
            else:
                # 初始化变量
                print('Initializing------')
                tf.initialize_all_variables().run()

            # 开始迭代 使用Adam优化的随机梯度下降法
            print('training------')
            for i in range(self.train_num):
                # 采样参数
                self.data_index = 0
                # 开始训练
                for step in range(int(self.steps)):
                    encoder_inputs, encoder_inputs_actual_length, decoder_targets_matrix, decoder_lost_mask = self.__generate_batch()
                    feed_dict = {self.encoder_inputs_actual_length: encoder_inputs_actual_length,
                                 self.encoder_inputs: encoder_inputs,
                                 self.decoder_targets_matrix: decoder_targets_matrix,
                                 self.decoder_lost_mask: decoder_lost_mask}
                    self.session.run([self.train_op], feed_dict=feed_dict)

                # test
                self.data_index = int(self.steps) - 1
                encoder_inputs, encoder_inputs_actual_length, decoder_targets_matrix, decoder_lost_mask = self.__generate_batch()
                loss = self.session.run([self.loss],
                                        feed_dict={self.encoder_inputs_actual_length: encoder_inputs_actual_length,
                                                   self.encoder_inputs: encoder_inputs,
                                                   self.decoder_targets_matrix: decoder_targets_matrix,
                                                   self.decoder_lost_mask: decoder_lost_mask})

                print('Average loss at step %d: %f' % (i, loss[0]))

            # 保存模型
            print('save model------')
            self.saver.save(self.session, self.model_save_name)

        pass

    def calculate(self):
        with tf.Session(graph=self.graph) as self.session:
            self.saver.restore(self.session, self.model_save_name)

            self.data_index = 0
            encoder_inputs, encoder_inputs_actual_length, decoder_targets_matrix, decoder_lost_mask = self.__generate_batch()
            feed_dict = {self.encoder_inputs_actual_length: encoder_inputs_actual_length,
                         self.encoder_inputs: encoder_inputs,
                         self.decoder_targets_matrix: decoder_targets_matrix,
                         self.decoder_lost_mask: decoder_lost_mask}
            result = self.session.run([self.decoder_outputs_arr[i].sample_id for i in range(self.data_max_length)],
                                      feed_dict=feed_dict)

            return result
