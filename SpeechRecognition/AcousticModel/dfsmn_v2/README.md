# 背景知识
FSMN和DFSMN系列模型的结构及其实现原理可参考如下两篇博客：
1.[FSMN结构快速解读](https://www.cnblogs.com/machine-lyc/p/10572936.html)
2.[DFSMN结构快速解读](https://www.cnblogs.com/machine-lyc/p/10573743.html)

# 基于CNN+DFSMN的声学模型实现
本模型是在传统CNN模型的基础上，引入2018年阿里提出的声学模型DFSMN，论文地址：https://arxiv.org/pdf/1803.05030.pdf。

该声学模型使用的输入是具有16KHZ采样率，单声道音频数据经过fbank特征提取以后的特征数据。

DFSMN结构如下图，与[语音识别|基于CNN+DFSMN（简化版：标量+无步长因子）的声学模型实现及代码开源（keras）](https://blog.csdn.net/qq_28385535/article/details/100236023)相比：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024104328759.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI4Mzg1NTM1,size_16,color_FFFFFF,t_70)

**（1）简化版**

　　在记忆单元计算上，使用的是类似sfsmn中的标量权重来计算第$t$时刻的隐藏状态，且没有引入步长因子$stride$，即

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026171531488.png)

其中，$l$表示dfsmn的第$l$层，$t$表示第$t$时刻的隐藏状态，$l\_mem\_siz$表示前向记忆单元长度，$r\_mem\_siz$表示后向记忆单元长度，$mem\_weight$用于存储权重，是一个长度为$l\_mem\_si+r\_mem\_siz+1$一维向量。

**（2）完整版**

　　在记忆单元计算上，使用的是类似vfsmn中的向量权重来计算第$t$时刻的隐藏状态，且引入步长因子$stride$，即

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026171604377.png)

此时，$mem\_weight$是一个形状为$[l\_mem\_si+r\_mem\_siz+1，hidden\_num]$的二维矩阵。

在该模块中，主要包含了以下4个部分内容：
* [模型实现代码](#模型实现代码)
  * [1.卷积层](##1.卷积层)
  * [2.DFSMN层](##2.DFSMN层)
  * [3.softmax层](##3.softmax层)
  * [4.梯度更新部分](#4.梯度更新部分)
* [模型调用方式](#模型调用方式)
* [模型训练数据](#模型训练数据)
* [已训练模型库](#已训练模型库)

模型结构和模型调用方式和简化版基本相同，这里不再描述，主要区别在dfsmn单元的实现方式。

## 已训练模型库
Name | WER | date | link | extraction code | ps
--- | --- | --- | --- | --- | ---
3*cnn[64-1]-6*dfsmn[2048-512-20-20-2]-1289 | 16% | 2020.1.5 | [点击](https://pan.baidu.com/s/15l_Cn62CefNAtgoSrtUa5A) | vzk4 | 该模型可继续训练，目前效果最好为13% WER

# 更新内容
## 2019.12.7
* 优化fbank特征提取方式，采用80维的log mel fbank
* 优化数据读取方式，模型训练前需要调用generate_data_set()函数
* 优化记忆单元的计算方式
## 2020.1.5
* 优化gpu模型训练方式，可指定具体的gpu核
* 引入混合精度进行模型训练,

# Bug汇总
* 该模型由于v2版本记忆单元的计算方式，最少能够处理1.615s的音频文件，解决方案：1.在模型预测时可使用v1版本的记忆单元计算；2.对输入数据补长