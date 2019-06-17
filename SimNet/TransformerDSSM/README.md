# 基于Transformer的语义相似度计算模型DSSM
该模型在DSSM模型的基础上，将模型的表示层使用基于Transformer的Encoder部分来实现，匹配层将通过表示层得到问题query和答案answer的特征表示进行余弦相似度计算，由于问题i除了与答案i相匹配以外，其余答案均为问题i的负样本，因此需要对每一个问题进行负采样。

在该模块中，主要包含了以下4个部分内容：
* [模块内容](#模块内容)
  * [模型实现代码](#模型实现代码)
  * [模型调用方式](#模型调用方式)
  * [模型训练数据](#模型训练数据)
  * [已训练模型库](#已训练模型库)

## 模型实现代码
模型的实现代码位于目录：`/NlpModel/SimNet/TransformerDSSM/Model/TransformerDSSM.py`，其实现顺序从`TransformerDSSM`类开始

首先，通过调用`build_graph_by_cpu`或者`build_graph_by_gpu`对模型整个数据流图进行构建，以上两种构建方式，分别对应着模型的cpu版本和gpu版本。