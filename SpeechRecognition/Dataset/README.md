# 1.模型标注数据
模型标注数据文件位于：https://github.com/makeplanetoheaven/NlpModel/tree/master/SpeechRecognition/Dataset

其文件中每行代表一条数据，其格式为：

```clike
wav_data_path \t pinyin_list \t hanzi_list \n
```
其数据名称和数据量如下：

Name | total | train | dev | test
--- | --- | --- | --- | ---
aishell 1 | 141593 | 120098 | 14322 | 7173
primewords set 1 | 50902 | - | - | -
thchs-30 | 13388 | 10000 | 893 | 2495
st-cmd | 102597 | - | - | -
magicdata | 608756 | 572723 | 11776 | 24257 |
aidatatang | - | - | - | - |

# 2.模型训练数据

包括【**st-cmd、primewords、aishell 、thchs-30、magicdata、aidatatang**】六个数据集，共计约【**1385**】小时

若需要使用所有数据集，只需解压到统一路径下，然后设置数据所在根目录路径即可。

下面分别为开源数据及对应下载链接

Name | total | train | dev | test | link
--- | --- | --- | --- | --- | ---
aishell 1 | 178h | - | - | - | [点击](http://www.aishelltech.com/kysjcp)
primewords set 1 | 100h | - | - | - | [点击](http://www.openslr.org/47/)
thchs-30 | 30h | - | - | - |  [点击](http://www.openslr.org/18/)
st-cmd | 122h | - | - | - | [点击](https://openslr.org/38/)
magicdata | 755h | 712.09h | 14.84h | 28.08h | [点击](http://www.imagicdatatech.com/index.php/home/dataopensource/data_info/id/101)
aidatatang | 200h | - | - | - | [点击](http://www.openslr.org/62/)
