# AlbertNerDemo
#附：对bojone模型加载后预测失败的个人解读及解决方案
(bojone源码链接：[https://github.com/bojone/bert4keras/blob/master/examples/task_sequence_labeling_ner_crf.py])

#1. 问题  以example中的NER任务为例，模型核心源码如下：

```
model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output
output = Dense(num_labels)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()
```

该任务源码中，苏神采用的是Bert/Albert+CRF的方案，其中，Bert/Albert 用来做embedding，Dense后接CRF。关键在CRF层，这里苏神使用的是**自定义**的 ConditionalRandomField（**敲黑板**!）。
根据keras中对自定义layer的限定，模型保存后想通过简单的一句code: load_model('.h5') 往往会报错。*因为自定义层并非keras自带layer，若不加说明，keras会采用默认的model加载方式，这时很有可能出现自定义的loss提示找不到或者自定义layer变量无法初始化等情况。

#2. 解决思路  鉴于上述情况，个人解决方案如下：

- 首先，模型正常训练，在model保存时，若是仅仅需要权重，仅保存*.weights，否则更改为save('*.h5')

- 其次，由于模型保存时，有.weights 及 .h5两种情况，此处需要分开说明。

##2.1 模型以.weight 保存  
此时，由于仅仅保存了模型权重，预测时需要重新构建模型拓扑结构后加载权重。我的方案是将模型训练时的拓扑结构code提取出来，另辟一个类，初始化模型拓扑结构后，加载weights即可预测。（**注意，这里有一前提是所构建的模型拓扑结构一定要与训练模型拓扑结构相同**）

##2.2 模型以.h5 保存 
此时，模型的拓扑结构连同相应权重均被保存到了hdf5文件中，若是模型训练时采用的均为keras内置layer，模型可以正常加载运行。否则，需要依据模型拓扑结构加载相应layer，及其权重（**敲黑板x3！！！**），此时模型加载时需要额外指出对应参数加载。到这里，基本上解决了自定义layer或是loss的加载问题。

*但是，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，还没完。*

对于上述链接中的NER 例子，可以看到模型推断时苏神使用了CRF中的trans方法：

`NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])`

*这里的CRF是模型的CRF，所以若是参考2.2简简单单加载完模型后会发现预测结果奇差！！*
我的解决方案是，**既然模型推断需要用到自定义layer中的函数，那么就将这个模型中该layer的相应函数传给它（话有点拗口，但就是这个意思。。。）。**

代码请参考model文件夹中NER_Demo.py
