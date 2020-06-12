# _*_ coding:utf-8 _*_
"""
@time: 2020年6月12日
@author: hecl
@desc: Named Entity Recognition based on ALBERT family
@notice：模型拓扑结构不正确，或者crf结果解析有误，会严重影响结果，
ps，若模型加载后预测结果很差，请仔细阅读注释，尤其是133行
"""
import os,numpy as np
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" ##默认不使用GPU
from bert4keras.backend import K
K.set_learning_phase(0)
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import ViterbiDecoder
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model


class AlbertNerModel(object):
    # model=None
    def __init__(self,
                 model_name:str,
                 path:str,
                 config_path:str,
                 checkpoint_path:str,
                 dict_path:str,
                 layers: int = 0,
                 unshared:bool=False
                 ):
        """
        Albert 初始化参数
        :param model_name: 模型名称，albert_base/albert_small/albert_tiny, 不推荐albertbase/albertsmall/alberttiny
        :param path: 权重路径
        :param config_path: 预训练模型配置文件
        :param checkpoint_path: 预训练模型文件
        :param dict_path: 预训练模型字典
        :param layers: 可选自定义层数，base最大12层，small最大6层，tiny最大4层
        :param unshared: 是否以Bert形式做层分解，默认为否
        """
        if tf.__version__ >= '2.0':
            print('暂不支持tensorflow 2.0 以上版本')
            raise
        self.weight_path=path
        self.__maxlen=256
        self.__crf_lr_multiplier=1000

        if str(model_name).upper()=='ALBERT_BASE' or str(model_name).upper()=='ALBERTBASE':
            self.albert_layers=12
        elif str(model_name).upper()=='ALBERT_SMALL' or str(model_name).upper()=='ALBERTSMALL':
            self.albert_layers=6
        elif str(model_name).upper()=='ALBERT_TINY' or str(model_name).upper()=='ALBERTTINY':
            self.albert_layers=4
        if layers > 0:
            self.albert_layers=layers
        self.pretrain_name=model_name
        self.config=config_path
        self.checkpoint=checkpoint_path
        self.dict=dict_path
        self.unshared=unshared

        self.tokenizer= Tokenizer(self.dict, do_lower_case=True)
        # 类别映射
        labels = ['PER', 'LOC', 'ORG']
        id2label = dict(enumerate(labels))
        # label2id={j: i for i,j in id2label.items()}
        self.__id2label=id2label
        self.__num_labels = len(labels) * 2 + 1
        # label2id = {j: i for i, j in id2label.items()}
        assert self.config and self.checkpoint and self.dict
        # self.__crf= ConditionalRandomField(lr_multiplier=self.crf_lr_multiplier)
        self.__crf= None
        self._model=None


# region 为便于多模型配置调试，对所有配置参数做setter处理,配置完毕需要重新build model

    def set_layers(self,value):
        self.albert_layers=value

    def set_unshared(self,value):
        self.unshared=value

    def set_dict_path(self,path):
        self.dict=path
        self.tokenizer=Tokenizer(self.dict, do_lower_case=True)

    def set_checkpoint_path(self,path):
        self.checkpoint=path

    def set_config_path(self,path):
        self.config=path

    def set_weight_path(self,weight_path):
        self.weight_path=weight_path
# endregion

    @property
    def maxlen(self):
        return self.__maxlen

    @maxlen.setter
    def maxlen(self,value):
        self.__maxlen=value

    @property
    def crf_lr_multiplier(self):
        return self.__crf_lr_multiplier

    @ crf_lr_multiplier.setter
    def crf_lr_multiplier(self,value):
        self.__crf_lr_multiplier=value

    @property
    def albert_model(self):
        return self._model
    @albert_model.setter
    def albert_model(self,model_path:str):
        from keras.models import load_model
        from keras.utils import CustomObjectScope
        # self.__model=load_model(model_path,custom_objects={'ConditionalRandomField':
        #                             ConditionalRandomField,
        #                         'sparse_loss':ConditionalRandomField.sparse_loss},
        #                         compile=False)##两种自定义loss加载方式均可
        with CustomObjectScope({'ConditionalRandomField':
                                    ConditionalRandomField,
                                'sparse_loss':ConditionalRandomField.sparse_loss}):
            self._model=load_model(model_path)
            ##此处是重点！！，本机电脑及服务器上model中crf层名字如下，实际情况若名称不一致，需根据模型拓扑结构中的名字更改！！！
            self.__crf=self._model.get_layer('conditional_random_field_1')
            assert isinstance(self.__crf,ConditionalRandomField)
    @albert_model.deleter
    def albert_model(self):
        K.clear_session()
        del self._model


    def build_albert_model(self):
        del self.albert_model
        file_name=f'albert_{self.pretrain_name}_pretrain.h5'##这里，为了方便预训练模型加载，我预先将加载后的预训练模型保存为了.h5
        if os.path.exists(file_name):
            pretrain_model=load_model(file_name,compile=False)
        else:
            pretrain_model=build_transformer_model(
                config_path=self.config,checkpoint_path=self.checkpoint,
                model='albert_unshared' if self.unshared else 'albert',
                return_keras_model=True
            )

        if not self.unshared:
            output_layer = 'Transformer-FeedForward-Norm'
            output=pretrain_model.get_layer(output_layer).get_output_at(self.albert_layers-1)
        else :
            output_layer='Transformer-%s-FeedForward-Norm' % (self.albert_layers - 1)
            output = pretrain_model.get_layer(output_layer).output
        output=Dense(self.__num_labels)(output)
        self.__crf = ConditionalRandomField(lr_multiplier=self.crf_lr_multiplier)
        output = self.__crf(output)
        model = Model(pretrain_model.input, output)
        model.load_weights(self.weight_path)
        self._model = model

    def viterbi_decode(self,nodes, trans,starts=[0],ends=[0]):
        """Viterbi算法求最优路径
        """
        num_labels=len(trans)
        non_starts=[]
        non_ends=[]
        if starts is not None:
            for i in range(num_labels):
                if i not in starts:
                    non_starts.append(i)
        if ends is not None:
            for i in range(num_labels):
                if i not in ends:
                    non_ends.append(i)
                # 预处理
        nodes[0, non_starts] -= np.inf
        nodes[-1, non_ends] -= np.inf
        labels = np.arange(num_labels).reshape((1, -1))
        scores = nodes[0].reshape((-1, 1))
        # scores[1:] -= np.inf  # 第一个标签必然是0
        paths = labels
        for l in range(1, len(nodes)):
            M = scores + trans + nodes[l].reshape((1, -1))
            idxs = M.argmax(0)
            scores = M.max(0).reshape((-1, 1))
            paths = np.concatenate([paths[:, idxs], labels], 0)
        return paths[:, scores[:, 0].argmax()] # 最优路径

    def recognize(self, text):
        """
        # 识别实体
        :param text:
        :return: entities list
        """
        tokens = self.tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        try:
            mapping = self.tokenizer.rematch(text, tokens)
            token_ids = self.tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            nodes = self._model.predict([[token_ids], [segment_ids]])[0]
            # print('nodes:',nodes)
            _trans=K.eval(self.__crf.trans)
            labels = self.viterbi_decode(nodes,trans=_trans)
            entities, starting = [], False
            for i, label in enumerate(labels):
                if label > 0:
                    if label % 2 == 1:
                        starting = True
                        entities.append([[i], self.__id2label[(label - 1) // 2]])
                    elif starting:
                        entities[-1][0].append(i)
                    else:
                        starting = False
                else:
                    starting = False

            return [
                (text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l) for w, l in entities
            ]
        except:
            import traceback
            traceback.print_exc()
            # print('text:', text)
            # print('tokens:', tokens)




