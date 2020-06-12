# _*_ coding:utf-8 _*_
"""
@time: 2020年6月12日
@desc:实体识别类
"""
import os,sys
base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from ALBERT_CRF_NER import AlbertNerModel
import re
from utils import brace_remove
from models import conf


class NER(object):

    def __init__(self,model_name:str,path:str,**kwargs):
        """
        :param model_name: 模型名称
        :param path: 模型路径
        :param kwargs: 其他配置项，或预训练模型配置（e.g.:config_path,checkpoint_path,dict_path) 等等
        """
        if str(model_name).upper() == 'LTP':
            self.type=0
        elif str(model_name).upper().startswith('ALBERT'):  # 目前支持albert_base,albert_small,albert_tiny
            self.type=1
        else:
            self.type=-1
        ##为便于以weights形式模型加载使用，初始化时构建好了模型
        self.model=self.__get_model_instance(model_name,path,**kwargs)
        pass

    def __get_model_instance(self,model_name,path,**kwargs):

        if self.type==0:
            pass
        elif self.type==1: # 目前支持albert_base,albert_small,albert_tiny
            return AlbertNerModel(model_name=model_name,path=path,**kwargs)
        else:
            raise NotImplementedError


    def entity_recognize(self,sentence)->dict:

        """
        实体识别方法，根据选择的模型，做对应处理
        :param sentence: 待识别句子
        :param ner_type: 返回识别的实体类型 0：ORG | 1：LOC | 2：PER
        :return: org_entity_dict, 此处仅返回组织；后续若有其他返回要求（比如，日期、地点）做对应返回处理即可

        """
        entity_dict={}
        org_entity_dict={}
        loc_entity=None
        per_entity=None
        if self.type==0:
            pass
        elif self.type==1:
            org_entity_dict,loc_entity=self.__do_albert_ner(brace_remove(sentence))

        entity_dict['ORG']=org_entity_dict
        entity_dict['LOC']=loc_entity
        entity_dict['PER']=per_entity
        return entity_dict
        pass
# region LTP 实体识别

# endregion
    def handle_regix_charactors(self,entity_str:str):
        '''
        日期：2020年5月17日
        修复由于英文括弧引起的正则匹配异常的bug
        '''
        return entity_str.replace('(','（').replace('[','（').replace('{','（').replace(')','）')\
        .replace(']','）').replace('}','）')
# region ALBERT  + CRF
    def __do_albert_ner(self,sentence):
        """
        因为项目中不需要识别人名，所以PER给过滤掉了
        """
        #这里为了防止以weight形式加载模型时忘记手动构建模型结构的情况，做了模型的build处理
        if not self.model.albert_model:
            self.model.build_albert_model()
        org_entity_dict = {}
        entity_list = self.model.recognize(sentence)
        print('ner results:',entity_list)
        org_list = []
        if entity_list:
            for tpl in entity_list:
                if tpl[1] == 'org'.upper() or tpl == 'org':
                    org_list.append(self.handle_regix_charactors(tpl[0]))
        if len(org_list) > 0:
            result1 = '|'.join(org_list)
            entity_compile=None
            try:
                entity_compile = re.compile(result1)
            except:
                print('------------------->errsentence<-----------------------------:\n',sentence)
                import traceback
                traceback.print_exc()
                raise
            entity_result = entity_compile.finditer(sentence)
            for entity in entity_result:
                if entity.group():
                    print(entity.group())
                    if len(entity.group()) > 1:  # 去除单字实体
                        org_entity_dict[entity.group()] = entity.span()

        loc_entity=None
        entity_list = self.model.recognize(sentence)
        if entity_list:
            for tpl in entity_list:
                if tpl[1] == 'loc'.upper():
                    loc_entity=self.handle_regix_charactors(tpl[0])
                    break
        return org_entity_dict,loc_entity

if __name__=='__main__':
# region small

    weight_path = conf.weight_path
    model_path = conf.model_path
    config_path = model_path + r'/albert_config_small_google.json'
    checkpoint_path = model_path + '/albert_model.ckpt'
    dict_path = model_path + '/vocab.txt'
# endregion

# region    tiny
#     weight_path=r'E:/hcl/ALBERT_NER_KERAS-master/albert_tiny_lstm_crf_ner_unshared.weights'
#     model_path = r'E:\hcl\ALBERT_NER_KERAS-master\albert_tiny'
#     config_path = model_path + r'/albert_config.json'
#     checkpoint_path = model_path + '/albert_model.ckpt'
#     dict_path = model_path + '/vocab.txt'
# endregion
    ner=NER(model_name='albert_small',path=weight_path,config_path=config_path,checkpoint_path=checkpoint_path,dict_path=dict_path,unshared=False)

    albert_ner_path = conf.albert_ner_path
    if not ner.model.albert_model:
        if os.path.exists(albert_ner_path):
            ner.model.albert_model = albert_ner_path
        else:
            ner.model.build_albert_model()
    print(ner.entity_recognize('张晓强2019年12月16日担任上海浦东发展银行股份有限公司法定代表人,董事长。'))
    print(ner.entity_recognize(' 现任上海浦东发展银行股份有限公司党委书记。'))