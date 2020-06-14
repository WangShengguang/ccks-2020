import gc
import logging
import os

import jieba
from LAC import LAC
from LAC.ahocorasick import Ahocorasick
from tqdm import tqdm

from ckbqa.utils.decorators import singleton
from ckbqa.utils.tools import tqdm_iter_file, pkl_dump, pkl_load, json_load, json_dump
from config import Config


class Ngram(object):
    def __init__(self):
        pass

    def ngram(self, text, n):
        ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
        return ngrams

    def get_all_grams(self, text):
        for n in range(2, len(text)):
            for ngram in self.ngram(text, n):
                yield ngram
                # for gram in ngrams:
                #     yield gram


@singleton
class JiebaLac(object):
    def __init__(self, load_custom_dict=True):
        # jieba.enable_paddle()  # 启动paddle模式。 0.40版之后开始支持，早期版本不支持
        if load_custom_dict:
            self.load_custom_dict()

    def load_custom_dict(self):
        if not os.path.isfile(Config.jieba_custom_dict):
            all_ent_mention_words = set(json_load(Config.mention2count_json).keys())
            entity_set = set(json_load(Config.entity2id).keys())
            for ent in tqdm(entity_set, total=len(entity_set), desc='create jieba words '):
                if ent.startswith('<'):
                    ent = ent[1:-1]
                all_ent_mention_words.add(ent)
            # FREQ,total,
            # 模仿jieba.add_word，重写逻辑，加速
            # jieba.dt.FREQ = {}
            # jieba.dt.total = 0
            for word in tqdm(all_ent_mention_words, desc='jieba custom create '):
                freq = len(word) * 3
                jieba.dt.FREQ[word] = freq
                jieba.dt.total += freq
                for ch in range(len(word)):
                    wfrag = word[:ch + 1]
                    if wfrag not in jieba.dt.FREQ:
                        jieba.dt.FREQ[wfrag] = 0
            del all_ent_mention_words
            gc.collect()
            json_dump({'dt.FREQ': jieba.dt.FREQ, 'dt.total': jieba.dt.total},
                      Config.jieba_custom_dict)
            logging.info('create jieba custom dict done ...')
        # load
        jieba_custom = json_load(Config.jieba_custom_dict)
        jieba.dt.check_initialized()
        jieba.dt.FREQ = jieba_custom['dt.FREQ']
        jieba.dt.total = jieba_custom['dt.total']
        logging.info('load jieba custom dict done ...')

    def cut_for_search(self, text):
        return jieba.cut_for_search(text)

    def cut(self, text):
        return jieba.cut(text)


@singleton
class BaiduLac(LAC):
    def __init__(self, model_path=None, mode='lac', use_cuda=False,
                 _load_customization=True, reload=False,
                 customization_path=Config.lac_custom_dict_txt):
        super().__init__(model_path=model_path, mode=mode, use_cuda=use_cuda)
        self.mode = mode  # lac,seg
        self.reload = reload
        self.customization_path = customization_path
        if _load_customization:
            self._load_customization()

    def _save_customization(self):
        # lac = LAC(mode=self.mode)  # 装载LAC模型,LAC(mode='seg')
        # lac.load_customization(self.customization_path)  # 35万2min;100万>20min;
        parms_dict = {'dictitem': self.custom.dictitem,
                      # 'ac': self.custom.ac,
                      'mode': self.mode}
        pkl_dump(parms_dict, Config.lac_model_pkl)
        return Config.lac_model_pkl

    def _load_customization(self):
        logging.info('暂停载入BaiduLac自定词典 ...')
        return
        # self.custom = Customization()
        # if self.reload or not os.path.isfile(Config.lac_model_pkl):
        #     self.custom.load_customization(self.customization_path)  # 35万2min;100万，20min;
        #     self._save_customization()
        # else:
        #     logging.info('load lac customization start ...')
        #     parms_dict = pkl_load(Config.lac_model_pkl)
        #     self.custom.dictitem = parms_dict['dictitem']  # dict
        #     self.custom.ac = Ahocorasick()
        #     for phrase in tqdm(parms_dict['dictitem'], desc='load baidu lac '):
        #         self.custom.ac.add_word(phrase)
        #     self.custom.ac.make()
        #     logging.info('loaded lac customization Done ...')


class Customization(object):
    """
    基于AC自动机实现用户干预的功能
    """

    def __init__(self):
        self.dictitem = {}
        self.ac = None
        pass

    def load_customization(self, filename):
        """装载人工干预词典"""
        self.ac = Ahocorasick()

        for line in tqdm_iter_file(filename, prefix='load_customization '):
            words = line.strip().split()
            if len(words) == 0:
                continue

            phrase = ""
            tags = []
            offset = []
            for word in words:
                if word.rfind('/') < 1:
                    phrase += word
                    tags.append('')
                else:
                    phrase += word[:word.rfind('/')]
                    tags.append(word[word.rfind('/') + 1:])
                offset.append(len(phrase))

            if len(phrase) < 2 and tags[0] == '':
                continue

            self.dictitem[phrase] = (tags, offset)
            self.ac.add_word(phrase)
        self.ac.make()

    def parse_customization(self, query, lac_tags):
        """使用人工干预词典修正lac模型的输出"""

        def ac_postpress(ac_res):
            ac_res.sort()
            i = 1
            while i < len(ac_res):
                if ac_res[i - 1][0] < ac_res[i][0] and ac_res[i][0] <= ac_res[i - 1][1]:
                    ac_res.pop(i)
                    continue
                i += 1
            return ac_res

        if not self.ac:
            logging.warning("customization dict is not load")
            return

        ac_res = self.ac.search(query)

        ac_res = ac_postpress(ac_res)

        for begin, end in ac_res:
            phrase = query[begin:end + 1]
            index = begin

            tags, offsets = self.dictitem[phrase]
            for tag, offset in zip(tags, offsets):
                while index < begin + offset:
                    if len(tag) == 0:
                        lac_tags[index] = lac_tags[index][:-1] + 'I'
                    else:
                        lac_tags[index] = tag + "-I"
                    index += 1

            lac_tags[begin] = lac_tags[begin][:-1] + 'B'
            for offset in offsets:
                index = begin + offset
                if index < len(lac_tags):
                    lac_tags[index] = lac_tags[index][:-1] + 'B'
