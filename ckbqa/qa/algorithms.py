import logging
import re
from typing import Set

legal_char_partten = re.compile(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])")  # 中英文和数字


def sequences_set_similar(s1: Set, s2: Set):
    overlap = len(s1 & s2)
    jaccard = overlap / len(s1 | s2)  # 集合距离
    # 词向量相似度
    # wordvecsim = model.similarity(''.join(s1),''.join(s2))
    return overlap, jaccard


class Algorithms(object):
    def __init__(self):
        self.invalid_names = {'是什么', ''}  # 汽车的动力是什么

    def get_most_overlap_path(self, q_text, paths):
        # 从排名前几的tuples里选择与问题overlap最多的
        max_score = 0
        top_path = []
        q_words = set(q_text)
        pure_qtext = legal_char_partten.sub('', q_text)
        for path in paths:
            path_words = set()
            score5 = 0
            for ent_rel in path:
                if ent_rel.startswith('<'):
                    ent_rel = ent_rel[1:-1].split('_')[0]
                path_words.update(set(ent_rel))
                pure_ent = legal_char_partten.sub('', ent_rel)
                if pure_ent not in self.invalid_names and pure_ent in pure_qtext:
                    score5 += 1
                    # print(pure_ent)
            common_words = path_words & q_words  # 交集
            score1 = len(common_words)  # 主score,交集长度  ；权重排名:1>2>3>4
            score2 = 1 / len(path_words)  # 候选词长度越短越好，权重4
            score3 = 1 / len(path)  # 跳数越少越好，权重2
            score4 = 0.5 if path[-1].startswith('<') else 0  # 实体大于属性，实体加分，权重3；会造成越长分数越高
            score = score1 + score2 + score3 + score4 + score5
            # log_str = (f'score : {score:.3f}, score1 : {score1}, score2 : {score2:.3f}, '
            #            f'score3 : {score3:.3f}, score4 : {score4}, score5: {score5}, '
            #            f'path: {path}')
            # logging.info(log_str)
            # print(log_str)
            if score > max_score:
                top_path = path
                max_score = score
        # logging.info(f'score: {max_score}, top_path: {top_path}')
        return top_path, max_score


if __name__ == '__main__':
    # q_text = '怎样的检查项目能对小儿多源性房性心动过速、急性肾>功能不全以及动静脉血管瘤做出检测？'
    # out_paths = [['<肾功能不全>', '<>来源>']]
    # in_paths = [['肾功能', '<检查项目>', '<标签>']]
    # q_text = '演员梅艳芳有多高？'
    # out_paths = [['<梅艳芳>', '<身高>']]  # 6.2
    # in_paths = [['<梅艳芳>', '<演员>']]  # 8.2
    q_text = '叶文洁毕业于哪个大学？'
    out_paths = [['<叶文洁>', '<毕业院校>', '<学校代码>']]  # 7.9333
    in_paths = [['<大学>', '<毕业于>', '<类型>']]  # 7.9762
    algo = Algorithms()
    algo.get_most_overlap_path(q_text, out_paths)
    algo.get_most_overlap_path(q_text, in_paths)
