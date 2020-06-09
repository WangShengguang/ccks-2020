import logging
from typing import Set


def sequences_set_similar(s1: Set, s2: Set):
    overlap = len(s1 & s2)
    jaccard = overlap / len(s1 | s2)  # 集合距离
    # 词向量相似度
    # wordvecsim = model.similarity(''.join(s1),''.join(s2))
    return overlap, jaccard


class Algorithm(object):
    def __init__(self):
        pass

    def get_most_overlap_path(self, q_text, paths):
        # 从排名前几的tuples里选择与问题overlap最多的
        max_score = 0
        top_path = paths[0]
        q_words = set(q_text)
        for path in paths:
            path_words = set()
            for ent_rel in path:
                if ent_rel.startswith('<'):
                    ent_rel = ent_rel[1:-1].split('_')[0]
                path_words.update(set(ent_rel))
            common_words = path_words & q_words  # 交集
            score1 = len(common_words)  # 主score,交集长度  ；权重排名:1>2>3>4
            score2 = 1 / len(path_words)  # 候选词长度越短越好，权重4
            score3 = 6 / len(path)  # 跳数越少越好，权重2
            score4 = 1 if path[-1].startswith('<') else 0  # 实体大于属性，实体加分，权重3；会造成越长分数越高
            score = score1 + score2 + score3 + score4
            logging.info(f'score :{score:.4f}, score1 :{score1:.4f}, score2 :{score2:.4f}, '
                         f'score3 :{score3:.4f}, score4 :{score4:.4f}, '
                         f'top_path: {path}')
            if score > max_score:
                top_path = path
                max_score = score
        logging.info(f'score: {max_score}, top_path: {top_path}')
        return top_path
