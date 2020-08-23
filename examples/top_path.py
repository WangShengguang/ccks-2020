import logging


def add_root_path():
    import sys
    from pathlib import Path
    cur_dir = Path(__file__).absolute().parent
    times = 10
    while not str(cur_dir).endswith('ccks-2020') and times:
        cur_dir = cur_dir.parent
        times -= 1
    print(cur_dir)
    sys.path.append(str(cur_dir))


add_root_path()


# ---------------------


def get_most_overlap_path(q_text, paths):
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
        score1 = len(common_words)
        score2 = len(common_words) / len(path_words)  # 跳数 or len(path_words)
        score3 = len(common_words) / len(path)
        score = score1 + score2  # + score3
        if score > max_score:
            logging.info(f'score :{score}, score1 :{score1}, score2 :{score2}, score3 :{score3}, '
                         f'top_path: {top_path}')
            top_path = path
            max_score = score
    return top_path


def main():
    q='莫妮卡·贝鲁奇的代表作？'


if __name__ == '__main__':
    main()