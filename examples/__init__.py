
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
