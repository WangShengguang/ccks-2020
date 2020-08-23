import os
import time
import traceback


def get_available_gpu(num_gpu=1, min_memory=1000, try_times=3, allow_gpus="0,1,2,3", verbose=False):
    '''
    :param num_gpu: number of GPU you want to use
    :param min_memory: minimum memory MiB
    :param sample: try_times
    :param all_gpu_nums: accessible gpu nums 允许挑选的GPU编号列表
    :param verbose: verbose mode
    :return: str of best choices, e.x. '1, 2'
    '''
    selected = ""
    while try_times:
        try_times -= 1
        info_text = os.popen('nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv').read()
        try:
            gpu_info = [(str(gpu_id), int(memory.replace('%', '').replace('MiB', '').split(',')[1].strip()))
                        for gpu_id, memory in enumerate(info_text.split('\n')[1:-1])]
        except:
            if verbose:
                print(traceback.format_exc())
            return "Not found gpu info ..."
        gpu_info.sort(key=lambda info: info[1], reverse=True)  # 内存从高到低排序
        avilable_gpu = []
        for gpu_id, memory in gpu_info:
            if gpu_id in allow_gpus:
                if memory >= min_memory:
                    avilable_gpu.append(gpu_id)
        if avilable_gpu:
            selected = ",".join(avilable_gpu[:num_gpu])
        else:
            print('No GPU available, will retry after 2.0 seconds ...')
            time.sleep(2)
            continue
        if verbose:
            print('Available GPU List')
            first_line = [['id', 'utilization.gpu(%)', 'memory.free(MiB)']]
            matrix = first_line + avilable_gpu
            s = [[str(e) for e in row] for row in matrix]
            lens = [max(map(len, col)) for col in zip(*s)]
            fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
            table = [fmt.format(*row) for row in s]
            print('\n'.join(table))
            print('Select id #' + selected + ' for you.')
    return selected
