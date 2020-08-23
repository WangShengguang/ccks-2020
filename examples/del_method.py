import time


class DelMethod(object):

    def __del__(self):
        seconds = 10
        time.sleep(seconds)
        print(f'sleep seconds : {seconds}')


def main():
    DelMethod()


if __name__ == '__main__':
    main()
