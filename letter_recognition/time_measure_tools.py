import time


def measure_time(func):
    def decorated_func(*args, **kwargs):
        st = time.time()
        result = func(*args, **kwargs)
        et = time.time()

        print '%s - %2.2f sec' % (func.__name__, et - st)
        return result

    return decorated_func
