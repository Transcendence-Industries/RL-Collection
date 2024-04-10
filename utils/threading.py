import os
import time
from multiprocessing.pool import Pool


class MultiCore_Controller:
    def __init__(self, n_cores):
        if n_cores > os.cpu_count():
            raise Exception(f"It's impossible to utilize more instances than available CPU cores!")

        self.pool = Pool(processes=n_cores)

    def run(self, function, param_list):
        results = []

        for params in param_list:
            results.append(self.pool.apply_async(func=function, args=params))
            print(f"Started a new instance for function '{function.__name__}'.")
            time.sleep(5)

        final_results = [result.get() for result in results]
        print(final_results)
