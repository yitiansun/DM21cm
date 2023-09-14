"""Simple profiler for the evolve function."""

import time
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt


class Profiler:

    def __init__(self, start=False):

        self.t_dict = OrderedDict()
        if start:
            self.start()

    def start(self):
        self.t = time.perf_counter()

    def record(self, name, restart=True):

        dt = time.perf_counter() - self.t

        if name in self.t_dict:
            self.t_dict[name].append(dt)
        else:
            self.t_dict[name] = [dt]
        
        if restart:
            self.start()

    def print_last(self):
        for name, t_list in self.t_dict.items():
            print(f'{name}: {t_list[-1]:.4f} s')

    def print_summary(self, ignore_first_n=1):
        for name, t_list in self.t_dict.items():
            print(f'{name}: {np.mean(t_list[ignore_first_n:]):.4f} +/- {np.std(t_list[ignore_first_n:]):.4f} s')

    def plot(self, ax=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots()

        for name, t_list in self.t_dict.items():
            ax.plot(t_list, label=name, **kwargs)

        ax.legend()
        ax.set(xlabel='Iteration', ylabel='Time [s]')
        return ax