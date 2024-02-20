"""Simple profiler for the evolve function."""

import os
import time
import numpy as np
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc_file(f"{os.environ['DM21CM_DIR']}/examples/matplotlibrc")


class Profiler:
    """Simple profiler for the evolve function.
    
    Args:
        start (bool): If True, start the timer.
    """

    def __init__(self, start=False):
        self.t_dict = OrderedDict()
        if start:
            self.start()

    def start(self):
        """Start the timer."""
        self.t = time.perf_counter()

    def record(self, name, restart=True):
        """Record the time since last record. If restart, restart the timer."""
        dt = time.perf_counter() - self.t
        if name in self.t_dict:
            self.t_dict[name].append(dt)
        else:
            self.t_dict[name] = [dt]
        if restart:
            self.start()

    def print_last(self):
        """Print the last recorded time."""
        try:
            for name, t_list in self.t_dict.items():
                print(f'{name}: {t_list[-1]:.4f} s')
        except:
            print('Error printing last.')

    def print_summary(self, ignore_first_n=1):
        """Print the mean and standard deviation of the recorded times."""
        try:
            for name, t_list in self.t_dict.items():
                if len(t_list) == 1:
                    tl = t_list
                    print(f'{name:12}: {np.mean(tl):.4f} s * {len(tl)} steps : {np.sum(tl):.4f} s')
                else:
                    tl = t_list[ignore_first_n:]
                    print(f'{name:12}: {np.mean(tl):.4f} +/- {np.std(tl):.4f} s * {len(tl)} steps : {np.sum(tl):.4f} s')
        except:
            print('Error printing summary.')

    def plot(self, ax=None, **kwargs):
        """Plot the recorded times."""
        if ax is None:
            fig, ax = plt.subplots()

        max_len = max([len(t_list) for t_list in self.t_dict.values()])

        for name, t_list in self.t_dict.items():
            if len(t_list) == 1:
                continue
            subcycle_factor = int(np.round(max_len / len(t_list)))
            ax.plot(np.arange(len(t_list)) * subcycle_factor, np.array(t_list) / subcycle_factor, label=name, **kwargs)

        ax.legend()
        ax.set(xlabel='Subcycle iteration', ylabel='Time per subcycle step [s]')

        if ax is None:
            return fig, ax