import time

import matplotlib.pyplot as plt
import numpy as np

from lab import *

import numba


class ZernikeWeight(Experiment):
    def run(self, iterations):
        np.random.seed(42)