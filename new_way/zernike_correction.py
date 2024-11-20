import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import direct

from lab import *

import numba
from functools import lru_cache
from math import factorial
import scipy.optimize

class Zernike(Experiment):
    def run(self, iterations):
        weights = np.random.uniform(-1, 1, 17)

        self.add_trap(1500 * UM, 0)
        self.register_traps()
        self.counter = 0

        def focusing_metric(solution):
            holo = calc_by_spec([0,0,0,*solution], self.slm.width, self.slm.height)
            self.slm.translate(holo)
            shot = self.vision.take_shot()
            return 1 - np.count_nonzero(shot)

        def call(xk):
            print(self.counter, ' :: ', xk)
            self.counter += 1

        bounds = [(-1,1) for i in range(17)]
        res = direct(focusing_metric, bounds, args=(), eps=0.0001, maxfun=None, maxiter=30, callback=call)
        return res.x

    def zernike(self):
        pass


@lru_cache
def binom(a: int, b: int):
    a = int(a)
    b = int(b)
    if a >= b:
        return factorial(a) / factorial(b) / factorial(a - b)
    else:
        return 0


@lru_cache
def calc_nm_list(number):
    out = [[0, 0]]
    n = 0
    m = 0
    for i in range(number - 1):
        m += 2
        if m <= n:
            out.append([n, m])

        else:
            n += 1
            m = -n
            out.append([n, m])
    return out


@lru_cache
def zernike(n, m, res_x, res_y):
    radius_y = 1

    radius_x = radius_y / res_y * res_x

    _x = np.linspace(-radius_x, radius_x, res_x)
    _y = np.linspace(-radius_y, radius_y, res_y)

    _x, _y = np.meshgrid(_x, _y)

    r = np.sqrt(_x ** 2 + _y ** 2)

    phi = np.arctan2(_y, _x)

    array = np.zeros((res_y, res_x))
    for k in range(0, int((n - abs(m)) / 2) + 1):
        array = array + (-1) ** k * binom(n - k, k) * binom(n - 2 * k, (n - abs(m)) / 2 - k) * r ** (
                n - 2 * k)

    if m >= 0:
        array = array * np.cos(m * phi)
    elif m < 0:
        array = array * np.sin(m * phi)

    array = array

    return array


def calc_by_spec(values, res_x, res_y):
    calc = calc_nm_list(len(values))
    array = np.zeros((res_y, res_x))
    for counter, i in enumerate(values):
        if i != 0:
            array = array + zernike(calc[counter][0], calc[counter][1], res_x, res_y) * i

    array_min = np.min(array)

    array = array + array_min
    array = array % (2 * np.pi)
    return array


if __name__ == '__main__':
    plt.ion()
    exp = Zernike()
    rs = exp.run(300)
    print(rs)
    plt.show()
