from functools import lru_cache

from core.services.abstract_plugin import AbstractPlugin
from core.event_bus import Event

import customtkinter as ctk
from core.widgets.mask import MaskView

import numpy as np
from math import factorial


class Plugin(AbstractPlugin):
    def __init__(self, master, sub_master, *args, **kwargs):
        super().__init__(master, sub_master, *args, name='Цернике', width=500, height=40, **kwargs)

        self.mask = None

        self.scroll = ctk.CTkScrollableFrame(self.frame, orientation='vertical', height=600)
        self.scroll.grid(row=1, column=1, rowspan=2, padx=5, pady=5)
        self.cages = []
        self.add_cages()

        self.button_make = ctk.CTkButton(self.frame, text='Рассчет', command=self.start_calc, fg_color='#000',
                                         text_color='#1E90FF', )
        self.button_make.grid(row=2, column=0, padx=5, pady=5)

        self.request_reactions['TAKE_ZERNIKE'] = lambda: self.mask.get_array()

    def start_calc(self):
        values = [self.cages[i].get() for i in range(len(self.cages))]
        array = np.zeros_like(self.mask.get_array())

        res_x = self.event_bus.raise_request(Event('SLM_WIDTH'))
        res_y = self.event_bus.raise_request(Event('SLM_HEIGHT'))

        array = calc_by_spec(values, res_x, res_y)

        self.mask.set_array(array)

    def add_cages(self):
        items = ['Плоский',
                 'Наклон Y',
                 'Наклон X',
                 'Астигматизм 0/90',
                 'Сфера',
                 'Астигматизм -45/+45',
                 'Трилистник 0/90',
                 'Кома Y',
                 'Кома X',
                 'Трилистник -45/+45']

        calc = calc_nm_list(len(items))
        for counter, item in enumerate(items):
            self.add_cage(item, calc[counter][0], calc[counter][1], counter)

    def add_cage(self, name, n, m, counter):
        cage = Cage(self.scroll, name, n, m, counter)
        cage.grid(row=counter, column=0, padx=5, pady=5)
        self.cages.append(cage)

    def start(self):
        res_x = self.event_bus.raise_request(Event('SLM_WIDTH'))
        res_y = self.event_bus.raise_request(Event('SLM_HEIGHT'))
        gray = self.event_bus.raise_request(Event('SLM_GRAY'))

        self.mask = MaskView(master=self.frame, array=np.zeros((res_y, res_x)), slm_gray_edge=gray, small_res_x=320,
                             small_res_y=200)

        self.mask.grid(row=1, column=0, padx=5, pady=5)

        self.mask.add_menu_command('Отправить на SLM',
                                   lambda: self.event_bus.raise_event(Event('TO_SLM', self.mask.get_array())))

        self.mask.add_menu_command('Отправить в Атлас',
                                   lambda: self.event_bus.raise_event(Event('TO_ATLAS', self.mask.get_array())))
        self.mask.add_menu_command('Отправить в Аккумулятор',
                                   lambda: self.event_bus.raise_event(Event('TO_ACCUMULATOR', self.mask.get_array())))

        self.result_reactions['COMPUTE_ZERNIKE_BY_SPEC'] = lambda spec: calc_by_spec([0, 0, 0, *spec], res_x, res_y)

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


class Cage(ctk.CTkFrame):
    def __init__(self, master, name, n, m, counter, *args, **kwargs):
        super().__init__(*args, master=master, **kwargs)

        self.label = ctk.CTkLabel(self, text=name, bg_color='#000000')
        self.label.grid(row=0, column=0, columnspan=4, sticky='nsew')

        ctk.CTkLabel(self, text=f'{counter + 1}', width=40).grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkLabel(self, text=f'n={n}', width=40).grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkLabel(self, text=f'm={m}', width=40).grid(row=1, column=2, padx=5, pady=5)

        self.entry = ctk.CTkEntry(master=self, width=40)
        self.entry.grid(row=1, column=3, padx=5, pady=5)
        self.entry.insert(0, '0')

        if m > 0:
            self.label.configure(bg_color='#004dcf')
        elif m < 0:
            self.label.configure(bg_color='#b80000')

    def get(self):
        return float(self.entry.get())
