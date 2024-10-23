import customtkinter as ctk
import tkinter as tk
import numpy as np

from core.sub_systems.slm import SLMPanel
from core.widgets.mask import MaskView
from typing import List


class Atlas(ctk.CTkFrame):
    def __init__(self, master, slm: SLMPanel, arrays: List[np.ndarray] = None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.slm = slm
        self.masks = []
        self.buttons = []

        self.clear_button = ctk.CTkButton(self, text='Очистить\nАтлас', width=80, command=self.clear)
        self.clear_button.grid(row=0, column=0, padx=5)

        self.scroll = ctk.CTkScrollableFrame(self, orientation='horizontal', height=150, width=1000)
        self.scroll.grid(row=0, column=1, )

        if arrays is not None:
            for item in arrays:
                self.add_mask(item)

        self.cascaded = False
        self.addons = []

    def register_addon(self, name, func):
        self.addons.append((name, func))
        if not self.cascaded:
            self.add_cascades()
            self.cascaded = True
        for item in self.masks:
            item.menus['add'].add_command(label=name, command=lambda: func(item.get_array()))

    def add_command(self, name, func, item):
        item.menus['add'].add_command(label=name, command=lambda: func(item.get_array()))

    def add_cascades(self):
        for item in self.masks:
            self.add_cascade(item)

    def add_cascade(self, item):
        add_menu = tk.Menu(self, tearoff=0)
        item.menus['add'] = add_menu
        item.menus['menu'].add_cascade(label='Добавить в', menu=add_menu)

    def add_mask(self, array):
        mask = MaskView(array, master=self.scroll, slm_gray_edge=int(self.slm.gray_edge.get()))
        mask.grid(row=0, column=len(self.masks))
        self.add_cascade(mask)
        for spec in self.addons:
            self.add_command(spec[0], spec[1], mask)
        value = self.create_lambda(len(self.masks))
        button = ctk.CTkButton(self.scroll, text='\u2716', command=lambda: self.delete_mask(value()), width=50)
        button.grid(row=1, column=len(self.masks))
        self.masks.append(mask)
        self.buttons.append(button)

    def delete_mask(self, number):
        print(number)
        b = self.buttons.pop(number)
        b.grid_forget()
        m = self.masks.pop(number)
        m.grid_forget()
        masks = self.masks
        self.clear()

        for item in masks:
            self.add_mask(item.get_array())

    def clear(self):
        for i in range(len(self.masks)):
            self.buttons[i].grid_forget()
            self.masks[i].grid_forget()
        self.masks = []
        self.buttons = []

    @staticmethod
    def create_lambda(x=None):
        return lambda: x

    def on_event(self, event):
        if event == 'SLM':
            gray = int(self.slm.gray_edge.get())
            for mask in self.masks:
                mask.set_gray_edge(gray)
