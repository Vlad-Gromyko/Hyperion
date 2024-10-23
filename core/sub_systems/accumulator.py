import customtkinter as ctk
from core.sub_systems.slm import SLMPanel
from core.widgets.mask import MaskView
from core.sub_systems.atlas import Atlas
import numpy as np


class Accumulator(Atlas):
    def __init__(self, master, slm: SLMPanel, *args, **kwargs):
        super().__init__(master, slm, *args, **kwargs)

        res_x = int(self.slm.res_x.get())
        res_y = int(self.slm.res_y.get())
        gray = int(self.slm.gray_edge.get())

        self.mask = MaskView(master=self, array=np.zeros((res_y, res_x)), slm_gray_edge=gray)
        self.mask.grid(row=1, column=0, padx=5, pady=5)

        self.scroll.grid(row=0, column=1, padx=5, pady=5, rowspan=2)

        self.clear_button.configure(text='Очистить\nАккумулятор')

        self.second_cascaded = False

    def register_addon(self, name, func):
        super().register_addon(name, func)
        if not self.second_cascaded:
            self.add_cascade(self.mask)
            self.second_cascaded = True
        self.mask.menus['add'].add_command(label=name, command=lambda: func(self.mask.get_array()))


    def on_event(self, event):
        super().on_event(event)
        gray = (self.slm.gray_edge.get())
        self.mask.set_gray_edge(gray)

    def add_mask(self, array):
        super().add_mask(array)
        array = np.zeros_like(self.masks[0].get_array())

        for item in self.masks:
            array = array + item.get_array()

        array = array % (2 * np.pi)

        self.mask.set_array(array)

    def clear(self):
        super().clear()
        res_x = int(self.slm.res_x.get())
        res_y = int(self.slm.res_y.get())
        self.mask.set_array(np.zeros((res_y, res_x)))
