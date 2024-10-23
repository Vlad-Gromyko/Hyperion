import customtkinter as ctk
import tkinter as tk
from core.widgets.mask import MaskView
from core.widgets.parameters import ButtonParameter

import numpy as np
import screeninfo
import cv2


class Translator(ctk.CTkFrame):
    def __init__(self, master, slm, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.slm = slm

        self.label = ctk.CTkLabel(self, text='Трансляция на SLM', width=200, bg_color='#7b64ff')
        self.label.grid(row=0, column=0, sticky='nsew')

        res_x = int(self.slm.res_x.get())
        res_y = int(self.slm.res_y.get())
        gray = int(self.slm.gray_edge.get())

        self.mask = MaskView(master=self, array=np.zeros((res_y, res_x)), slm_gray_edge=gray, small_res_x=240,
                             small_res_y=200)
        self.mask.grid(row=1, column=0, padx=5, pady=5)

        self.check_var = ctk.StringVar(value="off")
        self.checkbox = ctk.CTkCheckBox(self, text="Транслировать на SLM", command=self.checkbox_event,
                                        variable=self.check_var, onvalue="on", offvalue="off")
        self.checkbox.grid(row=2, column=0, padx=5, pady=5)

        self.monitor = ButtonParameter(master=self, name='Монитор', value=1, down_value=0,
                                       up_value=len(screeninfo.get_monitors()) - 1,
                                       second_color='#7b64ff',
                                       first_color="#bab3e5")
        self.monitor.grid(row=3, column=0, padx=5, pady=5)

    def register_addon(self, name, func):
        self.add_cascade(self.mask)
        self.mask.menus['add'].add_command(label=name, command=lambda: func(self.mask.get_array()))

    def add_cascade(self, item):
        add_menu = tk.Menu(self, tearoff=0)
        item.menus['add'] = add_menu
        item.menus['menu'].add_cascade(label='Добавить в', menu=add_menu)

    def on_event(self, event):
        if event=='SLM':
            self.set_array(self.mask.get_array())

    def set_array(self, array):
        self.mask.set_array(array)
        self.mask.set_gray_edge(self.slm.gray_edge.get())
        if self.check_var.get() == 'on':
            window_name = 'slm' + str(int(self.monitor.get()))
            cv2.imshow(window_name, self.mask.get_pixels())

    def checkbox_event(self):
        self.translate()

    def translate(self):
        window_name = 'slm' + str(int(self.monitor.get()))

        if self.check_var.get() == 'on':
            screen_id = int(self.monitor.get())
            screen = screeninfo.get_monitors()[screen_id]
            width, height = screen.width, screen.height

            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            image = self.mask.get_pixels()
            cv2.imshow(window_name, image)
        else:
            cv2.destroyWindow(window_name)
