from core.services.device import Device

import customtkinter as ctk
import tkdial

from core.widgets.mask import MaskView
from core.event_bus import Event

import numpy as np
import cv2
import screeninfo


class SLM(Device):
    def __init__(self, master, width, height, gray):
        super().__init__(master, name='Камера')

        self.port.delete(0)
        self.port.insert(0, '1')

        self.check_var = ctk.StringVar(value="off")
        self.checkbox = ctk.CTkSwitch(self.frame, text="Трансляция на SLM", progress_color='#DC143C',
                                      text_color='#DC143C',
                                      variable=self.check_var, onvalue="on", offvalue="off", command=self.translate)
        self.checkbox.grid(row=0, column=0, padx=5, pady=5)

        self.mask = MaskView(self.frame, np.zeros((1200, 1920)), 255, 160, 100)
        self.mask.grid(row=1, column=0, padx=5, pady=5)
        self.mask.post_commands.append(self.translate)

        frame = ctk.CTkFrame(self.frame)
        frame.grid(row=2, column=0, padx=5, pady=5)
        self.wheel = tkdial.Dial(frame, color_gradient=("yellow", "red"),
                                 text_color="white", text="2\u03C0: ", unit_length=10, radius=50, integer=True, start=1,
                                 end=255, command=self.wheel_change)
        self.wheel.set(gray)
        self.wheel.grid(row=0, column=0, padx=5, pady=5, rowspan=2)

        ctk.CTkButton(frame, text='\u25B2', width=10, fg_color='#DC143C', command=self.up).grid(row=0, column=1, padx=5,
                                                                                                pady=5)
        ctk.CTkButton(frame, text='\u25BC', width=10, fg_color='#FFD700', command=self.down).grid(row=1, column=1,
                                                                                                  padx=5, pady=5)

        frame_pitch = ctk.CTkFrame(frame)
        frame_pitch.grid(row=2, column=0, padx=5, columnspan=2)
        ctk.CTkLabel(frame_pitch, text='Ширина\nПикселя (мкм):', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)

        self.pitch = ctk.CTkEntry(frame_pitch, width=45, bg_color='#DC143C')
        self.pitch.insert(0, '8')
        self.pitch.grid(row=0, column=1)

        self.request_reactions['SLM_PITCH'] = lambda: float(self.pitch.get()) * 10 ** -6
        self.request_reactions['SLM_WIDTH'] = lambda: width
        self.request_reactions['SLM_HEIGHT'] = lambda: height
        self.request_reactions['SLM_GRAY'] = lambda: self.wheel.get()
        self.request_reactions['MASK_ON_SLM'] = lambda: self.mask.get_array()

        self.event_reactions['TO_SLM'] = lambda array : self.mask.set_array(array)
        self.event_reactions['TURN_ON_SLM'] = lambda data: self.turn_on()

    def turn_on(self):
        if self.check_var.get()=='off':
            self.checkbox.toggle()


    def up(self):
        gray = self.wheel.get()
        if gray + 1 <= 255:
            self.wheel.set(gray + 1)

    def down(self):
        gray = self.wheel.get()
        if gray - 1 >= 1:
            self.wheel.set(gray - 1)

    def wheel_change(self):
        if self.event_bus is not None:
            self.event_bus.raise_event(Event(name='SLM_GRAY_CHANGED', data=self.wheel.get()))
        self.mask.set_gray_edge(self.wheel.get())

    def set_array(self, array):
        self.mask.set_array(array)
        if cv2.getWindowProperty('SLM', cv2.WND_PROP_VISIBLE):
            cv2.imshow('SLM', self.mask.get_pixels())

    def translate(self):
        if self.check_var.get() == 'off':
            if cv2.getWindowProperty('SLM', cv2.WND_PROP_VISIBLE):
                cv2.destroyWindow('SLM')
        elif self.check_var.get() == 'on':
            screen_id = int(self.port.get())
            screen = screeninfo.get_monitors()[screen_id]
            width, height = screen.width, screen.height

            cv2.namedWindow('SLM', cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow('SLM', screen.x - 1, screen.y - 1)
            cv2.setWindowProperty('SLM', cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            image = self.mask.get_pixels()
            cv2.imshow('SLM', image)
