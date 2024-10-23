from abc import ABC

from core.services.service import Service

import customtkinter as ctk


class Device(Service):
    def __init__(self, master, *args, name='Device', **kwargs):
        super().__init__(master, *args, name=name, **kwargs)

        ctk.CTkLabel(self.top_frame, text='Порт:').grid(row=0, column=0, padx=5, pady=5)
        self.port = ctk.CTkEntry(self.top_frame, width=50, bg_color='#7FFF00')
        self.port.insert(0, '0')
        self.port.grid(row=0, column=1, padx=5, pady=5)
