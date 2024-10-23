

from core.services.service import Service

import customtkinter as ctk



class Lab(Service):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, name='Редактор Ловушек', width=80, height=1, **kwargs)

        self.frame.grid_forget()

        ctk.CTkLabel(self.top_frame, text='Длина Волны (нм):').grid(row=0, column=0, padx=5, pady=5)
        self.wave = ctk.CTkEntry(self.top_frame, width=50, bg_color='#7FFF00')
        self.wave.insert(0, '850')
        self.wave.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(self.top_frame, text='Фокус (мм):').grid(row=1, column=0, padx=5, pady=5)
        self.focus = ctk.CTkEntry(self.top_frame, width=50, bg_color='#7FFF00')
        self.focus.insert(0, '100')
        self.focus.grid(row=1, column=1, padx=5, pady=5)

        self.request_reactions['WAVE'] = lambda: float(self.wave.get()) * 10 ** -9
        self.request_reactions['FOCUS'] = lambda: float(self.focus.get()) * 10 ** -3

