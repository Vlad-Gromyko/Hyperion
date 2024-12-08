from core.services.service import Service

import customtkinter as ctk


class Progress(Service):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, name='Редактор Ловушек', width=80, height=1, **kwargs)

        self.frame.grid_forget()

        self.label = ctk.CTkLabel(self.top_frame, text='Статус:', anchor='w')
        self.label.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        self.progress = ctk.CTkProgressBar(self.top_frame, fg_color='#000', bg_color='#000', progress_color='#1E90FF', width=1000)
        self.progress.set(0)
        self.progress.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

        self.event_reactions['PROGRESS_UPDATE'] = lambda data: self.on_update(data)

    def on_update(self, data):
        self.update_progress(data['value'], data['max_value'], data['name'])

    def update_progress(self, value, max_value, name):
        self.progress.set(value/max_value)
        self.label.configure(text=f'Статус: {value} из {max_value},  <{name}>')
        self.progress.update_idletasks()
