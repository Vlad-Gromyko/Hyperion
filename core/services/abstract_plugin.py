from core.services.service import Service
from core.event_bus import Event, EventBus

import customtkinter as ctk
from core.widgets.frames import ToggledFrame


class AbstractPlugin(Service):
    def __init__(self, master, sub_master, *args, name='Abstract', width=500, height=40, **kwargs):
        Service.__init__(self, *args, name=name, master=master, width=width, height=height, **kwargs)

        self.top_frame.grid_forget()
        self.frame.grid_forget()

        self.sub_master = sub_master

        self.toggle = ToggledFrame(sub_master, text=name)
        self.toggle.grid()
        self.toggle.additional_commands['open'] = self.on_open
        self.toggle.additional_commands['close'] = self.on_close

        ctk.CTkButton(self.toggle.frame, text='\u003F', width=20, fg_color='#000', text_color='#FF4500',
                      hover_color='#DC143C').grid()

        ctk.CTkLabel(self.top_frame, text=name, width=520).grid(row=0, column=0, sticky='nsew')

        self.event_reactions['CLOSE_PLUGINS'] = lambda a: self.on_close()

    def on_open(self):
        self.event_bus.raise_event(Event('CLOSE_PLUGINS', self.name))
        self.grid(row=1, column=0, sticky='nsew')
        self.top_frame.grid(row=0, column=0, sticky='nsew')
        self.frame.grid(row=1, column=0, sticky='nsew')
        self.toggle.frame.grid()

    def on_close(self):
        self.grid_forget()
        self.top_frame.grid_forget()
        self.frame.grid_forget()
        self.toggle.frame.grid_forget()
