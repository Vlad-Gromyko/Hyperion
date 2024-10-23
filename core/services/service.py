from abc import ABC, abstractmethod

from core.event_bus import Event

import customtkinter as ctk


class Service(ABC, ctk.CTkFrame):
    def __init__(self, master, *args, name='Abstract', width=100, height=40, **kwargs):
        ctk.CTkFrame.__init__(self, *args, master=master, width=width, height=height, **kwargs)
        self.name = name
        self.event_bus = None
        self.event_reactions = {}
        self.request_reactions = {}
        self.result_reactions = {}

        self.top_frame = ctk.CTkFrame(master, fg_color='#000')
        self.top_frame.grid(row=0)

        self.frame = ctk.CTkFrame(master)
        self.frame.grid(row=1)

    def raise_event(self, event: Event):
        if event.name in self.event_reactions.keys():
            operation = self.event_reactions[event.name]
            operation(event.data)

    def raise_request(self, event: Event):
        if event.name in self.request_reactions.keys():
            return self.request_reactions[event.name]

    def raise_result(self, event: Event):
        if event.name in self.result_reactions.keys():
            return self.result_reactions[event.name](event.data)

    def start(self):
        pass
