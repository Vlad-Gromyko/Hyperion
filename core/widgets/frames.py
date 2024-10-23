import customtkinter as ctk


class ToggledFrame(ctk.CTkFrame):

    def __init__(self, master, text="", height=100, *args, **kwargs):
        ctk.CTkFrame.__init__(self, master, *args, **kwargs)

        self.button = ctk.CTkButton(self, text=text, width=185,
                                    command=self.toggle, fg_color='#000', text_color='#FF4500',
                                    hover_color='#DC143C').grid(row=0,
                                                                column=0, sticky='nsew')

        self.frame = ctk.CTkFrame(self, width=200, height=height, fg_color='#000', bg_color='#000')

        self.opened = False

        self.additional_commands = {}

    def toggle(self):
        if self.opened:
            self.additional_commands['close']()
        else:
            self.additional_commands['open']()
