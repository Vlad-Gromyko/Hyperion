import customtkinter as ctk
from core.widgets.parameters import ButtonParameter


class SLMPanel(ctk.CTkFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.listeners = []

        self.res_x = ButtonParameter(master=self, name='Разрешение: X', value=1920, down_value=1)
        self.res_x.grid(row=1, column=0)

        self.res_y = ButtonParameter(master=self, name='Разрешение: Y', value=1200, down_value=1)
        self.res_y.grid(row=1, column=1)

        self.pitch = ButtonParameter(master=self, name='Размер Пикселя', value=20, down_value=1,
                                     descriptions={'мкм': 10 ** -6})
        self.pitch.grid(row=2, column=0, columnspan=2)

        self.gray_edge = ButtonParameter(master=self, name='Глубина модуляции (2\u03c0)', value=255, down_value=1,
                                         up_value=255)
        self.gray_edge.grid(row=3, column=0, columnspan=2)

        self.refresh_button = ctk.CTkButton(self, text='\u21BA', command=self.refresh)
        self.refresh_button.grid(row=4, columnspan=2, column=0, sticky='nsew')

    def register_listener(self, listener):
        self.listeners.append(listener)

    def refresh(self):
        for item in self.listeners:
            item.on_event('SLM')


if __name__ == '__main__':
    app = ctk.CTk()
    SLMPanel(app).grid()
    app.mainloop()
