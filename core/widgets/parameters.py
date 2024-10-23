import customtkinter as ctk
from typing import Union


class Parameter(ctk.CTkFrame):
    def __init__(self, name: str = 'PARAMETER', value: Union[int, float] = 0,
                 descriptions: dict = None,
                 down_value: Union[int, float] = None, up_value: Union[int, float] = None,
                 first_color='#3CB371',
                 second_color='#228B22',
                 *args, **kwargs):
        super().__init__(*args, fg_color=first_color, **kwargs)

        self.first_color = first_color
        self.second_color = second_color

        self.descriptions = descriptions
        self.up_value = up_value
        self.down_value = down_value

        self.label = ctk.CTkLabel(self, text=name, fg_color=second_color)
        self.label.grid(row=0, column=0, columnspan=1, sticky='nsew')

        self.entry = ctk.CTkEntry(self, width=70)
        self.entry.grid(row=1, column=0, padx=5, pady=5, columnspan=1)
        self.entry.insert(0, value)

        if descriptions:
            self.combobox_var = ctk.StringVar(value=list(descriptions.keys())[0])
            self.combobox = ctk.CTkComboBox(self, values=list(descriptions.keys()), variable=self.combobox_var,
                                            width=100)
            self.combobox_var.set(list(descriptions.keys())[0])
            self.combobox.grid(row=1, column=1, padx=5, pady=5)

        self.entry.bind('<Leave>', self.leave)

        max_min_frame = ctk.CTkFrame(self)

        if down_value is not None:
            ctk.CTkLabel(max_min_frame, text=f'Min: {down_value}').grid(row=0, column=0, padx=5)
        if up_value is not None:
            ctk.CTkLabel(max_min_frame, text=f'Max: {up_value}').grid(row=0, column=1, padx=5)
        if up_value is not None or down_value is not None:
            max_min_frame.grid(row=2, columnspan=4, sticky='nsew')

    def update_entry(self, new_value):
        self.entry.delete(0, 'end')
        self.entry.insert(0, str(new_value))

    def leave(self, event):
        try:
            new_value = self.validate(float(self.entry.get()))
            self.update_entry(new_value)
        except ValueError:
            self.label.configure(fg_color='#FF0000')

    def validate(self, new_value: Union[int, float]):
        if self.up_value is not None:
            if new_value > self.up_value:
                new_value = self.up_value
        if self.down_value is not None:
            if new_value < self.down_value:
                new_value = self.down_value
        self.label.configure(fg_color=self.second_color)
        return new_value

    def get(self):
        if self.descriptions is not None:
            coefficient = self.descriptions[self.combobox.get()]
        else:
            coefficient = 1
        return coefficient * float(self.entry.get())


class ButtonParameter(Parameter):
    def __init__(self, name: str = 'PARAMETER', value: Union[int, float] = 0,
                 descriptions: dict = None,
                 down_value: Union[int, float] = None, up_value: Union[int, float] = None,
                 first_color='#3CB371',
                 second_color='#228B22',
                 *args, **kwargs):
        super().__init__(name, value,
                         descriptions,
                         down_value, up_value,
                         first_color,
                         second_color,
                         *args, **kwargs)

        buttonbox = ctk.CTkFrame(self)
        buttonbox.grid(row=1, column=3, padx=2, pady=2)

        self.button_up = ctk.CTkButton(buttonbox, text='\u2191', width=2, command=self.up, fg_color=second_color
                                       , bg_color=first_color)
        self.button_up.grid(row=0, column=0)

        self.button_down = ctk.CTkButton(buttonbox, text='\u2193', width=2, command=self.down, fg_color=second_color
                                         , bg_color=first_color)
        self.button_down.grid(row=1, column=0)

        self.entry.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.label.grid(row=0, column=0, columnspan=2, sticky='nsew')

        self.entry.bind('<Leave>', self.leave)

    def up(self):
        new_value = float(self.entry.get()) + 1
        new_value = self.validate(new_value)
        self.update_entry(new_value)

    def down(self):
        new_value = float(self.entry.get()) - 1
        new_value = self.validate(new_value)
        self.update_entry(new_value)


class ConstParameter(Parameter):
    def __init__(self, name: str = 'CONST_PARAMETER', value: Union[int, float] = 0,
                 description: str = None, coefficient: Union[int, float] = 1, *args, **kwargs):
        super().__init__(*args, name=name, descriptions=None, **kwargs)
        self.entry.grid_forget()
        self.entry = ctk.CTkLabel(self, text=str(value), fg_color='#343638')
        self.entry.grid(row=1, column=0, sticky='nsew')
        self.value = value
        self.coefficient = coefficient

        self.label.configure(fg_color="#4169E1")
        self.configure(fg_color='#6495ED')
        self.configure(bg_color='#000000')

        if description:
            ctk.CTkLabel(self, text=description).grid(row=1, column=1, sticky='nsew')

    def get(self):
        return self.value * self.coefficient


if __name__ == '__main__':
    app = ctk.CTk()
    app.geometry("400x400")

    widget = ConstParameter(master=app, description='dldldlhi')
    widget.grid(padx=10, pady=10, row=1)

    app.mainloop()
