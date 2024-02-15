import tkinter as tk

class NumberButton(tk.Button):
    def __init__(self, number, **kwargs):
        super().__init__(**kwargs)
        self.number = number

class Calculator():
    def __init__(self, title):
        self.window_title = title
        self.start_number="0"
        self.window = tk.Tk()
        self.window.title(self.window_title)
        self.window.resizable(True, True)
    
        # Number pad
        self.frm_number_pad = tk.Frame(master=self.window, height=200, width=245, background="red", relief=tk.RIDGE, borderwidth=5)
        self.frm_number_pad.grid(row=1, column=0, pady=5, sticky="nsew")

        # Number display
        self.frm_number_display = tk.Frame(master=self.window,  height=40, width=300, background="blue", relief=tk.RAISED, borderwidth=3)
        self.frm_number_display.grid(row=0, column=0, columnspan=2, pady=5, sticky="nsew")
        self.txt_number_display = tk.Label(master=self.frm_number_display, text=self.start_number)
        self.txt_number_display.grid()

        # Numerical operators 
        self.frm_operators = tk.Frame(master=self.window, height=200, width=50, background="yellow", relief=tk.RIDGE, borderwidth=5)
        self.frm_operators.grid(column=1, row=1, padx=5, sticky="nsew")
        self.btn_add = tk.Button(master=self.frm_operators, text="+", height=3, width=5)
        self.btn_subtract = tk.Button(master=self.frm_operators, text="-", height=3, width=5)
        self.btn_multiply = tk.Button(master=self.frm_operators, text="*", height=3, width=5)
        self.btn_divide = tk.Button(master=self.frm_operators, text="/", height=3, width=5)
        self.btn_equals = tk.Button(master=self.frm_operators, text="=", height=3, width=5)
        self.btn_add.grid(row=0, pady=7)
        self.btn_subtract.grid(row=1, pady=7)
        self.btn_multiply.grid(row=2, pady=7)
        self.btn_divide.grid(row=3, pady=7)
        self.btn_equals.grid(row=4, pady=7)

        # window resizing configuration
        for i in range(2):
            self.window.rowconfigure(i, weight=1, minsize=50)
            self.window.columnconfigure(i, weight=1, minsize=50)
   
        for i in range(0, 10):
            if i == 0: 
                button = NumberButton(i, master=self.frm_number_pad, text="0",height=3, width=5).grid(column=1, row=4, padx=5, pady=5)
            else:
                button = NumberButton(i, master=self.frm_number_pad, text=f"{i}", height=3, width=5, command=self.display_number)\
                .grid(column=((i - 1) % 3), row=int((i - 1) / 3), padx=5, pady=5)


    def open_window(self):
        self.window.mainloop()

    def display_number(self):
        self.txt_number_display["text"] = "1"


def main():
    calc = Calculator("Nick's calculator")
    calc.open_window()

if __name__=="__main__": 
    main()

