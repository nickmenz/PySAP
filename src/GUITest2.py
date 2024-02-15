import tkinter as tk

window = tk.Tk()

frame1 = tk.Frame(master=window, height=100, width= 50, bg="red")
frame1.pack(side=tk.TOP)

frame2 = tk.Frame(master=window, height=50,  width= 50, bg="yellow")
frame2.pack(side=tk.LEFT)

frame3 = tk.Frame(master=window, height=25, width= 50, bg="blue")
frame3.pack(side=tk.RIGHT)

window.mainloop()