# from tkinter import *
#
# from tkinter import ttk
#
# window = Tk()
#
# window.title("WEC-Grid")
#
# tab_control = ttk.Notebook(window)
#
# tab1 = ttk.Frame(tab_control)
#
# tab2 = ttk.Frame(tab_control)
#
# tab3 = ttk.Frame(tab_control)
#
# tab_control.add(tab1, text='Main')
#
# tab_control.add(tab2, text='WEC-Sim')
#
# tab_control.add(tab3, text='PSSe')
#
# lbl1 = Label(tab1, text= "here I'll display the main run information")
#
# lbl1.grid(column=0, row=0)
#
# lbl2 = Label(tab2, text= "Here I'll display Wec-Sim settings, paths and Database snapshot")
#
# lbl2.grid(column=0, row=0)
#
# lbl3 = Label(tab3, text= "here I'll display PSSe stuff, path config, .raw file")
#
# lbl3.grid(column=0, row=0)
#
# tab_control.pack(expand=1, fill='both')
#
# var1 = IntVar()
# Checkbutton(tab1, text='on', variable=var1).grid(row=0, sticky=W)
# var2 = IntVar()
# Checkbutton(tab1, text='off', variable=var2).grid(row=1, sticky=W)
#
# Label(tab1, text='path 1').grid(row=3)
# Label(tab1, text='path 2').grid(row=4)
# e1 = Entry(tab1)
# e2 = Entry(tab1)
# e1.grid(row=3, column=1)
# e2.grid(row=4, column=1)
#
#
#
#
# window.mainloop()

from tkinter import *
from tkinter import ttk


class FeetToMeters:

    def __init__(self, root):

        root.title("Feet to Meters")

        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.feet = StringVar()
        feet_entry = ttk.Entry(mainframe, width=7, textvariable=self.feet)
        feet_entry.grid(column=2, row=1, sticky=(W, E))
        self.meters = StringVar()

        ttk.Label(mainframe, textvariable=self.meters).grid(column=2, row=2, sticky=(W, E))
        ttk.Button(mainframe, text="Calculate", command=self.calculate).grid(column=3, row=3, sticky=W)

        ttk.Label(mainframe, text="feet").grid(column=3, row=1, sticky=W)
        ttk.Label(mainframe, text="is equivalent to").grid(column=1, row=2, sticky=E)
        ttk.Label(mainframe, text="meters").grid(column=3, row=2, sticky=W)

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

        feet_entry.focus()
        root.bind("<Return>", self.calculate)

    def calculate(self, *args):
        try:
            value = float(self.feet.get())
            self.meters.set(int(0.3048 * value * 10000.0 + 0.5) / 10000.0)
        except ValueError:
            pass


root = Tk()
FeetToMeters(root)
root.mainloop()