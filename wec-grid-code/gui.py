from tkinter import *

root = Tk()
root.title("Practice with Grid")
root.geometry("210x180")  # set starting size of window

def display_checked():
    '''check if the checkbuttons have been toggled, and display
    a value of '1' if they are checked, '0' if not checked'''
    red = red_var.get()
    yellow = yellow_var.get()
    green = green_var.get()
    blue = blue_var.get()

    print("red: {}\nyellow:{}\ngreen: {}\nblue: {}".format(
        red, yellow, green, blue))

# Create label
label = Label(root, text="Which colors do you like below?")
label.grid(row=0)

# Create variables and checkbuttons
red_var = IntVar()
Checkbutton(root, width=10, text="red", variable=red_var, bg="red").grid(row=1)

yellow_var = IntVar()
Checkbutton(root, width=10, text="yellow", variable=yellow_var, bg="yellow").grid(row=2)

green_var = IntVar()
Checkbutton(root, width=10, text="green", variable=green_var, bg="green").grid(row=3)

blue_var = IntVar()
Checkbutton(root, width=10, text="blue", variable=blue_var, bg="blue").grid(row=4)

# Create Buttons, one to check which colors are checked,
# and another to close the window.
Button(root, text="Tally", command=display_checked).grid(row=5)
Button(root, text="End", command=root.quit).grid(row=6)

root.mainloop()