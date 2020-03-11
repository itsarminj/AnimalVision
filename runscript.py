import numpy as np
import cv2
import tkinter as tk
# import Image, ImageTk
from PIL import Image, ImageTk, ImageEnhance
from vision_model import Vision

#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("Animal Vision")
window.config(background="#B8B8B8")

#Graphics window
imageFrame = tk.Frame(window, width=600, height=500, background="#B8B8B8")
imageFrame.grid(row=0, column=0, padx=10, pady=2)

# Buttons window
controlFrame = tk.Frame(window, width=600, height=500, background="#B8B8B8")
controlFrame.grid(row=1, column=0, padx=10, pady=2)

display1 = tk.Label(imageFrame)
display1.grid(row=0, column=0, padx=10, pady=2)  #Display 1
display2 = tk.Label(imageFrame)
display2.grid(row=0, column=1) #Display 2

cap = cv2.VideoCapture(0)
model = Vision(display1)

args = (model.Human, model.Fly, model.Dog, model.Snake, model.Horse, model.Fish, model.Slug, model.Rat)

global counter
counter = 0

def show_frame():

    # Read frames
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # FOV + BLUR
    width = int(frame.shape[1] * model.resize_percent / 100)
    height = int(frame.shape[0] * model.resize_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # -- GUI --

    # Image
    frame_post = np.array(args[counter](frame=resized)).astype(np.uint8)

    # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(frame_post)
    imgtk = ImageTk.PhotoImage(image=img)
    display1.imgtk = imgtk
    display1.configure(image=imgtk)

    # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(np.array(args[0](frame=resized)))
    imgtk = ImageTk.PhotoImage(image=img)
    display2.imgtk = imgtk
    display2.configure(image=imgtk)

    label_text = str(args[counter].__name__)
    label_text = '{message:{fill}{align}{width}}'.format(
        message=label_text,
        fill=' ',
        align='^',
        width=16,
    )
    name = tk.Label(imageFrame, text=label_text)
    name.grid(row=1, column=0)
    name_og = tk.Label(imageFrame, text='{message:{fill}{align}{width}}'.format(
        message="Human",
        fill=' ',
        align='^',
        width=16,
    ))
    name_og.grid(row=1, column=1)

    window.after(10, show_frame)

# button function helper
def nclick_plus():
    global counter
    if counter < len(args)-1:
        counter += 1
    else:
        counter = 0

def nclick_minus():
    global counter
    if counter > 0 :
        counter -= 1
    else:
        counter = len(args)-1

def nclick_Dog():
    global counter
    for k, i in enumerate(args):
        if i.__name__ == 'Dog':
            counter = k

def nclick_Snake():
    global counter
    for k, i in enumerate(args):
        if i.__name__ == 'Snake':
            counter = k

def nclick_Fish():
    global counter
    for k, i in enumerate(args):
        if i.__name__ == 'Fish':
            counter = k

def nclick_Human():
    global counter
    for k, i in enumerate(args):
        if i.__name__ == 'Human':
            counter = k

def nclick_Horse():
    global counter
    for k, i in enumerate(args):
        if i.__name__ == 'Horse':
            counter = k

def nclick_Fly():
    global counter
    for k, i in enumerate(args):
        if i.__name__ == 'Fly':
            counter = k

def nclick_Slug():
    global counter
    for k, i in enumerate(args):
        if i.__name__ == 'Slug':
            counter = k

def nclick_Rat():
    global counter
    for k, i in enumerate(args):
        if i.__name__ == 'Rat':
            counter = k

next_button = tk.Button(controlFrame, text='{message:{fill}{align}{width}}'.format(
        message="next",
        fill=' ',
        align='^',
        width=20), command=nclick_plus)
next_button.grid(row=0, column=3)

previous_button = tk.Button(controlFrame, text='{message:{fill}{align}{width}}'.format(
        message="Previous",
        fill=' ',
        align='^',
        width=20), command=nclick_minus)
previous_button.grid(row=0, column=2)

# Choosing animal buttons
arg_list = []
for i in range(len(args)):
    arg_list.append(tk.Button(controlFrame, text='{message:{fill}{align}{width}}'.format(
        message=args[i].__name__,
        fill=' ',
        align='^',
        width=20), command=locals()["nclick_" + args[i].__name__]))
    arg_list[i].grid(row=2, column=i)

show_frame()  #Display 2
window.mainloop()  #Starts GUI


