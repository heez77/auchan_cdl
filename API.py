from tkinter import *
from PIL import Image, ImageTk
import os
import pandas as pd

df = pd.read_csv('gitignore/df.csv')
df_label = pd.read_csv("gitignore/df_label.csv")

def validate(variable):
    validation = variable.get()
    

root = Tk()

for i in range(len(df)):
    newWindow = Toplevel(root)
    newWindow.title('Labellisation')
    newWindow.geometry('1920x1010')
    newWindow.state("zoomed")

    frame_prediction = Frame(newWindow)
    frame_prediction.pack(side=LEFT)

    img = Image.open(os.listdir("gitignore/données/photos/image_{}.jpg".format(i)))
    resized_img = img.resize((800, 800), Image.ANTIALIAS)
    tk_img = ImageTk.PhotoImage(resized_img)
    label_image = Label(frame_prediction, image=tk_img)
    label_image.image = tk_img
    label_image.pack(expand=YES)

    text_count = Label(frame_prediction, text='Image n°{}'.format(i))
    text_count.pack(expand=YES)

    OptionList = df_label.niv1.tolist()

    variable = StringVar(newWindow)
    variable.set(0)
    opt = OptionMenu(newWindow, variable, *OptionList)
    opt.pack(expand=YES)

    valid_button = Button(newWindow, text='Valid', command=lambda:[validate(variable), newWindow.destroy()])
    valid_button.pack(expand=YES)




root.mainloop()