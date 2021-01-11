from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
import os
import cv2
from PIL import ImageGrab, Image
import numpy as np

#Load the model that is saved
model = load_model('mnist.h5')

def get_handle():
    """This function uses the wingui library to get the window handles of all the active windows.
    Then, the window with the name as 'tk' is selected and its handle is returned."""
    toplist = []
    windows_list = []
    canvas = 0
    def enum_win(hwnd, result):
        win_text = win32gui.GetWindowText(hwnd)
        #print(hwnd, win_text)
        windows_list.append((hwnd, win_text))
    win32gui.EnumWindows(enum_win, toplist)
    for (hwnd, win_text) in windows_list:
        if 'tk' == win_text:
            canvas = hwnd
    return canvas

def preprocessing_image():
    """function to preprocess the image to"""
    image = cv2.imread('test.jpg')
    #print(type(image))
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    preprocessed_digit = []
    index_sort = []
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    x_values = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18,18))
        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digit.append(padded_digit)
        x_values.append(x)
    x_args = np.argsort(x_values)
    prepro = np.array(preprocessed_digit)
    prepro = prepro[x_args]
    i = 0
    for im in prepro:
        i += 1
        cv2.imshow("im no : "+str(i),im)
    return prepro

def predict_digit(img):
    """function to predict the digit.
    Argument of function is PIL Image"""
    img.save('test.jpg')
    preprocessed_image = preprocessing_image()
    digit_predict = []
    accuracy = []
    # print(type(preprocessed_image))
    # print(preprocessed_image.shape)
    for image in preprocessed_image:
        img = image.reshape(1, 28, 28, 1)
        img = img/255.0
        #predicting the digit
        result = model.predict([img])[0]
        digit_predict.append(np.argmax(result))
        accuracy.append(max(result))
    os.remove('test.jpg')
    return digit_predict, accuracy

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        hwnd = get_handle()
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        x1, y1, x2, y2 = rect
        # print(x1,x2, y1,y2)
        im = ImageGrab.grab((x1+40, y1+40, x2+100, y2+100))
        digit, acc = predict_digit(im)
        #Convert digit to str
        digit_str = ""
        pred_accs  = 0
        for git in digit:
            digit_str += str(git)
        for accs in acc:
            pred_accs += accs
        pred_accs = pred_accs/(1.0 * len(acc))
        print("I predict >> "+ digit_str +" !.")
        self.label.configure(text= digit_str+', '+ str(int(pred_accs*100))+'%')
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()