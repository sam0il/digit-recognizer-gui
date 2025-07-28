import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

model = tf.keras.models.load_model("mnist_model.h5")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()
        self.button_predict = tk.Button(self, text="Predict", command=self.predict)
        self.button_predict.pack()
        self.button_clear = tk.Button(self, text="Clear", command=self.clear)
        self.button_clear.pack()

        # Use a larger PIL image to draw on, same size as canvas for better resolution
        self.image = Image.new("L", (280, 280), 255)  # White background
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        r = 8  # brush radius
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)  # draw black on white PIL image

    def preprocess_image(self, img):
        # Resize to 28x28
        img = img.resize((28, 28), Image.LANCZOS)
        # Invert colors: MNIST digits are white on black background
        img = ImageOps.invert(img)
        # Normalize pixels to [0,1]
        img_arr = np.array(img).astype('float32') / 255.0
        # Reshape for model input
        return img_arr.reshape(1, 28, 28, 1)

    def predict(self):
        processed = self.preprocess_image(self.image)
        prediction = model.predict(processed)[0]
        digit = np.argmax(prediction)
        confidence = prediction[digit] * 100
        messagebox.showinfo("Prediction", f"Digit: {digit}\nConfidence: {confidence:.2f}%")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)  # Reset white background
        self.draw = ImageDraw.Draw(self.image)

if __name__ == "__main__":
    app = App()
    app.mainloop()
