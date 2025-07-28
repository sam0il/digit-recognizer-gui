# Digit Recognizer GUI

This is a simple handwritten digit recognizer using a convolutional neural network trained on the MNIST dataset. It includes a Tkinter-based GUI where you can draw digits and get predictions from the trained model.

> **Note:**  
> This project is intended as a **test/demo** to explore basic AI model integration with a GUI. The accuracy is decent but not production-ready.

## Features

- Draw digits with your mouse on a canvas.
- Predicts the drawn digit using a trained CNN model.
- Shows the predicted digit and confidence score.
- Clear button to reset the drawing.

## How to Use

1. Train the model by running the training script (`train_model.py`) to generate `mnist_model.h5` or use the provided model file.
2. Run the GUI script (`digit_gui.py`).
3. Draw a digit on the canvas and click **Predict**.
4. See the predicted digit and confidence.
5. Click **Clear** to draw a new digit.

## Requirements

- Python 3.x
- TensorFlow
- Pillow
- numpy

Install requirements with:

```bash
pip install tensorflow pillow numpy
