import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import messagebox
import os

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Function to preprocess the drawn image
def preprocess_image(image_path):
    # Open the image and convert to grayscale
    img = Image.open(image_path).convert('L')
    # Resize to 28x28 (MNIST size)
    img = img.resize((28, 28))
    # Convert to numpy array and normalize
    img_array = np.array(img).astype('float32') / 255.0
    # Invert the image (MNIST has white digits on black background, but drawing is black on white)
    img_array = 1 - img_array
    # Reshape for the model
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# Function to predict the digit
def predict_digit(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    return np.argmax(prediction, axis=1)[0]

# Tkinter drawing app
class DigitDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        # Canvas for drawing
        self.canvas = tk.Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.pack(pady=10)
        
        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.start_draw)
        
        # Buttons
        self.predict_btn = tk.Button(self.root, text="Predict Digit", command=self.predict)
        self.predict_btn.pack(pady=5)
        
        self.clear_btn = tk.Button(self.root, text="Clear Canvas", command=self.clear)
        self.clear_btn.pack(pady=5)
        
        # Label to display prediction
        self.result_label = tk.Label(self.root, text="Prediction: None", font=("Arial", 14))
        self.result_label.pack(pady=10)
        
        # Variables for drawing
        self.last_x = None
        self.last_y = None
        self.image = Image.new("L", (200, 200), 255)  # White background
        self.draw_image = ImageDraw.Draw(self.image)
    
    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y
    
    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill='black', width=8)
            # Draw on PIL image
            self.draw_image.line([self.last_x, self.last_y, event.x, event.y], fill=0, width=8)
        self.last_x, self.last_y = event.x, event.y
    
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw_image = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: None")
        self.last_x, self.last_y = None, None
    
    def predict(self):
        # Save the drawn image
        self.image.save("drawn_digit.png")
        
        # Predict the digit
        try:
            predicted_digit = predict_digit("drawn_digit.png")
            self.result_label.config(text=f"Prediction: {predicted_digit}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
        
        # Clean up the saved image
        if os.path.exists("drawn_digit.png"):
            os.remove("drawn_digit.png")

# Run the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawingApp(root)
    root.mainloop()