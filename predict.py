import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("cat_dog_model.keras")

root = tk.Tk()
root.title("Cat VS Dog Classifier")
root.geometry("500x600")

label = tk.Label(root, text="Select an Image", font=("Arial", 18))
label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="", font=("Arial", 20, "bold"))
result_label.pack(pady=20)

def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    img = Image.open(file_path)
    img_display= img.resize((250, 250))
    photo = ImageTk.PhotoImage(img_display)
    image_label.config(image=photo)
    image_label.image=photo

    img=img.resize((150,150))
    img_array=np.array(img)/255.0
    img_array=np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0]>0.5:
        result="DOG🦮"
        confidence=prediction[0][0]*100
    else:
        result="CAT🐈"
        confidence=(1-prediction[0][0])*100
    result_label.config(text=f"{result}\nConfidence:{confidence:2f}%")
btn=tk.Button(
    root,text="Select Image",command=predict_image,font=("Arial,16")
)
btn.pack(pady=20)

root.mainloop()
