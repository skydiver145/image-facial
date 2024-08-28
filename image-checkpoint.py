import cv2
import numpy as np
from PIL import Image
import os

def process_image(input_path, output_path, target_size=(250,250)):
    image = cv2.imread(input_path)

    if image is None:
        print(f"Error: Unable to read image")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    print(f"Number of faces detected: {len(faces)}")
    print(f"Image shape: {rgb_image.shape}")
    print(f"Image dtype: {rgb_image.dtype}")
    print(f"Image min value: {rgb_image.min()}")
    print(f"Image max value: {rgb_image.max()}")

    if len(faces) > 0:
       for i, (x, y, w, h) in enumerate(faces):
          padding_h = int(w * 0.5)
          padding_y = int(h * 0.5)
          left = max(0, x - padding_h)
          top = max(0, y - padding_y)
          right = min(rgb_image.shape[1], x + w + padding_h)
          bottom = min(rgb_image.shape[0], y + h + padding_y)

          face_image = rgb_image[top:bottom, left:right]

          pil_face = Image.fromarray(face_image)
          pil_face.thumbnail(target_size, Image.LANCZOS)

          base_name = os.path.splitext(os.path.basename(input_path))[0]
          output_filename = f"{base_name}_face_{i+1}.jpg"
          output_path = os.path.join(output_folder, output_filename)
          
          pil_face.save(output_path)
          print(f"Processed face{i+1} saved to {output_path}")
    else:
       print("No faces detected in {input_path}")

input_folder = r"C:\Users\jts_pc\Desktop\data\Your photos\Ashton"
output_folder = r"D:\nasigoreng\results\Ashton"  

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
   if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
      input_path = os.path.join(input_folder, filename)
      process_image(input_path, output_folder)