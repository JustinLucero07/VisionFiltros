from flask import Flask, render_template
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

IMAGE_FOLDER = 'static/images'
kernel_sizes = [15, 25, 37]


def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64


def apply_morphological_operations(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    results = {'original': image_to_base64(img)}

    for ksize in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

        # Erosión
        eroded = cv2.erode(img, kernel)
        results[f'erosion_{ksize}'] = image_to_base64(eroded)

        # Dilatación
        dilated = cv2.dilate(img, kernel)
        results[f'dilatation_{ksize}'] = image_to_base64(dilated)

        # Top Hat
        top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        results[f'top_hat_{ksize}'] = image_to_base64(top_hat)

        # Black Hat
        black_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        results[f'black_hat_{ksize}'] = image_to_base64(black_hat)

        # Original + (Top Hat - Black Hat)
        enhanced = cv2.add(img, cv2.subtract(top_hat, black_hat))
        results[f'enhanced_{ksize}'] = image_to_base64(enhanced)

    return results


@app.route('/')
def index():
    image_files = ['med5.jpg', 'med6.jpg', 'med7.jpg']
    processed_images = {}

    for img_file in image_files:
        path = os.path.join(IMAGE_FOLDER, img_file)
        processed_images[img_file] = apply_morphological_operations(path)

    return render_template('index.html', processed_images=processed_images)


if __name__ == '__main__':
    app.run(debug=True)
