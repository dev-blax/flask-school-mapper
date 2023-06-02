from flask import Flask, render_template, request
import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import uuid

app = Flask(__name__)

class_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


model = torchvision.models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'images' in request.files:
            image_classes = {}
            
            for image in request.files.getlist('images'):
                # Generate a unique name for the image
                unique_name = str(uuid.uuid4())
                file_ext = image.filename.rsplit('.', 1)[1].lower()
                file_name = f"{unique_name}.{file_ext}"

                # Preprocess the image
                img = Image.open(image)
                img = transform(img)
                img = img.unsqueeze(0)
                
                # Use the pre-trained model to predict the class probabilities
                outputs = model(img)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                probabilities = probabilities.detach().numpy()

                # Find the index of the class with the highest probability
                predicted_class_idx = np.argmax(probabilities)
                predicted_class = class_labels[predicted_class_idx]

                # Save the image to the static folder with the unique name
                image.save(os.path.join('static', file_name))

                # Record the image class
                image_classes[file_name] = predicted_class
                
            # Return the image names and their respective classes
            return render_template('form.html', image_classes=image_classes)
        
        else:
            statement = "No image in request"
            return render_template('form.html', statement=statement)
        
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
