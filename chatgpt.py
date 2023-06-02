import torch
import torchvision.transforms as transforms
from PIL import Image

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
            for image in request.files.getlist('images'):
                # Preprocess the image
                img = Image.open(image)
                img = transform(img)
                img = img.unsqueeze(0)
                
                # Use the pre-trained model to predict the class
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                predicted_class = class_labels[predicted.item()]
                
                # Save the image to the static folder
                image.save(os.path.join('static', image.filename))
                
                # Return the predicted class
                statement = f"Image saved as {image.filename}. Predicted class: {predicted_class}"
                return render_template('form.html', statement=statement)
        else:
            statement = "No image in request"
            return render_template('form.html', statement=statement)
        
    return render_template('form.html')
