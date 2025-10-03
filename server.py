import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify


# Define the same CNN architecture as in training
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=3, padding=1
            ),  # input channel 1, output channel 32
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model.eval()  # set to evaluation mode

# Define the image transformations (same as during training)
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # ensure image is grayscale
        transforms.Resize((28, 28)),  # resize to MNIST dimensions
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


@app.route("/")
def index():
    # Render a simple HTML page for uploading images
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    try:
        # Read image file
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("L")  # convert to grayscale

        # Preprocess the image
        img = transform(img)
        img = img.unsqueeze(0)  # add batch dimension

        # Run the model inference
        with torch.no_grad():
            outputs = model(img.to(device))
            _, predicted = torch.max(outputs.data, 1)

        # Return the prediction as JSON
        return jsonify({"prediction": int(predicted.item())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
