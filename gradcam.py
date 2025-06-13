import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# --- GradCAM Helper ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=[1, 2])
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.clamp(cam, min=0)
        cam -= cam.min()
        cam /= cam.max()
        return cam.cpu().numpy()

# --- Load Model ---
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 8)  # replace 8 with your number of classes
model.load_state_dict(torch.load('fingerprint_blood_group_model.pth'))
model.eval()

target_layer = model.layer4[1].conv2  # deeper layer for better heatmaps
gradcam = GradCAM(model, target_layer)

# --- Load Image ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
img_path = 'test_image.jpg'  # Replace with your fingerprint image
img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)

# --- Generate CAM ---
cam = gradcam(input_tensor)
cam = cv2.resize(cam, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
img_np = np.array(img.resize((224, 224)))
superimposed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

# --- Show ---
plt.imshow(superimposed)
plt.title("Grad-CAM Heatmap")
plt.axis('off')
plt.show()
