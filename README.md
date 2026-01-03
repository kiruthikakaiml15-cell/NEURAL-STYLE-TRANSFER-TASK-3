#codtechitsolutions-task 3

NAME:KIRUTHIKA K

INTERN ID:CTISO515

**Introduction**
Neural Style Transfer (NST) is a deep learning technique that combines the content of one image with the artistic style of another image to generate a visually appealing output. 
It uses Convolutional Neural Networks (CNNs) to separate and recombine content and style from images.
This project implements Neural Style Transfer using a pre-trained VGG19 network to apply artistic styles to photographs, producing images that preserve the structure of the original photo while adopting the colors and textures of the style image.

**Project Overview**
The goal of this project is to design and implement a Neural Style Transfer model that:
Takes a content image (photograph)
Takes a style image (artwork)
Generates a stylized output image
The implementation is done using Python and PyTorch in Google Colab, allowing GPU acceleration for faster training.

**Methodology**
The Neural Style Transfer process involves:

Feature Extraction
Use a pre-trained CNN (VGG19) to extract image features.

Content Representation
Higher-level layers capture image structure.

Style Representation
Style is captured using Gram matrices of feature maps.

Loss Optimization
Content loss + Style loss

Iterative Optimization
Update target image using gradient descent.

**Implementation**
Platform
Google Colab (GPU enabled)

Framework
PyTorch

Model
Pre-trained VGG19
The model does not require training from scratch. Instead, it optimizes a target image to minimize content and style loss.

**Techniques & Tools Used**
Techniques

Convolutional Neural Networks (CNN)

Transfer Learning

Gram Matrix for Style Representation

Gradient-based Optimization

Tools

Tool
Purpose

Python
Programming language

PyTorch
Deep learning framework

VGG19
Feature extraction

Google Colab
GPU execution

PIL
Image handling

Matplotlib
Visualization

**Step-by-Step Code Explanation**
Step 1: Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
These libraries handle neural networks, image processing, and visualization.

Step 2: Load Images
def load_image(image_path, max_size=256):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)
    Loads and preprocesses content and style images.

    Step 3: Load VGG19 Model
    vgg = models.vgg19(pretrained=True).features.eval()
    Uses a pre-trained VGG19 network for feature extraction

    Step 4: Gram Matrix Calculation
    def gram_matrix(tensor):
    c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())
    The Gram matrix represents style by capturing feature correlations.

    Step 5: Feature Extraction
    def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        features[name] = x
    return features
    Extracts content and style features from selected layers.

    Step 6: Loss Calculation
    Content Loss – Difference between content features
Style Loss – Difference between Gram matrices
total_loss = content_loss + style_weight * style_loss

Step 7: Optimization
optimizer = optim.Adam([target], lr=0.003)
Optimizes the target image using gradient descent.

Step 8: Output Generation
plt.imshow(output_image)
plt.axis("off")
Displays and saves the stylized image.

**Applications**
Artistic photo editing

Digital art generation

Film and game design

Mobile photo filters

Fashion and textile design

Augmented reality applications

**Conclusion**
This project successfully demonstrates the implementation of Neural Style Transfer using deep learning. 
By leveraging a pre-trained VGG19 network, the system effectively separates and recombines content and style features to produce visually appealing artistic images. 
The project highlights the power of transfer learning and deep neural networks in creative applications and provides a strong foundation for further optimization and real-time deployment.
