import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def dump_input_tensor(input_tensor, filename):
    flattened_tensor = input_tensor.flatten().numpy()
    flattened_tensor.tofile(filename)

def main():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open('./apple.png').convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    dump_input_tensor(input_tensor, '../apple.bin')


if __name__ == "__main__":
    main()
