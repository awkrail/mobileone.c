import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from mobileone import mobileone

def main():
    model = mobileone(inference_mode=True)
    checkpoint = torch.load('./mobileone_s0.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

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
    # stage 0
    output_stage0 = model.stage0(input_tensor)

    # stage 1
    output_stage1 = model.stage1(output_stage0)

    # stage 2
    output_stage2 = model.stage2(output_stage1)

    # stage 3
    output_stage3 = model.stage3(output_stage2)

    # stage 4
    output_stage4 = model.stage4(output_stage3)

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
