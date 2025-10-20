import torch
from mobileone import mobileone

def main():
    model = mobileone(inference_mode=True)
    checkpoint = torch.load('./mobileone_s0.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    for name, param in model.named_parameters():
        param = param.flatten().detach().numpy()
        param.tofile(f'../weights/{name}.bin')

    # save pytorch model as onnx
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        "mobileone_s0.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )


if __name__ == "__main__":
    main()
