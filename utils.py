import torch
from torchvision import transforms
from PIL import Image
from model import DEVICE

CLASSES = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]

CUSTOM_CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

RESNET18_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def image_transform(uploaded_file, model_name: str) -> torch.Tensor:
    image = Image.open(uploaded_file).convert("RGB")
    if model_name == "Custom_CNN":
        return CUSTOM_CNN_TRANSFORM(image)
    return RESNET18_TRANSFORM(image)


def predict(transformed_image: torch.Tensor, model: torch.nn.Module):
    batch_img = transformed_image.unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.inference_mode():
        raw_logits = model(batch_img)
        probs = torch.softmax(raw_logits, dim=1)[0]
        predicted_index = probs.argmax().item()
        confidence = probs[predicted_index].item() * 100.0

    return CLASSES[predicted_index], confidence