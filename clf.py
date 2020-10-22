import torch
from torchvision import models, transforms
from PIL import Image

def predict(image_path):
    resnet = models.resnet101(pretrained=True)
    
    #https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])
    
    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)
    
    resnet.eval()
    out = resnet(batch_t)
    
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
        
    # https://stackoverflow.com/questions/49036993/pytorch-softmax-what-dimension-to-use
    # softmax(input, dim = 0) # normalizes values along axis 0
    # softmax(input, dim = 1) # normalizes values along axis 1
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

