
import torchvision
from PIL import Image as PIL_Image


def preprocess_image(image_path, img_size):
    transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size))])
    transf_2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    pil_image = PIL_Image.open(image_path)
    if pil_image.mode != 'RGB':
        pil_image = PIL_Image.new("RGB", pil_image.size)
    preprocess_pil_image = transf_1(pil_image)
    image = torchvision.transforms.ToTensor()(preprocess_pil_image)
    image = transf_2(image)
    return image.unsqueeze(0)



import torch

def pre_process(tens_image):
    tens_image *= 2    # [0, 2]
    tens_image -= 1    # [-1, +1]
    return tens_image

def post_process(tens_image):

    #tens_image += 1
    # [c, h, w] -> [h, w, c]
    tens_image = tens_image.permute(1, 2, 0)
    # is different than transpose or reshape!
    #tens_image *= 255

    tens_min = torch.min(tens_image)
    tens_image -= tens_min  # offset to 0 as minimum

    tens_max = torch.max(tens_image)

    # devo mappare ora il valore a 255
    old_interval = (tens_max - 0)
    new_interval = (255 - 0)
    tens_image *= (new_interval) / (old_interval)

    return tens_image
