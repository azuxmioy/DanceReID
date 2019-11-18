import cv2
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import logging
import numpy as np


logger = logging.getLogger()


class ImageDataset(data.Dataset):

    def __init__(self, img_dir):
        super(ImageDataset, self).__init__()
        self.img_dir = img_dir
        self.list_file_paths = os.listdir(img_dir)
        self.list_file_paths.sort()

    def __getitem__(self, i):

        image_path = os.path.join(self.img_dir, self.list_file_paths[i].rstrip('\n').rstrip('\r'))
        image = get_image(image_path)
        image_dim = image.shape[1], image.shape[0]
        tensor = convert_image_to_tensor(image)

        return tensor, image, image_dim

    def __len__(self):
        return len(self.list_file_paths)


class YoloDataset(ImageDataset):

    def __init__(self, img_dir, input_dim):
        super(YoloDataset, self).__init__(img_dir=img_dir)
        self.input_dim = input_dim
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, i):

        image_path = os.path.join(self.img_dir, self.list_file_paths[i].rstrip('\n').rstrip('\r'))
        image = get_image(image_path)
        image_dim = image.shape[1], image.shape[0]
        image_crop = resize_image(image, self.input_dim)
        tensor_crop = convert_image_to_tensor(image_crop)

        return tensor_crop, image, image_dim


class PoseFlowImageDataset(ImageDataset):

    def __init__(self, img_dir, detector_input_dim):
        super(PoseFlowImageDataset, self).__init__(img_dir=img_dir)
        self.detector_input_dim = detector_input_dim

    def __getitem__(self, i):

        image_path = os.path.join(self.img_dir, self.list_file_paths[i].rstrip('\n').rstrip('\r'))
        image = get_image(image_path)
        image_dim = image.shape[1], image.shape[0]
        image_dim = torch.FloatTensor([image_dim]).repeat(1, 2)
        image_resize = resize_image(image, self.detector_input_dim)
        tensor = convert_image_to_tensor(image)
        tensor_resize = convert_image_to_tensor(image_resize)

        if torch.cuda.is_available():
            tensor = tensor.cuda()
            tensor_resize = tensor_resize.cuda()
            image_dim = image_dim.cuda()

        return tensor_resize, tensor, image, image_dim


def get_image(img_path):
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def convert_image_to_tensor(img, unsqueeze=False):
    img_tensor = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_tensor = torch.from_numpy(img_tensor).float().div(255.0)
    if unsqueeze:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def resize_image(img, inp_dim, maintain_aspect=True):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim

    if maintain_aspect:
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    else:
        canvas = cv2.resize(img, (inp_dim[1], inp_dim[0]))

    return canvas


# DEPRECATED
def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_im(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray
