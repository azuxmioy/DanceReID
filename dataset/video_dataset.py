import cv2
import numpy as np
import math
import torch.utils.data as data
import logging
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from PIL import Image
from time import time
from tqdm import tqdm

from dataset.image_dataset import convert_image_to_tensor, resize_image
#from coviar import load, get_num_gops, get_num_frames
#from settings import PERSON_ID, BOX_POS, BOX_SCORE, POSE_POS, IMAGENET_TRANSFORM
#from settings import DICT_OUTPUT_DIR, DICT_MODEL_DIR, DICT_URL

#from interface.flownet2 import flownet

logger = logging.getLogger()


class VideoDataset(data.Dataset):

    def __init__(self, input_path, init_frame=None, num_frames=None):
        super(VideoDataset, self).__init__()
        self.input_path = input_path
        self.input_cap = cv2.VideoCapture(self.input_path)
        self.init_frame = 0 if init_frame is None else init_frame
        self.input_cap.set(cv2.CAP_PROP_POS_FRAMES, self.init_frame)
        num_frames_total = int(self.input_cap.get(cv2.CAP_PROP_FRAME_COUNT) - self.init_frame)
        self.num_frames = num_frames_total if num_frames is None else min(num_frames_total, num_frames)
        self.list_images = []

        start_time = time()
        logger.info('Start to read frames from {0}'.format(self.input_path))
        for i in tqdm(range(self.num_frames)):
            try:
                ret, image = self.input_cap.read()
                self.list_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            except Exception as e:
                logger.error(e)
                logger.error('Error occurs at {0}'.format(i))
                logger.error('Image: {0}'.format(image))

        logger.info('Complete to read {0} frames by {1}'.format(self.num_frames, time()-start_time))

    def read_frame(self, i):
        return self.list_images[i]

    @property
    def width(self):
        return int(self.input_cap.get(3))

    @property
    def height(self):
        return int(self.input_cap.get(4))

    def __getitem__(self, i):
        image = self.read_frame(i)
        return {
            'image': image
        }

    def __len__(self):
        return self.num_frames

    def __del__(self):
        self.input_cap.release()


class PoseFlowVideoDataset(data.Dataset):

    def __init__(self, video_dataset, detector_input_dim):
        super(PoseFlowVideoDataset, self).__init__()
        self.video_dataset = video_dataset
        self.detector_input_dim = detector_input_dim

    def __getitem__(self, i):

        image = self.video_dataset.read_frame(i)
        image_dim = image.shape[1], image.shape[0]
        image_dim = torch.FloatTensor([image_dim]).repeat(1, 2)
        image_resize = resize_image(image, self.detector_input_dim)
        tensor = convert_image_to_tensor(image)
        tensor_resize = convert_image_to_tensor(image_resize, unsqueeze=True)

        if torch.cuda.is_available():
            tensor = tensor.cuda()
            tensor_resize = tensor_resize.cuda()
            image_dim = image_dim.cuda()

        return {
            'tensor_resize': tensor_resize,
            'tensor': tensor,
            'image': image,
            'image_dim': image_dim
        }

    @property
    def width(self):
        return self.video_dataset.width

    @property
    def height(self):
        return self.video_dataset.height

    def __len__(self):
        return len(self.video_dataset)


# Add it later
"""
class CoviarDataset(VideoDataset):
    def __init__(self, list_file_paths):
        super(Coviar).__init__(list_file_paths)
        self.current_video_gops = None
        # fpg stands for frame per GOP, number of frames in each GOP
        self.current_video_fpg = None
        self.repr_type = None
        self.accumulated = False

    def set_video(self, index):
        super(Coviar, self).set_video(index)
        self.current_video_gops = get_num_gops(self.current_video)
        self.current_video_frames = get_num_frames(self.current_video)
        self.current_video_fpg = int(self.current_video_frames / self.current_video_gops)

    def set_type(self, repr_type, accumulated):
        self.repr_type = repr_type # 'rgb' or 'p-frame'
        self.accumulated = accumulated # True or False

    def get_rgb(self, index):
        gop, frame = self.get_gop_and_frame(index)
        return load(self.current_video, gop, frame, 0, self.accumulated)

    def get_p_frame(self, index):
        gop, frame = self.get_gop_and_frame(index)
        return [load(self.current_video, gop, frame, 1, self.accumulated),
                load(self.current_video, gop, frame, 2, self.accumulated)]

    def get_gop_and_frame(self, index):
        gop = int(index / self.current_video_fpg)
        frame = index % self.current_video_fpg
        return gop, frame

    def __getitem__(self, frame_n):
        if self.repr_type == 'rgb':
            frame = self.get_rgb(frame_n)
        elif self.repr_type == 'p-frame':
            frame = self.get_p_frame(frame_n)
        else:
            return None
        return self.get_data(frame)
"""


class VideoWriter(object):

    def __init__(self, output_path, width, height):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_writer = cv2.VideoWriter(self.output_path, fourcc, 20.0, (width, height))

    def write_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.output_writer.write(frame)

    def __del__(self):
        self.output_writer.release()
