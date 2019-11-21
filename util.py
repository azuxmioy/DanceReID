import os
from tqdm import tqdm
from collections import defaultdict 
import cv2
import h5py
import logging
from logging.config import dictConfig

import numpy as np
from time import time
from PIL import Image, ImageColor

dict_config_log = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - [%(levelname)s][%(module)s.%(funcName)s:%(lineno)d] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        }
    },

    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        }
    },
    'loggers': {
        '__main__': { # logging from this module will be logged in VERBOSE level
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console'],
        'propagate': False
    }
}
dt = h5py.special_dtype(vlen=np.uint8)

logger = logging.getLogger(__name__)
dictConfig(dict_config_log)

DICT_POSE_COCO = dict()
DICT_POSE_COCO['line_pair'] = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12),  # Body
    (11, 13), (12, 14), (13, 15), (14, 16)
]
DICT_POSE_COCO['point_col'] = [
    'green', 'blue', 'blue', 'blue', 'blue', 'yellow', 'orange', 'yellow', 'orange', 'yellow', 'orange',
    'purple', 'red', 'purple', 'red', 'purple', 'red'
]
DICT_POSE_COCO['acc_idx'] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
DICT_POSE_COCO['flip_ref'] = ((2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17))

def gen_video_detections (args, video_path, save_path, video_name, dict_track, valids, outfile, global_pid, output_height = 256, output_width = 128):

    input_cap = cv2.VideoCapture(video_path)
    num_frames = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    id_dict = defaultdict(list)

    assert (num_frames==len(dict_track))

    start_time = time()
    logger.info('Start to read frames from {0}'.format(video_path))
    
    for frame_idx, dict_poses in tqdm(dict_track.items()):
        try:
            ret, image = input_cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(e)
            logger.error('Error occurs at {0}'.format(frame_idx))
            logger.error('Image: {0}'.format(image))

        for bid, dict_pose in dict_poses.items():

            pid = dict_pose['id'] - 1

            if valids is not None:
                if frame_idx not in valids[str(pid)]: continue

            box_pos = dict_pose['box_pos']
            up_left = [box_pos[2], box_pos[0]]
            bot_right = [box_pos[3], box_pos[1]]

            image_crop, shift_pose = cropBox(image, dict_pose['pose_pos'], up_left, bot_right, output_height, output_width)
            image_pil = Image.fromarray(image_crop)

            filename = '%04d' % (global_pid + pid)
            filename += '_%06d' % frame_idx

            if args.split_folder:
                meta_name = video_name + '/' + filename + '.jpg'
            else: 
                meta_name = filename + '.jpg'

            id_dict[pid].extend( [meta_name])


            if not os.path.isfile(os.path.join(save_path[0], filename+'.txt')) or not args.no_duplicate:
                np.savetxt(os.path.join(save_path[0], filename+'.txt'), shift_pose, fmt="%s")

            if not os.path.isfile(os.path.join(save_path[1], filename+'.jpg')) or not args.no_duplicate:
                image_pil.save(os.path.join(save_path[1], filename+'.jpg'),'JPEG')

            if args.gen_skeleton:
                if not os.path.isfile(os.path.join(save_path[2], filename+'.jpg')) or not args.no_duplicate:
                    render_img = draw_pose_cv2(image_pil, shift_pose, pid)
                    render_img.save(os.path.join(save_path[2], filename+'.jpg'),'JPEG')
            
            if args.h5:
                jpg = open(os.path.join(save_path[1], filename+'.jpg'), 'rb')
                binary_data = jpg.read()
                dset = outfile.create_dataset( 'dataset/images/' + filename, shape= (1,), chunks=True, dtype=dt )
                dset[0] = np.frombuffer(binary_data, dtype=np.uint8)
                jpg.close()

                dset = outfile.create_dataset( 'dataset/poses/' + filename, data=shift_pose )

                if args.gen_skeleton:
                    jpg = open(os.path.join(save_path[2], filename+'.jpg'),'rb')
                    binary_data = jpg.read()
                    dset = outfile.create_dataset( 'dataset/skeleton/' + filename, shape= (1,), chunks=True, dtype=dt )
                    dset[0] = np.frombuffer(binary_data, dtype=np.uint8)
                    jpg.close()
                  

    logger.info('Complete to read {0} frames by {1}'.format(num_frames, time()-start_time))

    return id_dict

def cropBox(img, pose, ul, br, resH, resW):

    im_H = img.shape[0]
    im_W = img.shape[1]

    lenH = max(br[0] - ul[0], (br[1] - ul[1]) * resH / resW)
    lenW = lenH * resW / resH

    center = [ (br[0] + ul[0]) / 2, (br[1] + ul[1]) / 2]

    new_ul = [max (0, int(center[0] - lenH / 2)), max (0, int(center[1] - lenW / 2))]
    new_br = [min (im_H-1, int(center[0] + lenH / 2)), min (im_W-1, int(center[1] + lenW / 2))]

    if new_ul[0] == 0:
        new_br[0] = lenH
    elif new_br[0] == im_H-1:
        new_ul[0] = im_H-lenH

    if new_ul[1] == 0:
        new_br[1] = lenW
    elif new_br[1] == im_W-1:
        new_ul[1] = im_W-lenW


    center = [(new_br[0] + new_ul[0]) / 2, (new_br[1] + new_ul[1]) / 2]

    newImg = img[int(new_ul[0]) : int(new_br[0]), int(new_ul[1]) : int(new_br[1])].copy()

    factor = float(resH) / lenH
    newImg = cv2.resize(newImg, ( int(resW), int(resH) ) )

    shift_pose = pose - np.array([center[1], center[0]])
    shift_pose = shift_pose * factor + np.array([resW/2, resH/2])

    return newImg, shift_pose

def draw_pose_cv2(img, pose_pos, pid, point_size=5, line_size=2, draw_pose_line=True):

    img = np.asarray(img)

    def convert_int_to_color(i):
        list_colors = list(ImageColor.colormap)
        return list_colors[i * 10 % len(list_colors)]


    def convert_color_to_rgb(col, default_col='red'):
        if col in ImageColor.colormap:
            return ImageColor.getrgb(col)
        else:
            return ImageColor.getrgb(default_col)

    line_col = convert_int_to_color(pid)

    dict_pose_line = dict()
    line_pair = DICT_POSE_COCO['line_pair']
    point_col = [ImageColor.getrgb(col.strip()) for col in DICT_POSE_COCO['point_col']]
    line_col = convert_color_to_rgb(line_col)

    for i in range(pose_pos.shape[0]):
        x, y = int(pose_pos[i, 0]), int(pose_pos[i, 1])
        if x > 0 and y > 0:
            dict_pose_line[i] = (x, y)
            cv2.circle(img, (x, y), point_size, point_col[i], -1)

    if draw_pose_line:
        for start_p, end_p in line_pair:
            if start_p in dict_pose_line and end_p in dict_pose_line:
                start_p = dict_pose_line[start_p]
                end_p = dict_pose_line[end_p]
                cv2.line(img, start_p, end_p, line_col, line_size)

    return Image.fromarray(img.astype('uint8'), 'RGB')
