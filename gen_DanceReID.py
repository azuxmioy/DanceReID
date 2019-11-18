import os
import argparse
from tqdm import tqdm
from collections import defaultdict 
import re
import cv2
import glob
import json

import numpy as np
from time import time
from PIL import Image, ImageColor
from dataset.video_dataset import VideoDataset


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

def parse_dict_track(video_dataset, dict_track, valids, output_height = 256, output_width = 128):

    frame_dict = defaultdict(dict)
    id_dict = defaultdict(lambda: defaultdict(list))

    for frame_idx, dict_poses in tqdm(dict_track.items()):

        image = video_dataset.read_frame(frame_idx)

        list_labels = []
        list_images = []
        list_image_pose = []

        for bid, dict_pose in dict_poses.items():

            pid = dict_pose['id'] - 1

            if valids is not None:
                if frame_idx not in valids[str(pid)]: continue

            box_pos = dict_pose['box_pos']
            up_left = [box_pos[2], box_pos[0]]
            bot_right = [box_pos[3], box_pos[1]]

            image_crop, shift_pose = cropBox(image, dict_pose['pose_pos'], up_left, bot_right, output_height, output_width)
            image_pil = Image.fromarray(image_crop)

            id_dict [pid]['frame_idx'].extend([frame_idx])
            id_dict [pid]['images'].extend([image_pil])
            id_dict [pid]['poses'].extend([shift_pose])

    print('Parsing to dictionary done!!!')

    return id_dict


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



def main(args):

    video_list=[]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    global_pid = 0
    vid = 0

    video_id_dict = defaultdict(list)
    global_id_list= []


    annotation = None
    if args.annotation is not None:
        with open(args.annotation) as pf:
            annotation = json.load(pf)

    video_list= [f for f in annotation]

    print(video_list)

    for video_id, video_name in enumerate(video_list):

        video_path = list(filter( re.compile('.*'+video_name+'.*'+'.mp4').match, os.listdir(args.input_path)))
        if (len(video_path)==0):
            print ("Warning: mp4 file not fould")
            continue
        video_path = os.path.join(args.input_path, video_path[0])

        npy_list = list(filter( re.compile('.*'+video_name+'.*').match, os.listdir(args.npy_path)))
        if (len(npy_list)==0):
            print ("Warning: npy file not fould")
            continue
        npy_path = os.path.join(args.npy_path, npy_list[0])

        valid_frames = None
        if annotation is not None:
            valid_frames = annotation[video_name]

        print(video_path)
        print(npy_path)

        dict_track = np.load(npy_path, allow_pickle=True).item()
        video_dataset = VideoDataset(video_path)

        assert (video_dataset.num_frames == len(dict_track))

        print("Read %d frames in %s" % (len(dict_track), video_name))
        print("Now parsing tracklets with sample")


        id_dict =  parse_dict_track (video_dataset, dict_track, valid_frames)

        del video_dataset


        if args.split_folder:
            pose_savepath = os.path.join(args.output_path, 'poses/' + '%3d_%s' %(video_id, video_name))
            image_savepath = os.path.join(args.output_path, 'images/' + '%3d_%s' %(video_id, video_name))
            skeleton_savepath = os.path.join(args.output_path, 'skeleton/' + '%3d_%s' %(video_id, video_name))
        else:
            pose_savepath = os.path.join(args.output_path, 'poses')
            image_savepath = os.path.join(args.output_path, 'images')
            skeleton_savepath = os.path.join(args.output_path, 'skeleton')

        if not os.path.exists(pose_savepath):
            os.makedirs(pose_savepath)
        if not os.path.exists(image_savepath):
            os.makedirs(image_savepath)
        if args.gen_skeleton and not os.path.exists(skeleton_savepath):
            os.makedirs(skeleton_savepath)

        print('Start writing cropped images and coresponding poses......')

        #for re_idx, (pid, dict_list) in enumerate(id_dict.items()):
        for pid in sorted(id_dict):
            print('pid %d' % pid)
            dict_list = id_dict[pid]
            
            pid_list=[]

            for frame, im, pose in zip( dict_list['frame_idx'], dict_list['images'], dict_list['poses']):

                filename = '%08d' % (global_pid + pid)
                filename += '_%06d' % frame

                if args.split_folder:
                    pid_list.append( video_name + '/' + filename + '.jpg')
                else: 
                    pid_list.append( filename + '.jpg')

                if not os.path.isfile(os.path.join(pose_savepath, filename+'.txt')) or not args.no_duplicate:
                    np.savetxt(os.path.join(pose_savepath, filename+'.txt'), pose, fmt="%s")

                if not os.path.isfile(os.path.join(image_savepath, filename+'.jpg')) or not args.no_duplicate:
                    im.save(os.path.join(image_savepath, filename+'.jpg'),'JPEG')

                if args.gen_skeleton:
                    if not os.path.isfile(os.path.join(skeleton_savepath, filename+'.jpg')) or not args.no_duplicate:
                        render_img = draw_pose_cv2(im, pose, pid)
                        render_img.save(os.path.join(skeleton_savepath, filename+'.jpg'),'JPEG')

            global_id_list.append(pid_list)


        video_id_dict[video_name].extend(list(range(global_pid, global_pid+len(id_dict))))
        global_pid += len(id_dict)
        vid += 1
        del id_dict
    
    meta_dict = {}
    meta_dict ['identities'] = global_id_list
    meta_dict ['shot'] = 'single'
    meta_dict ['name'] = 'IdolTrack-Reid'

    '''
    split_dict = {}
    split_vid = len(video_id_dict)//2
    split_dict ['query'] = list(range(0, )
    split_dict ['trainval'] = list(range(global_pid//2, global_pid))
    '''

    with open(os.path.join(args.output_path, 'meta.json'), 'w') as fp:
        json.dump(meta_dict, fp)
    #with open(os.path.join(args.output_path, 'split.json'), 'w') as fp:
    #    json.dump(split_dict, fp)
    with open(os.path.join(args.output_path, 'video.json'), 'w') as fp:
        json.dump(video_id_dict, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generating image-based reid dataset for videos')

    parser.add_argument('-i', '--input-path', type=str)
    parser.add_argument('-n', '--npy-path', type=str)
    parser.add_argument('-a', '--annotation', type=str, default=None)
    parser.add_argument('-o', '--output-path', type=str, default='DanceReiD/')
    parser.add_argument('-nd', '--no-duplicate', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('-gs', '--gen-skeleton', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--split-folder', type=bool, nargs='?', const=True, default=False)


    main(parser.parse_args())

