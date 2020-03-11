import os
import argparse
from collections import defaultdict, OrderedDict
import re
import json
import logging
import h5py

import numpy as np
from util import  gen_video_detections

logger = logging.getLogger()

def main(args):

    video_list=[]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    outfile=None
    if args.h5:
        outfile_path = os.path.join(args.output_path, 'DanceReID.h5')
        outfile = h5py.File(outfile_path, 'w')



    global_pid = 0
    vid = 0
    count_list = []
    video_id_dict = defaultdict(list)
    global_id_list= []


    annotation = None
    if args.annotation is not None:
        with open(args.annotation) as pf:
            annotation = json.load(pf, object_pairs_hook=OrderedDict)

    video_list= [f for f in annotation]

    logger.info('list of videos: {0}'.format(video_list))

    for video_id, video_name in enumerate(video_list):

        video_path = list(filter( re.compile('.*'+video_name+'.*'+'.mp4').match, os.listdir(args.input_path)))
        if (len(video_path)==0):
            logger.warning('{0} mp4 file not fould'.format(video_name))
            continue
        video_path = os.path.join(args.input_path, video_path[0])

        npy_list = list(filter( re.compile('.*'+video_name+'.*').match, os.listdir(args.npy_path)))
        if (len(npy_list)==0):
            logger.warning('{0} npy file not fould'.format(video_name))
            continue
        npy_path = os.path.join(args.npy_path, npy_list[0])

        valid_frames = None
        if annotation is not None:
            valid_frames = annotation[video_name]

        folder_name = '%03d_%s' %(video_id, video_name)

        if args.split_folder:
            pose_savepath = os.path.join(args.output_path, 'poses/' + folder_name)
            image_savepath = os.path.join(args.output_path, 'images/' + folder_name)
            skeleton_savepath = os.path.join(args.output_path, 'skeleton/' + folder_name)
        else:
            pose_savepath = os.path.join(args.output_path, 'poses')
            image_savepath = os.path.join(args.output_path, 'images')
            skeleton_savepath = os.path.join(args.output_path, 'skeleton')

        save_path = [pose_savepath, image_savepath, skeleton_savepath]

        logger.info("Read video in {0}".format(video_path))
        logger.info("Read dict track in {0}".format(npy_path))


        dict_track = np.load(npy_path, allow_pickle=True).item()

        if not os.path.exists(pose_savepath):
            os.makedirs(pose_savepath)
        if not os.path.exists(image_savepath):
            os.makedirs(image_savepath)
        if args.gen_skeleton and not os.path.exists(skeleton_savepath):
            os.makedirs(skeleton_savepath)

        id_dict = gen_video_detections(args, video_path, save_path, folder_name, dict_track, valid_frames, outfile, global_pid, args.down_sample)

        video_detections_cnt = 0
        for pid in sorted(id_dict):
            global_id_list.append(id_dict[pid])
            video_detections_cnt += len(id_dict[pid])

        logger.info("Total {0} detections are generated".format(video_detections_cnt))

        video_id_dict[video_name].extend(list(range(global_pid, global_pid+len(id_dict))))
        global_pid += len(id_dict)
        count_list.append(video_detections_cnt)
        vid += 1
    
    meta_dict = {}
    meta_dict ['identities'] = global_id_list
    meta_dict ['shot'] = 'single'
    meta_dict ['name'] = 'IdolTrack-Reid'
    meta_dict ['n_detects'] = count_list
    meta_dict ['total_detects'] = sum(count_list)

    split_dict = {}

    split_dict ['trainval'] = list(range(17,88))
    split_dict ['query'] = [0,1,2,3,4,5,6,7,8,9,10,
                               11,12,13,14,15,16,88,89,90,
                               91,92,93,94,95,96,97,98,99]
    split_dict ['gallery'] = split_dict ['query']


    with open(os.path.join(args.output_path, 'meta.json'), 'w') as fp:
        json.dump(meta_dict, fp)
    with open(os.path.join(args.output_path, 'split.json'), 'w') as fp:
        json.dump([split_dict], fp)
    with open(os.path.join(args.output_path, 'video.json'), 'w') as fp:
        json.dump(video_id_dict, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generating image-based reid dataset from videos')

    parser.add_argument('-i', '--input-path', type=str, help='folder path of downloaded videos')
    parser.add_argument('-n', '--npy-path', type=str, help='folder path of raw detection files')
    parser.add_argument('-a', '--annotation', type=str, help='path to human labeled json file')
    parser.add_argument('-o', '--output-path', type=str, default='DanceReiD/')
    parser.add_argument('-d', '--down-sample', type=int, default=1)

    parser.add_argument('-nd', '--no-duplicate', action='store_true', help='not overwrite images if exist')
    parser.add_argument('-gs', '--gen-skeleton', action='store_true', help='generate skeleton rendering')
    parser.add_argument('-h5', action='store_true', help='generate a single h5 files')

    parser.add_argument('--split-folder', action='store_true', help='separate image files for different videos')


    main(parser.parse_args())

