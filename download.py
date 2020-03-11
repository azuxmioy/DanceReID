import argparse
import glob
import csv
import os

def get_video_list(video_path, csv_file):
    # Get existing videos
    existing_vids = glob.glob("%s/*.mp4" % video_path)
    for idx, vid in enumerate(existing_vids):
        basename = os.path.basename(vid).split(".mp4")[0]
        if len(basename) != 0:
            existing_vids[idx] = basename
        else:
            raise RuntimeError("Unknown filename format: %s", vid)
    # Read an get video IDs from annotation file

    result = {}

    with open (csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column, value in row.items():
                result.setdefault(column,[]).append(value)

    all_file_ids = result['File_name']
    all_vid = result['Youtube_id']

    non_existing_fid = []
    non_existing_vid = []

    for fid, vid in zip(all_file_ids, all_vid):
        if fid in existing_vids:
            continue
        else:
            non_existing_fid.append(fid)
            non_existing_vid.append(vid)

    return non_existing_fid, non_existing_vid

def main(args):
    non_existing_fid, non_existing_vid  = get_video_list(args.output_path, args.csv_file)
    filename = os.path.join(args.output_path, "%s")
    cmd_base = "youtube-dl -f 'bestvideo[height<=1080][fps<=30][ext=mp4]' "
    cmd_base += '"https://www.youtube.com/watch?v=%s" '
    cmd_base += '-o "%s"' % filename

    with open(args.tmp_file, "w") as f:
        for i, (fid, vid) in enumerate(zip(non_existing_fid, non_existing_vid)):
            cmd = cmd_base % (vid, fid)
            f.write("%s\n" % cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='to download videos for ReID')

    parser.add_argument('-o', "--output_path", type=str, default='videos', help="Path of the output video folder")
    parser.add_argument('-c', "--csv_file", default='video_data.csv', help="Location of the csv file (video list)")
    parser.add_argument('-t', "--tmp_file", default='tmp.sh', help="Output script location.")
    
    main(parser.parse_args())

