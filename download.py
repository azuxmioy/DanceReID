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

    all_vids = result['File_name']

    non_existing_videos = []

    for vid in all_vids:
        if vid in existing_vids:
            continue
        else:
            non_existing_videos.append(vid)

    return non_existing_videos

def main(args):
    non_existing_videos = get_video_list(args.output_path, args.csv_file)
    filename = os.path.join(args.output_path, "%03d_%s")
    cmd_base = "youtube-dl -f 'bestvideo[height<=1080][fps<=30][ext=mp4]' "
    cmd_base += '"https://www.youtube.com/watch?v=%s" '
    cmd_base += '-o "%s.mp4"' % filename

    with open(args.tmp_file, "w") as f:
        for i, vid in enumerate(non_existing_videos):
            cmd = cmd_base % (vid, i, vid)
            f.write("%s\n" % cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='to download videos for ReID')

    parser.add_argument('-o', "--output_path", type=str, help="Path of the output video folder")
    parser.add_argument('-c', "--csv_file", help="Location of the csv file (video list)")
    parser.add_argument('-t', "--tmp_file", help="Output script location.")
    
    main(parser.parse_args())

