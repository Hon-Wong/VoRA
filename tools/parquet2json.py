import io
import json
import os
import os.path as osp
from multiprocessing import Pool

import pyarrow.parquet as pq
from PIL import Image


def process_row(row):
    # Save frames as JPEG images
    if isinstance(row['id'], str):
        row['id'] = row['id'].replace("/", "_")
    save_subfolder = save_root
    image_path = osp.join(save_subfolder, f"{row['id']}.jpg")
    image_rela_path = f"frames/{row['id']}.jpg"
    try:
        frame = row['frames'][0]
    except:
        frame = row['image']["bytes"]
    try:
        frame = Image.open(io.BytesIO(frame))
        frame.save(image_path)
    except Exception as e:
        print(e)
        return None

    row['frames'] = [image_rela_path]
    return row


def process_parquet_file(parquet_path):
    print(f"Processing {os.path.basename(parquet_path)}")
    table = pq.read_table(parquet_path)
    print(f"{os.path.basename(parquet_path)} loaded!")

    df = table.to_pandas()
    lines = [l[-1] for l in df.iterrows()]
    print(f"{os.path.basename(parquet_path)} loaded into mem!")
    with Pool(256) as pool:
        results = pool.map(process_row, lines)

    results = [r for r in results if r is not None]
    # Write the results to the JSON file
    for row in results:
        f.write(json.dumps(row.to_dict()) + '\n')

    print(f"{parquet_path} saved!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--is_video", type=bool, default=True)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    is_video = args.is_video
    dataset_name = osp.basename(dataset_dir)

    parquet_paths = [osp.join(dataset_dir, path) for path in os.listdir(dataset_dir) if path.endswith(".parquet")]

    dataset_name = "ElysiumTrack-val500"
    hdfs_paths = ['/mnt/bn/video-grounding-data-making/GroundingDataMaking/web10m_raw_save/final_version/test/500videos_new.parquet']

    save_root = f"{save_dir}/{dataset_name}/frames"
    anno_root = f"{save_dir}/{dataset_name}/annotations/"
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(anno_root, exist_ok=True)

    json_dir = osp.join(anno_root, f"{dataset_name}.json")
    f = open(json_dir, "w")

    for parquet_path in parquet_paths:
        process_parquet_file(parquet_path)
