# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile

import mmcv
import numpy as np
# from mmengine.utils import ProgressBar, mkdir_or_exist
from multiprocessing.dummy import Pool as ThreadPool


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert whu-opt-sar dataset to mmsegmentation format')
    parser.add_argument(
        '--dataset_path', 
        default="/map-hl/qibin.he/data/rs/whu-opt-sar/",
        help='potsdam folder path')
    parser.add_argument(
        '-o', '--out_dir', 
        default="/map-hl/qibin.he/data/rs/whu-opt-sar/",
        help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, clip_save_dir, args, to_label=False):
    # Original image of Potsdam dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.
    if to_label:
        image = mmcv.imread(image_path, flag="unchanged") // 10
        image[image == 7] = 0
        h, w = image.shape
    else:
        image = mmcv.imread(image_path)
        if "optical" in image_path:
            image = image[..., [2, 1, 0]]
        h, w, c = image.shape
    
    clip_size = args.clip_size
    stride_size = args.stride_size

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    # xmin = x * clip_size
    # ymin = y * clip_size
    xmin = x * stride_size
    ymin = y * stride_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack(
        [xmin + xmin_offset, ymin + ymin_offset,
         np.minimum(xmin + clip_size, w),
         np.minimum(ymin + clip_size, h)],
        axis=1)

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        file_name = osp.basename(image_path).split('.')[0]
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(
                clip_save_dir,
                f'{file_name}_{start_x}_{start_y}_{end_x}_{end_y}.png'))


def label_vis_single(img_name, color_map, clip_dir, label_dir, out_dir):
    if not img_name.endswith(".png"):
        return
    img_path = osp.join(clip_dir, img_name)
    label_path = osp.join(label_dir, img_name)
    img = mmcv.imread(img_path)
    label = mmcv.imread(label_path, flag="grayscale")
    label_vis = np.zeros(img.shape)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            label_vis[i, j] = color_map[int(label[i, j])]
    # label_blend = img * 0.5 + label_vis * 0.3
    label_blend = label_vis 
    mmcv.imwrite(label_blend, osp.join(out_dir, img_name))
    return


def label_vis(clip_dir, label_dir, out_dir, pool_num=10):
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    color_map = np.array([[152, 102, 152], [0, 102, 204], [0, 0, 255],
                          [0, 255, 255], [255, 0, 0], [0, 165, 85],
                          [255, 255, 93]])
    class_names = ["others", "farmland", "city", "village", "water", "forest", "road"]
    
    pool = ThreadPool(pool_num)
    for img_name in os.listdir(clip_dir):
        pool.apply_async(label_vis_single, args=(img_name, color_map, clip_dir, label_dir, out_dir))
        # label_vis_single(img_name, color_map, clip_dir, label_dir, out_dir)
    pool.close()
    pool.join()
    return


def read_txt(txt_path):
    ids = []
    with open(txt_path, "r") as txt:
        for line in txt.readlines():
            ids.append(line.strip("\n"))
    return ids


def main():
    args = parse_args()
    train_ids_txt = "/map-hl/qibin.he/data/rs/whu-opt-sar/train_ids.txt"
    test_ids_txt = "/map-hl/qibin.he/data/rs/whu-opt-sar/test_ids.txt"
    splits = {
        'train': read_txt(train_ids_txt),
        'val': read_txt(test_ids_txt)
    }

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'potsdam')
    else:
        out_dir = args.out_dir
    
    label_dir = osp.join(args.dataset_path, "lbl")
    opt_img_dir = osp.join(args.dataset_path, "optical")
    sar_img_dir = osp.join(args.dataset_path, "sar")
    
    for img_dir in [opt_img_dir, sar_img_dir]:
    # for img_dir in [opt_img_dir]:
        if img_dir == opt_img_dir:
            out_dir = osp.join(args.out_dir, f"optical_{args.clip_size}_s{args.stride_size}")
            print('Making directories for Optical data.')
        else:
            out_dir = osp.join(args.out_dir, f"sar_{args.clip_size}_s{args.stride_size}")
            print('Making directories for SAR data.')
        mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
        mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
        mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
        mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

        src_path_list = []
        for source in [label_dir, img_dir]:
            for fl in os.listdir(source):
                if fl.endswith(".tif"):
                    src_path_list.append(osp.join(source, fl))
        
        prog_bar = mmcv.ProgressBar(len(src_path_list))
        for i, src_path in enumerate(src_path_list):
            file_name = osp.basename(src_path)
            data_type = 'train' if file_name in splits['train'] else 'val'
            if 'lbl' in src_path:
                dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                clip_big_image(src_path, dst_dir, args, to_label=True)
            else:
                dst_dir = osp.join(out_dir, 'img_dir', data_type)
                clip_big_image(src_path, dst_dir, args, to_label=False)
            prog_bar.update()

    print('Removing the temporary files...')
    print('Done!')


if __name__ == '__main__':
    main()

    # clip_dir = "/map-hl/qibin.he/data/rs/whu-opt-sar/optical_512_s512/img_dir/val"
    # label_dir = "/map-hl/qibin.he/data/rs/whu-opt-sar/optical_512_s512/ann_dir/val"
    # out_dir = "/map-hl/qibin.he/data/rs/whu-opt-sar/optical_512_s512/vis_dir/val"
    # label_vis(clip_dir, label_dir, out_dir, pool_num=10)
