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
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument(
        '--dataset_path', 
        default="/map-hl/qibin.he/data/rs/potsdam/",
        help='potsdam folder path')
    parser.add_argument(
        '--tmp_dir', 
        default="/map-hl/qibin.he/data/rs/potsdam/tmp",
        help='path of the temporary directory')
    parser.add_argument(
        '-o', '--out_dir', 
        default="/map-hl/qibin.he/data/rs/potsdam/",
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
    image = mmcv.imread(image_path)

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
    xmin = x * stride_size
    ymin = y * stride_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ],
                     axis=1)

    if to_label:
        """
        color_map = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                              [255, 255, 0], [0, 255, 0], [0, 255, 255],
                              [0, 0, 255]])
        """
        color_map = np.array([[255, 255, 255], [255, 0, 0],
                              [255, 255, 0], [0, 255, 0], [0, 255, 255],
                              [0, 0, 255]])
        flatten_v = np.matmul(
            image.reshape(-1, c),
            np.array([2, 3, 4]).reshape(3, 1))
        out = np.zeros_like(flatten_v)
        for idx, class_color in enumerate(color_map):
            value_idx = np.matmul(class_color,
                                  np.array([2, 3, 4]).reshape(3, 1))
            out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        idx_i, idx_j = osp.basename(image_path).split('_')[2:4]
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(
                clip_save_dir,
                f'{idx_i}_{idx_j}_{start_x}_{start_y}_{end_x}_{end_y}.png'))


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
    label_blend = img * 0.5 + label_vis * 0.3
    mmcv.imwrite(label_blend, osp.join(out_dir, img_name))
    return


def label_vis(clip_dir, label_dir, out_dir, pool_num=10):
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    color_map = np.array([[255, 255, 255], [255, 0, 0], [255, 255, 0], 
                          [0, 255, 0], [0, 255, 255], [0, 0, 255]])
    
    pool = ThreadPool(pool_num)
    for img_name in os.listdir(clip_dir):
        pool.apply_async(label_vis_single, args=(img_name, color_map, clip_dir, label_dir, out_dir))
        # label_vis_single(img_name, color_map, clip_dir, label_dir, out_dir)
    pool.close()
    pool.join()
    return


def main():
    args = parse_args()
    splits = {
        'train': [
            '2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11',
            '4_12', '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7',
            '6_8', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8', '7_9'
        ],
        'val': [
            '5_15', '6_15', '6_13', '3_13', '4_14', '6_14', '5_14', '2_13',
            '4_15', '2_14', '5_13', '4_13', '3_14', '7_13'
        ]
    }

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'potsdam')
    else:
        out_dir = args.out_dir

    """
    print('Making directories...')
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    zipp_list = glob.glob(os.path.join(dataset_path, '*.zip'))
    print('Find the data', zipp_list)

    for zipp in zipp_list:
        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
            zip_file = zipfile.ZipFile(zipp)
            zip_file.extractall(tmp_dir)
            src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
            if not len(src_path_list):
                sub_tmp_dir = os.path.join(tmp_dir, os.listdir(tmp_dir)[0])
                src_path_list = glob.glob(os.path.join(sub_tmp_dir, '*.tif'))

            prog_bar = mmcv.ProgressBar(len(src_path_list))
            for i, src_path in enumerate(src_path_list):
                idx_i, idx_j = osp.basename(src_path).split('_')[2:4]
                data_type = 'train' if f'{idx_i}_{idx_j}' in splits[
                    'train'] else 'val'
                if 'label' in src_path:
                    dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                    clip_big_image(src_path, dst_dir, args, to_label=True)
                else:
                    dst_dir = osp.join(out_dir, 'img_dir', data_type)
                    clip_big_image(src_path, dst_dir, args, to_label=False)
                prog_bar.update()
    """
    
    label_dir = osp.join(args.dataset_path, "labels")
    rgb_img_dir = osp.join(args.dataset_path, "2_Ortho_RGB")
    irrg_img_dir = osp.join(args.dataset_path, "3_Ortho_IRRG")
    
    for img_dir in [rgb_img_dir, irrg_img_dir]:
        if img_dir == rgb_img_dir:
            out_dir = osp.join(args.out_dir, f"rgb_{args.clip_size}_s{args.stride_size}")
            print('Making directories for RGB.')
        else:
            out_dir = osp.join(args.out_dir, f"irrg_{args.clip_size}_s{args.stride_size}")
            print('Making directories for IRRG.')
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
            idx_i, idx_j = osp.basename(src_path).split('_')[2:4]
            data_type = 'train' if f'{idx_i}_{idx_j}' in splits[
                'train'] else 'val'
            if 'label' in src_path:
                dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                clip_big_image(src_path, dst_dir, args, to_label=True)
            else:
                dst_dir = osp.join(out_dir, 'img_dir', data_type)
                clip_big_image(src_path, dst_dir, args, to_label=False)
            prog_bar.update()

    print('Removing the temporary files...')
    print('Done!')


if __name__ == '__main__':
    # main()

    clip_dir = "/map-hl/qibin.he/data/rs/potsdam/rgb_512_s256/img_dir/train"
    label_dir = "/map-hl/qibin.he/data/rs/potsdam/rgb_512_s256/ann_dir/train"
    out_dir = "/map-hl/qibin.he/data/rs/potsdam/rgb_512_s256/vis_dir/train"
    label_vis(clip_dir, label_dir, out_dir, pool_num=10)