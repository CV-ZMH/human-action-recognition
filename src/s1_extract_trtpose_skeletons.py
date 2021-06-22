import _init_paths
import os
import time
import json
import cv2
from tqdm import tqdm
from tabulate import tabulate

from utils import utils, vis
from utils.config import Config
from utils.skeletons_io import ReadValidImagesAndActionTypesByTxt
from pose_estimation.trtpose.trtpose import TrtPose
# from tracking import DeepSort


def main():
    t0 = time.time()
    # Settings
    cfg = Config(config_file='../configs/training_config.yaml')
    cfg.merge_from_file('../configs/trtpose.yaml')
    cfg_stage = cfg[os.path.basename(__file__)]

    img_format = cfg.img_format
    weight_name = '_'.join(map(str, cfg.TRT_CFG.weight))
    cfg.TRT_CFG.weight = os.path.join(cfg.weight_folder, weight_name)

    ## IO folders
    src_imgs_folder = os.path.join(*cfg_stage.input.imgs_folder)
    src_valid_imgs = os.path.join(*cfg_stage.input.valid_imgs)
    dst_skeletons_folder = os.path.join(*cfg_stage.output.skeletons_folder)
    dst_imgs_folder = os.path.join(*cfg_stage.output.imgs_folder)
    dst_imgs_info_txt = os.path.join(*cfg_stage.output.imgs_info_txt)

    # initiate trtpose
    trtpose = TrtPose(**cfg.TRTPOSE, **cfg.TRT_CFG)
    # deepsort = DeepSort(**cfg.DEEPSORT)

     # Init output path
    print(f"[INFO] Creating output folder -> {os.path.dirname(dst_skeletons_folder)}")
    os.makedirs(dst_imgs_folder, exist_ok=True)
    os.makedirs(dst_skeletons_folder, exist_ok=True)
    os.makedirs(os.path.dirname(dst_imgs_info_txt), exist_ok=True)

    # train val images reader
    images_loader = ReadValidImagesAndActionTypesByTxt(src_imgs_folder,
                                                       src_valid_imgs,
                                                       img_format)
    images_loader.save_images_info(dst_imgs_info_txt)
    print(f'[INFO] Total Images -> {len(images_loader)}')

    # Read images and process
    tq = tqdm(range(len(images_loader)), total=len(images_loader))
    for i in tq:
        img_bgr, label, img_info = images_loader.read_image()
        img_disp = img_bgr.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # predict trtpose skeleton and save to file as openpose format
        counts, objects, peaks = trtpose.predict(img_rgb)
        trtpose_keypoints = trtpose.get_keypoints(objects, counts, peaks)
        # remove bad skeletons
        trtpose_keypoints = trtpose.remove_persons_with_few_joints(trtpose_keypoints,
                                                                   min_total_joints=5,
                                                                   min_leg_joints=0,
                                                                   include_head=True)
        if not trtpose_keypoints.tolist(): continue
        openpose_keypoints = utils.trtpose_to_openpose(trtpose_keypoints)
        skeletons, _ = trtpose.keypoints_to_skeletons_list(openpose_keypoints)

        # save skeleton draw image
        save_name = img_format.format(i)
        img_name = os.path.join(dst_imgs_folder, save_name)

        vis.draw_persons_keypoints(img_disp, trtpose_keypoints, draw_numbers=True)
        cv2.imwrite(img_name, img_disp)
        cv2.imshow('result', img_disp)
        cv2.waitKey(1)

        # save skeleton txt
        skeleton_txt = os.path.join(dst_skeletons_folder, save_name[:-4]+'.txt')
        save_data = [img_info + skeleton for skeleton in skeletons]
        with open(skeleton_txt, 'w') as f:
            json.dump(save_data, f)

        # update progress bar descriptions
        tq.set_description(f'action -> {label}')
        tq.set_postfix(num_of_person=len(trtpose_keypoints))

    tq.close()
    cv2.destroyAllWindows()
    t1 = time.gmtime(time.time()-t0)
    total_time = time.strftime("%H:%M:%S", t1)

    print('Total Extraction Time', total_time)
    print(tabulate([list(images_loader.labels_info.values())],
                   list(images_loader.labels_info.keys()), 'grid'))

if __name__ == '__main__':
    main()