import time
import os
import os.path as osp

import cv2
import numpy as np
import pyrealsense2 as rs


class RosbagReader:
    def __init__(self, src):
        self.stopped = False
        self.frame_cnt = 0
        self.src = src
        self.name = osp.split(src)[-1]
        self.display = osp.basename(src)

    def start(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, self.src, False)
        self.profile = self.pipeline.start(config)
        self.profile.get_device().as_playback().set_real_time(False) #get playback device and disable realtime playback

        for x in range(5):
            self.pipeline.wait_for_frames()
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()
        return self

    def __iter__(self):
        while True:
            try:
                frameset = self.pipeline.wait_for_frames()
                extracted_data = self.extract_frameset(frameset)
                # if len(extracted_data) == 0:
                #     continue
                self.frame_cnt += 1
                yield extracted_data

            except Exception as e:
                # print(f"Error : {e}")
                break
        self.stop()

    def stop(self):
        # print('reading frame stopped')
        self.pipeline.stop()
        self.frame_cnt = 0

    def extract_frameset(self, frameset):
        frameset = self.align.process(frameset)
        depth_frame = frameset.get_depth_frame()
        rgb_frame = frameset.get_color_frame()

        extracted_data = {}
        if depth_frame or rgb_frame:
            # depth and frames are converted to numpy
            extracted_data["rgb"] = np.asanyarray(rgb_frame.get_data())
            extracted_data["depth"]= np.asanyarray(depth_frame.get_data())
            colored_depth = self.colorizer.colorize(depth_frame).get_data()
            extracted_data["colored_depth"]= np.asanyarray(colored_depth)
        return extracted_data

    def show(self, frame, winname=None, downsample=1.0):
        # Resize to lower resolution for faster streaming over slow connections
        winname = winname if winname else self.name
        if self.frame_cnt == 1:
            cv2.namedWindow(winname, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(winname, 640, 480)
            cv2.moveWindow(winname, 20, 20)
        if downsample != 1.0:
            frame = cv2.resize(
                frame,
                (frame.shape[1] // downsample, frame.shape[0] // downsample),
            )
        cv2.imshow(winname, frame)
        return cv2.waitKey(1)

    def get_output_file_path(self, output_folder, suffix):
        os.makedirs(output_folder, exist_ok=True)
        filename = '{}_' * (len(suffix)+1)
        filename = filename.format(
            'webcam' if isinstance(self.src, int) else osp.splitext(self.display)[0],
            *iter(suffix)
        )
        output_path = osp.join(output_folder, f'{filename[:-1]}.avi')
        return output_path

    def get_writer(self, frame, output_path, fps=20):
        output_path = osp.join(
            output_path, osp.splitext(osp.basename(self.src))[0]+'.avi') \
            if output_path[-4:] != '.avi' else output_path
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_size = (frame.shape[1], frame.shape[0]) # OpenCV format is (width, height)
        writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)
        print(f'[INFO] Writing output to {output_path}')
        return writer


if __name__ == '__main__':
    from drawer import draw_frame_info
    bag_file = '/home/zmh/Desktop/HDD/Datasets/action_dataset/gw_drunken_action/realsense_records/fighting_test3.bag'

    reader = RosbagReader(bag_file).start()
    for extracted_data in reader:
        bgr_img = extracted_data['rgb'][...,::-1]
        depth = extracted_data['depth']
        display_img = draw_frame_info(
            bgr_img,
            add_blank=False,
            frame=reader.frame_cnt,
        )
        key = reader.show(display_img)
        if key == 27:
            print('Elapsed Time {:.5f}s'.format(time.time()-reader.start_time))
            reader.stop()
            break
    cv2.destroyAllWindows()
