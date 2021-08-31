"""
Credit to original code https://github.com/tryolabs/norfair/blob/master/norfair/video.py
modified to get output file path as user provided filename suffix and
changed constructor of Video class
"""

import os
import os.path as osp
import time
from typing import List, Optional, Union, Tuple

import cv2
import numpy as np
from rich import print
from rich.progress import BarColumn, Progress, ProgressColumn, TimeRemainingColumn

def get_terminal_size(default: Tuple[int, int] = (80, 24)) -> Tuple[int, int]:
    columns, lines = default
    for fd in range(0, 3):  # First in order 0=Std In, 1=Std Out, 2=Std Error
        try:
            columns, lines = os.get_terminal_size(fd)
        except OSError:
            continue
        break
    return columns, lines


class Video:
    def __init__(self, src: str):
        self.src = src
        is_webcam = lambda x: isinstance(x, int)
        self.display = 'webcam' if is_webcam(src) \
            else osp.basename(src)

        # Read Input Video
        self.video_capture = cv2.VideoCapture(src)
        if not self.video_capture.isOpened:
            self._fail(
                f"[bold red]Error:[/bold red] '{self.src}' does not seem to be a video file supported by OpenCV. If the video file is not the problem, please check that your OpenCV installation is working correctly."
                )
        self.total_frames = 0 if is_webcam(src) \
            else int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_cnt = 0

        description = 'Run'
        # Setup progressbar
        if self.display:
            description += f" | {self.display}"
            
        progress_bar_fields: List[Union[str, ProgressColumn]] = [
            "[progress.description]{task.description}",
            BarColumn(),
            "[yellow]{task.fields[process_fps]:.2f}fps[/yellow]",
        ]
        progress_bar_fields.insert(
            2, "[progress.percentage]{task.percentage:>3.0f}%"
        )
        progress_bar_fields.insert(
            3, TimeRemainingColumn(),
        )
        self.progress_bar = Progress(
            *progress_bar_fields,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self.task = self.progress_bar.add_task(
            self.abbreviate_description(description),
            total=self.total_frames,
            start=self.src,
            process_fps=0,
        )

    # This is a generator, note the yield keyword below.
    def __iter__(self):
        with self.progress_bar as progress_bar:
            start = time.time()
            # Iterate over video
            while True:
                self.frame_cnt += 1
                ret, img = self.video_capture.read()
                if ret is False or img is None:
                    break
                self.fps = self.frame_cnt / (time.time() - start)
                progress_bar.update(
                    self.task, advance=1, refresh=True, process_fps=self.fps
                )
                yield img
            self.stop()

    def stop(self):
        self.frame_cnt = 0
        self.video_capture.release()
        cv2.destroyAllWindows()

    def _fail(self, msg: str):
        print(msg)
        exit()

    def show(self, frame: np.array, winname: str='show',downsample_ratio: float = 1.0):
        # Resize to lower resolution for faster streaming over slow connections
        if self.frame_cnt == 1:
            cv2.namedWindow(winname, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(winname, 640, 480)
            cv2.moveWindow(winname, 20, 20)

        if downsample_ratio != 1.0:
            frame = cv2.resize(
                frame,
                (
                    frame.shape[1] // downsample_ratio,
                    frame.shape[0] // downsample_ratio,
                ),
            )
        cv2.imshow(winname, frame)
        return cv2.waitKey(1)

    def get_writer(self, frame, output_path, fps=20):
        output_path = osp.join(
            output_path, osp.splitext(osp.basename(self.src))[0]+'.avi') \
            if output_path[-4:] != '.avi' else output_path
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_size = (frame.shape[1], frame.shape[0]) # OpenCV format is (width, height)
        writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)
        print(f'[INFO] Writing output to {output_path}')
        return writer

    def get_output_file_path(self, output_folder, suffix: List=[]) -> str:
        os.makedirs(output_folder, exist_ok=True)
        filename = '{}_' * (len(suffix)+1)
        filename = filename.format(
            'webcam' if isinstance(self.src, int) else osp.splitext(self.display)[0],
            *iter(suffix)
            )
        output_path = osp.join(output_folder, f'{filename[:-1]}.avi')
        return output_path

    def abbreviate_description(self, description: str) -> str:
        """Conditionally abbreviate description so that progress bar fits in small terminals"""
        terminal_columns, _ = get_terminal_size()
        space_for_description = (
            int(terminal_columns) - 25
        )  # Leave 25 space for progressbar
        if len(description) < space_for_description:
            return description
        else:
            return "{} ... {}".format(
                description[: space_for_description // 2 - 3],
                description[-space_for_description // 2 + 3 :],
            )

if __name__ == '__main__':
    path = '/home/zmh/hdd/Test_Videos/Tracking/aung_la_fight_cut_1.mp4'
    video = Video(path)
    for i in video:
        video.show(i, 'debug')
