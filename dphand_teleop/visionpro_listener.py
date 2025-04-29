from threading import Thread, Lock
from avp_stream import VisionProStreamer
from dphand_utils.math_utils import angle_between
import time

import numpy as np

class VisionProListener():
    def __init__(self, ip, record=False):
        self.streamer = VisionProStreamer(ip, record)
        self.lock = Lock()
        self.hand_angles = np.zeros((5, 4))

        if record == True:
            self.start_time = time.time()

    def get_data(self):
        with self.lock:
            return self.streamer.get_latest()  # 使用get_latest方法确保线程安全
    
    def get_left_keypoints(self):
        with self.lock:
            latest = self.streamer.get_latest()
            if latest is not None:
                return latest['left_fingers'][:, :3, 3]
            return None
        
    def get_left_wrist(self):
        with self.lock:
            latest = self.streamer.get_latest()
            if latest is not None:
                left_wrist = latest['left_wrist']
                return left_wrist[0, :3, 3], left_wrist[0, :3, :3]
            return None, None

    def save_recording_as_npz(self, filename):
        if self.streamer.record:
            total_run_time = time.time() - self.start_time
            np.savez_compressed(filename, data=self.streamer.recording, total_time=total_run_time, allow_pickle=True)
            print(f"Recording saved as {filename}.npz")
        else:
            print("Recording is not enabled.")
        

import time
import numpy as np

class DataReplayer:
    def __init__(self, filename, circle_replay=True):
        data = np.load(filename, allow_pickle=True)
        self.data = data['data']
        self.fps = self.data.size / data['total_time']
        self.frame_duration = 1.0 / self.fps  # 每帧的理论持续时间

        self.start_time = None          # 回放开始时间
        self.paused = False             # 暂停状态
        self.circle_replay = circle_replay

        self.current_index = 0          # 当前帧索引

    def get_frame(self):
        if self.start_time is None:
            self.start_time = time.time()
            return self.data[0]

        if self.paused:
            return self.data[self.current_index]

        # 计算经过的时间
        elapsed = time.time() - self.start_time

        # 计算理论上的当前帧索引
        target_index = int(elapsed * self.fps)

        # 处理索引越界
        if target_index >= len(self.data):
            if self.circle_replay:
                # 计算循环后的实际索引
                target_index %= len(self.data)
                # 调整基准时间保证时间连续性
                self.start_time = time.time() - (target_index * self.frame_duration)
            else:
                target_index = len(self.data) - 1
                self.paused = True

        # 更新当前索引
        self.current_index = target_index
        return self.data[self.current_index]

    def pause(self):
        self.paused = True

    def resume(self):
        if self.paused:
            # 恢复时重新校准基准时间
            self.start_time = time.time() - (self.current_index * self.frame_duration)
            self.paused = False


if __name__ == '__main__': 
    from pathlib import Path
    CUR_DIR = Path(__file__).resolve().parent
    # listener = VisionProListener(ip='192.168.3.27', record=True)
    # import time
    # start_time = time.time()
    # while time.time() - start_time < 10.0:
    #     print(listener.get_left_keypoints())
    # listener.save_recording_as_npz('../data/test_recording')
    # test DataReplayer

    replayer = DataReplayer(CUR_DIR / '../data/test_recording_renamed.npz', circle_replay=True)
    start_time = time.time()
    while replayer.paused == False:
        data = replayer.get_frame()
        print(replayer.current_index)