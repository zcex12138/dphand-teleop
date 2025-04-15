from threading import Thread, Lock
from avp_stream import VisionProStreamer

class VisionProListener():
    def __init__(self, ip, record=False):
        self.streamer = VisionProStreamer(ip, record)
        self.lock = Lock()

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
    
if __name__ == '__main__': 
    listener = VisionProListener(ip='192.168.3.45')
    while True:
        data = listener.get_data()
        print(data)