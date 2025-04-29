import numpy as np

class MeanFilter:
    def __init__(self, window_size, dimensions):
        """
        参数:
            window_size (int): 滑动窗口的大小
            dimensions (int): 数据的维度数
        """
        self.window_size = window_size
        self.dimensions = dimensions
        # 使用环形缓冲区存储数据，提高效率
        self.buffer = np.zeros((window_size, dimensions))
        self.current_size = 0
        self.index = 0  # 环形缓冲区的当前位置
    
    def update(self, new_value):
        new_value = np.asarray(new_value)
        if new_value.shape != (self.dimensions,):
            raise ValueError(f"输入数据维度应为({self.dimensions},), 但得到{new_value.shape}")
        
        # 将新数据存入环形缓冲区
        self.buffer[self.index] = new_value
        self.index = (self.index + 1) % self.window_size
        self.current_size = min(self.current_size + 1, self.window_size)
        
        # 计算中值（使用向量化的np.median）
        if self.current_size == 0:
            return np.zeros(self.dimensions)
        
        # 提取有效窗口数据（按时间顺序）
        if self.current_size == self.window_size:
            window_data = np.roll(self.buffer, -self.index, axis=0)
        else:
            window_data = self.buffer[:self.current_size]
        
        # 计算各维度中值（向量化操作）
        return np.mean(window_data, axis=0)
    
    def reset(self):
        """重置滤波器状态"""
        self.buffer.fill(0)
        self.current_size = 0
        self.index = 0


# 示例使用
if __name__ == "__main__":
    # 创建一个窗口大小为3的2维中值滤波器
    median_filter = MeanFilter(window_size=3, dimensions=2)
    
    # 模拟一些2维数据
    test_data = [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0], 
        [4.0, 40.0],
        [5.0, 50.0],
        [4.0, 40.0],
        [3.0, 30.0],
        [2.0, 20.0],
        [1.0, 10.0]
    ]
    
    print("原始数据\t\t滤波后数据")
    for point in test_data:
        filtered = median_filter.update(point)
        print(f"{point}\t->\t{filtered}")