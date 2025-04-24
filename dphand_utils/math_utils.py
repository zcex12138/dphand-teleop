import numpy as np
from scipy.spatial.transform import Rotation

def mtx2quat(R):
    r = Rotation.from_matrix(R)
    q = r.as_quat() # 返回顺序为(x, y, z, w)
    # 转换为(w, x, y, z)格式
    return np.array([q[3], q[0], q[1], q[2]])

def quat2rpy(q):
    w, x, y, z = q

    # 计算pitch（绕Y轴旋转）
    sin_pitch = -2 * (x * z - w * y)
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)  # 避免数值误差
    pitch = np.arcsin(sin_pitch)

    # 检查万向锁（cos_pitch接近0）
    cos_pitch = np.cos(pitch)
    if np.abs(cos_pitch) < 1e-7:
        # 万向锁情况，设置roll为0并计算yaw
        roll = 0.0
        # 计算yaw时避免除以0，使用替代公式
        yaw = np.arctan2(2 * (x * y + w * z), 1 - 2 * (y**2 + z**2))
    else:
        # 计算roll（绕X轴旋转）
        roll = np.arctan2(2 * (y * z + w * x), 1 - 2 * (x**2 + y**2))
        # 计算yaw（绕Z轴旋转）
        yaw = np.arctan2(2 * (x * y + w * z), 1 - 2 * (y**2 + z**2))

    return np.array([roll, pitch, yaw])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    w = w / np.linalg.norm([w, x, y, z])
    x = x / np.linalg.norm([w, x, y, z])
    y = y / np.linalg.norm([w, x, y, z])
    z = z / np.linalg.norm([w, x, y, z])

    return np.array([w, x, y, z])

def rpy2mtx(r,p,y):
    # 内旋XYZ顺序
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(r), -np.sin(r)],
                    [0, np.sin(r), np.cos(r)]])
    R_y = np.array([[np.cos(p), 0, np.sin(p)],
                    [0, 1, 0],
                    [-np.sin(p), 0, np.cos(p)]])
    R_z = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0],
                    [0, 0, 1]])
    # 组合旋转矩阵
    R = np.dot(R_x, np.dot(R_y, R_z))
    return R

def rpy2quat(r, p, y):
    R = rpy2mtx(r, p, y)
    q = mtx2quat(R)
    return q

def mtx2rpy(R):
    r = Rotation.from_matrix(R)
    return r.as_euler('XYZ', degrees=False)

def angle_between(v1, v2):
    """
    计算两组向量之间的夹角
    :param v1: 向量1 Nx3
    :param v2: 向量2 Nx3
    :return: 夹角（弧度）
    """
    # 计算向量的单位向量
    v1_unit = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_unit = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    # 计算点积
    dot_product = np.sum(v1_unit * v2_unit, axis=-1)
    # 计算夹角
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # 限制在[-1, 1]范围内以避免数值误差
    return angle
