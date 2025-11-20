from dphand_teleop.retarget import Retarget
import mujoco
from pathlib import Path
import numpy as np

XML_PATH = Path(__file__).resolve().parent.parent / 'assets' / 'franka_emika_panda' / 'scene.xml'
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

from dphand_utils.math_utils import rpy2mtx

class FrankaRetargeting(Retarget):
    def __init__(self):
        self._model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self._data = mujoco.MjData(self._model)
        self.actuator_dim = 8
        self.rot_transform = rpy2mtx(np.pi/2, 0, -np.pi/2)
    
    def retarget(self, target_pos):
        return np.linalg.norm(target_pos[9] - target_pos[4])