import os
import json

from pathlib import Path

import numpy as np
import cv2


a = np.load("images/gaussian_map/0.npy")
print(a.shape)