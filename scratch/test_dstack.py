import numpy as np
image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
matte = np.zeros((100, 100), dtype=np.uint8)
try:
    res = np.dstack((image_rgb, matte))
    print(res.shape)
except Exception as e:
    print(repr(e))
