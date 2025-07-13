# Examples are given for numpy. This code also setups ipython/jupyter
# so that numpy arrays in the output are displayed as images
import numpy as np
from matplotlib import pyplot as plt

def show_data_as_image(data: np.array):
    plt.imshow(data, interpolation='nearest')
    plt.show()

from einops import rearrange, reduce, repeat

ims = np.load("/home/ong/personal/standford-cs336-2025/assignment1-basics/data/test_images.npy", allow_pickle=False)

print(ims.shape)

ims_r = rearrange(ims, "b (h h2) w c -> h (b w h2) c", h2=2)
print(ims_r.shape)
show_data_as_image(ims_r)
