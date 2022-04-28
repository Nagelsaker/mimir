import numpy as np
import cv2, PIL
import os, sys
from cv2 import aruco
from PIL import Image

def main():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # nx = 4
    # ny = 3
    # for i in range(1, nx*ny+1):
    #     ax = fig.add_subplot(ny,nx, i)
    #     img = aruco.drawMarker(aruco_dict,i, 700)
    #     plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
    #     ax.axis("off")
    aruco_idx = 7
    img = Image.fromarray(aruco.drawMarker(aruco_dict,aruco_idx, 700))
    file_path = os.path.dirname(sys.argv[0])
    img.save(f"{file_path}/lever_marker_a{aruco_idx}.png")


if __name__ == "__main__":
    main()