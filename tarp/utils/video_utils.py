import numpy as np
import cv2


def ch_first2last(video):
    return video.transpose((0,2,3,1))


def ch_last2first(video):
    return video.transpose((0,3,1,2))


def resize_video(video, size):
    if video.shape[1] == 3:
        video = np.transpose(video, (0,2,3,1))
    transformed_video = np.stack([cv2.resize(im, (size)) for im in video], axis=0)
    return transformed_video

