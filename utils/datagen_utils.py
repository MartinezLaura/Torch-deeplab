import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()

    elif dataset == 'dodo':
        n_classes = 17
        label_colours = get_dodo_labels()


    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb



def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_dodo_labels():
    return np.array([
        (201, 127, 171) ,
        (209, 103, 98) ,
        (152, 149, 217) ,
        (80, 11, 151) ,
        (169, 31, 133) ,
        (2, 21, 119) ,
        (127, 222, 59) ,
        (24, 39, 97) ,
        (0, 0, 0) ,
        (65, 105, 221) ,
        (236, 124, 186) ,
        (89, 227, 127) ,
        (13, 92, 172) ,
        (18, 175, 32) ,
        (53, 166, 39) ,
        (229, 111, 142) ,
        (75, 92, 33) ,
        (241, 247, 118),
        (26, 219, 116) ,
        (114, 80, 226) ,
        (94, 224, 34) ,
        (231, 97, 147) ,
        (220, 215, 28) ,
        (234, 217, 100) ,
        (105, 55, 62) ,
        (37, 246, 15) ,
        (99, 214, 200) ,
        (166, 156, 70) ,
        (168, 18, 9) ,
        (109, 251, 136) ,
        (1, 15, 208) ,
        (113, 68, 236) ,
        (155, 227, 201) ,
        (229, 186, 138) ,
        (178, 90, 254) ,
        (147, 186, 198)
    ])


def denormalize_image(image):
    mean=(0.485, 0.456, 0.406) 
    std=(0.229, 0.224, 0.225)

    image *= std
    image += mean

    return image