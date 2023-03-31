import json
import yaml
import glob
import os

def create_json(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    base_dir = config['dataset']['base_path']

    train_img=  glob.glob('{}/train/images/*jpg'.format(base_dir))
    train_ann = glob.glob('{}/train/masks/*png'.format(base_dir))
    train_items = []
    for i in range(len(train_img)):
        d = {'image': train_img[i], 'annotation':train_ann[i]}
        train_items.append(d)

    test_img= glob.glob('{}/test/images/*jpg'.format(base_dir))
    test_ann = glob.glob('{}/test/masks/*png'.format(base_dir))
    test_items = []
    for i in range(len(test_img)):
        d = {'image': test_img[i], 'annotation':test_ann[i]}
        test_items.append(d)

    val_img=  glob.glob('{}/val/images/*jpg'.format(base_dir))
    val_ann = glob.glob('{}/val/masks/*png'.format(base_dir))
    val_items = []
    for i in range(len(val_img)):
        d = {'image': val_img[i], 'annotation':val_ann[i]}
        val_items.append(d)


    final = {'train': train_items,
             'test': test_items,
             'val': val_items}

    with open('{}/data.json'.format(base_dir), 'w') as f:
        json.dump(final, f)



create_json('/mnt/HDD/LauraHD/deeplabV3-PyTorch/configs/config.yml')