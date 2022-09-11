#########################################################################
## Copyright (C) 2022, Roy Assa <assa.roy107@gmail.com>                ##
##                     Avraham Raviv <avrahamsapir1@gmail.com>         ##
##                     Itav Nissim <itav.nissim@gmail.com>             ##
#########################################################################

import configparser
import os
import time
import numpy as np
import torch
from termcolor import colored
from attack_OOP import Attack
from cocoms_class import *


#############################################################
# Function that loads a given model specified by model_name #
#############################################################
def load_model(weights='yolov5s'):
    model = None
    is_yolo = False
    if 'yolo' in weights.lower():
        model = torch.hub.load('ultralytics/yolov5', weights, device='cpu')
        is_yolo = True
    elif weights == 'ssd':
        return 0
    elif weights == 'mnist':
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
        # model = torch.jit.load('best_mnist.pt')
        # model.eval()
        is_yolo = "mnist"
    return model, is_yolo


##########################################################
# Function that runs the OD model on a dataset of images #
# In addition, runs different attacks on these images to #
# produce Adversarial examples on the OD model           #
##########################################################
def main():
    config = configparser.ConfigParser()
    config_file_path = 'config.txt'
    config.read(filenames=config_file_path)
    starting_time = time.time()
    success_rate = 0
    target = config['ATTACK']['target']  # Attack target (check config file for further details)
    max_iter = int(config['ATTACK']['max_iter'])  # Maximal number of iterations during the attack
    noise_algorithm = config['ATTACK']['noise_algorithm']  # Chosen_Noise_Attack/White_Noise_Attack/
    amount = float(config['ATTACK']['amount'])
    path = config['DATASET']['relative_path']
    conf_level_yolo = float(config['GENERAL']['conf_yolo'])
    iou_thresh_yolo = float(config['GENERAL']['iou_thresh_yolo'])
    upper_IoU = float(config['ATTACK']['upper_IoU'])
    lower_IoU = float(config['ATTACK']['lower_IoU'])

    # cpu/gpu configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = config['GENERAL']['device']
    results = None

    # Loading a pretrained OD Model
    model_name = config['GENERAL']['model']
    model, is_yolo = load_model(model_name)
    if is_yolo is True:
        classes = classes_80
        model.conf_thres = conf_level_yolo
        model.iou_thres = iou_thresh_yolo
    elif is_yolo is False:
        classes = classes_90
    elif is_yolo == "mnist":
        classes = mnist_classes
    else:
        raise Exception("Error: Please enter correct model!")

    is_single_image = not os.path.isdir(path)
    if not is_single_image:
        fns = np.asarray([os.path.join(path, i) for i in os.listdir(path)])
        while np.any([os.path.isdir(fn) for fn in fns]):
            for fn in [i for i in fns if os.path.isdir(i)]:
                new_fns = np.asarray([os.path.join(fn, i) for i in os.listdir(fn)])
                fns = np.concatenate((fns, new_fns))
                fns = np.delete(fns, np.argwhere(fns == fn))
    else:
        fns = [path]

    for image_index, fn in enumerate(fns):

        device = 'cpu'
        results = model(fn)  # Compute a feed-forward through the OD Net in order to get results of detection
        z = torch.tensor(results.imgs[0], device=device)  # Getting back the img to attack
        X = z.unsqueeze(0).to(dtype=torch.get_default_dtype(),
                              device=device)  # Adding another dimension
        X = X.permute(0, 3, 1, 2)  # Permuting img to be in correct shape
        X = X[:, :, :640, :1280]  # Recreating shape: (1,3,640,1280)
        results = model(np.asarray(X[0].permute(1, 2, 0)))

        # Prepare all kwargs for the attack
        attack_config_args = {'model': model, 'x': X, 'results': results, 'imgPath': fn,
                              'target': target, 'image_index': image_index + 1, 'max_iter': max_iter, 'amount': amount,
                              'classes': classes, 'normalize': lambda x: x, 'base_path': None,
                              "success_color": None, "noise_algorithm": noise_algorithm,
                              "iteration_num": None, "original_pred": None, "attack_pred": None, "starting_time": None,
                              "ending_time": None, "num_FP_or_IOU_Misses": None,
                              'outputImgName': None, 'original_img': None, 'attacked_img': None, 'upper_IoU': upper_IoU,
                              'lower_IoU': lower_IoU
                              }
        attack_obj = Attack(**attack_config_args)
        returned_args = attack_obj.main_attack()

        print(returned_args['message'])
        success = returned_args['success']
        if success is True:
            success_rate += 1

    ending_time = time.time()
    success_rate = success_rate / len(fns)
    summary = colored(f"""
    ##############################################################
    ###                        SUMMARY:                        ###
    ### Attack Results On Dataset: \"{path}\":              
    ###     1. Target: {target}
    ###     2. Max iterations: {max_iter}   
    ###     3. Noise algorithm: \"{noise_algorithm}\"                               
    ###     4. Total number of images: {len(fns)}                                         
    ###     5. Attack success rate: {round(success_rate * 100, 3)}%                                            
    ###     6. Attack duration: {round(ending_time - starting_time, 3)}s  
    ##############################################################
    """, "green")
    print(summary)


if __name__ == "__main__":
    main()
