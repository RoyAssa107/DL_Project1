#########################################################################
##         This file is part of the alpha-beta-CROWN verifier          ##
##                                                                     ##
## Copyright (C) 2021, Huan Zhang <huan@huan-zhang.com>                ##
##                     Kaidi Xu <xu.kaid@northeastern.edu>             ##
##                     Shiqi Wang <sw3215@columbia.edu>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Yihan Wang <yihanwang@ucla.edu>                 ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""alpha-beta-CROWN verifier interface to handle robustness verification."""
import configparser
import os
import socket
import random
import time
import gc

import numpy as np
import pandas as pd

import torch
from attack_pgd import pgd_attack


def load_model(model_name='ultralytics/yolov5', weights='yolov5s'):
    model = torch.hub.load(model_name, weights)
    return model


##########################################################
# Function that runs the OD model on a dataset of images #
# In addition, runs different attacks on these images to #
# produce Adversarial examples on the OD model           #
##########################################################
def main():
    print(f'Running experiment at {time.ctime()} on {socket.gethostname()}')
    config = configparser.ConfigParser()
    config_file_path = 'config.txt'
    config.read(filenames=config_file_path)

    # Loading a pretrained OD Model
    model = load_model('ultralytics/yolov5', 'yolov5s')

    #################################################################################
    # TODO: CHANGE THIS FOR LOOP TO RUN ON EVERY IMAGE IN A GIVEN DATASET OF IMAGES #
    #################################################################################
    path = config['DATASET']['relative_path']
    fns = None
    is_single_image = os.path.isdir(path)
    if is_single_image:
        fns = [i for i in os.listdir(path)]
    for new_idx, fn in enumerate(fns):
        # arguments.Config["bab"]["timeout"] = orig_timeout
        print('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx:', new_idx, 'img ID:', fn,
              '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        torch.cuda.empty_cache()
        gc.collect()

        # Image (Via URL)
        z = 'https://ultralytics.com/images/zidane.jpg'

        ## Image (Via Full Path)
        # z = "/home/avraham/alpha-beta-CROWN/complete_verifier/images/Airplane/airplane1.jpg"
        # z = "/home/avraham/alpha-beta-CROWN/complete_verifier/images/Horse/horse4_1.jpg"
        z = "images\\Truck\\truck.jpg"
        # z = config['DATASET']['image_path']
        if is_single_image:
            z = path
        else:
            z = os.path.join(path, fn)
        imgPath = z

        results = model(z)
        z = torch.tensor(results.imgs[0])  # Getting back the img to attack
        x = z.unsqueeze(0).to(dtype=torch.get_default_dtype(),
                              device=config['GENERAL']['device'])  # Adding another dimension
        x = x.permute(0, 3, 1, 2)  # Permuting img to be in shape: (1,3,780,1280) for Zidane
        x = x[:, :, :640, :1280]  # Recreating shape: (1,3,640,1280)

        # logit_pred = model_ori(x)[0]
        logit_pred = results.pred
        y_pred = 4  # For Zidane img, we have 4 objects; 2 persons, 2 tie
        y = results.pred[0].shape[0]

        '''  
        x, y = X[fn], int(labels[fn].item())
        x = x.unsqueeze(0).to(dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])

        # first check the model is correct at the input
        logit_pred = model_ori(x)[0]
        # y_pred = torch.max(logit_pred, 0)[1].item()
        '''

        '''
        ### FOR PLOTTING IMAGES IN MATPLOTLIB
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        '''

        attack_dataset = None
        data_min = None
        data_max = None
        perturb_eps = None
        attack_args = {'dataset': attack_dataset, 'model': model, 'x': x, 'max_eps': perturb_eps,
                       'data_min': data_min, 'data_max': data_max, 'y': y, 'results': results,
                       'imgPath': imgPath}
        attack_ret, attack_images, attack_margin = pgd_attack(**attack_args)
        print("##################         DONE!!!        ###################")


if __name__ == "__main__":
    # config_args()
    main()
