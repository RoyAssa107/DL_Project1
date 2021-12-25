#########################################################################
## Copyright (C) 2022, Roy Assa <assa.roy107@gmail.com>                ##
##                     Avraham Raviv <avrahamsapir1@gmail.com>         ##
##                     Itav Nissim <itav.nissim@gmail.com>             ##
#########################################################################
import configparser
import os
import socket
import random
import time
import gc
import numpy as np

import numpy as np
import pandas as pd

import torch
from termcolor import colored

from attack_pgd import pgd_attack


# Function that loads a given model specified by model_name
def load_model(model_name='ultralytics/yolov5', weights='yolov5s'):
    model = torch.hub.load(model_name, weights)
    return model


# # Function that creates a new directory for a new class for the attack output if doesnt exist
# def createDir(path):
#     index = 1
#     max = 2
#     while max > 0:
#         path_dir_test = path + str(index)
#         try:
#             # Trying to create a new directory in attack_output for the specified coco class (from path)
#             os.mkdir(path_dir_test)
#             print(f"Successfully created directory: \"{path_dir_test}\"")
#             return path_dir_test
#         except:
#             print("ERROR!!!")
#         # except:
#         #     print(f"Folder in path: {path_dir_test} already exists!")
#         index += 1
#         max -= 1
#
#
# # Function that finds the class of a given img path
# # The function uses the img path name to determine the class
# # The list of classes is found in cocoms_class.py file
# def getClass(imgPath):
#     from cocoms_class import classes
#     lst_classes = list(classes.values())
#     img_class = [coco_class for coco_class in lst_classes if imgPath.__contains__(coco_class)]
#
#     # Checking if a given image with class specified in the img path is found in the coco80 dataset classes
#     if img_class != []:
#         return img_class[0].capitalize()  # Return the corresponding coco80 class
#
#     raise Exception(f"Invalid image with path: \"{imgPath}\" No class exists in coco dataset for this image!\n"
#                     f"Please remove this image from directory and specify a new image from the coco80 dataset!")


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
    starting_time = time.time()
    success_rate = 0

    # Loading a pretrained OD Model
    model = config['GENERAL']['model']
    if model == 'yolov5n':
        model = load_model('ultralytics/yolov5', 'yolov5s')

    path = config['DATASET']['relative_path']
    fns = None
    is_single_image = not os.path.isdir(path)
    if not is_single_image:
        fns = np.asarray([os.path.join(path, i) for i in os.listdir(path)])
        while np.any([os.path.isdir(fn) for fn in fns]):
            for fn in [i for i in fns if os.path.isdir(i)]:
                new_fns = np.asarray([os.path.join(fn, i) for i in os.listdir(fn)])
                fns = np.concatenate((fns, new_fns))
                fns = np.delete(fns, np.argwhere(fns == fn))
    for image_index, fn in enumerate(fns):
        # arguments.Config["bab"]["timeout"] = orig_timeout
        # print('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx:', image_index, 'img ID:', fn,
        #       '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        torch.cuda.empty_cache()
        gc.collect()

        # Image (Via URL)
        # z = 'https://ultralytics.com/images/zidane.jpg'
        z = fn  # Get current image file name
        imgPath = z

        ## Image (Via Full Path)
        # z = "/home/avraham/alpha-beta-CROWN/complete_verifier/images/Airplane/airplane1.jpg"
        # z = "/home/avraham/alpha-beta-CROWN/complete_verifier/images/Horse/horse4_1.jpg"
        # z = "images\\Truck\\truck.jpg"
        # z = config['DATASET']['image_path']

        results = model(z)  # Compute a feed-forward through the OD Net in order to get results of detection
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
        target = config['ATTACK']['target']  # Attack target (check config file for further details)
        max_iter = int(config['ATTACK']['max_iter'])  # Maximal number of iterations during the attack
        noise_algorithm = config['ATTACK']['noise_algorithm']  # Chosen_Noise_Attack/White_Noise_Attack/
        # Bounding_Box_Attack/ Bounding_Box_Center_Attack/ Bernoulli_Bounding_Box_Attack/ Canny_Bernoulli_Attack

        attack_args = {'dataset': attack_dataset, 'model': model, 'x': x, 'max_eps': perturb_eps,
                       'data_min': data_min, 'data_max': data_max, 'y': y, 'results': results,
                       'imgPath': imgPath, 'noise_algorithm': noise_algorithm, 'target': target,
                       'image_index': image_index + 1, 'max_iter': max_iter}
        success, iteration_num, attack_duration, message = pgd_attack(**attack_args)
        print(message)
        if success:
            success_rate += 1

    ending_time = time.time()
    success_rate = success_rate / (image_index + 1)
    summary = colored(f"""
    ########################################################
    ###                     SUMMARY:                     ###
    ### Attack Results On Dataset: \"{path}\":              
    ###     1. Target: {target}
    ###     2. Max iterations: {max_iter}   
    ###     3. Noise algorithm: \"{noise_algorithm}\"                               
    ###     4. Total number of images: {image_index}                                         
    ###     5. Attack success rate: {success_rate*100}%                                            
    ###     6. Attack duration: {round(ending_time - starting_time, 3)}s  
    ########################################################
    ""","green")
    print(summary)


if __name__ == "__main__":
    # config_args()
    main()
