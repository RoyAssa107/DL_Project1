###########
# Imports #
###########
import os
import cv2
import matplotlib
import matplotlib.patches as pac
import torch
import numpy as np
from termcolor import colored
import torchvision.transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn
from PIL import Image
from cocoms_class import *
import matplotlib.pyplot as plt
import time
from shapely.geometry import Polygon
import torchvision.ops.boxes as bops

# import gc

# Normalize image
transform = T.Compose([
    # T.Resize(480),
    # T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

np.random.seed(42)  # For reproducibility


###############################################################
# TODO: Added support in the attack_OOP.py file to OOP attack #
###############################################################

# Function that saves results image to a given base path && the corresponding output image name
def save_result_image(plt1, base_path, outputImageName, suffix='', model_name=None):
    if suffix != '':
        suffix = '_' + suffix
    newResult = base_path + outputImageName + '_' + model_name + suffix + ".jpg"
    plt1.savefig(newResult)


# Function that plots the results received from attack_pgd and saving result's image to directory
def plot_attacked_image_BOX(original_img, attacked_img, noise, base_path, outputImageName, pred_attack, pred_real,
                            classes=None, plot_result=False, save_result=True, model_name=None):
    # save the attacked image without the bbox
    plt.clf()
    plt.imshow(attacked_img)
    if save_result:
        save_result_image(plt, base_path, outputImageName, model_name=model_name,
                          suffix='')  # Save attack result image in directory

    # create figure
    rows = 3
    cols = 1

    # Plotting original image
    fig = plt.figure(figsize=(20, 20))
    fig.add_subplot(rows, cols, 1)
    plt.imshow(original_img)
    plotType = 'Original image'
    plt.title(plotType)
    for box in pred_real:
        # Create new bounding box over detection
        # x,y,w,h
        rect = pac.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', fill=False)
        plt.gca().add_patch(rect)
        # h = box[3] - box[1]

        plt.text(x=box[0] + 20, y=box[1], s=classes[int(box[-1])] + f", {round(box[-2].item(), 3)}", c='black',
                 backgroundcolor='g')
        # plt.text(x=box[0], y=box[1], s=classes[int(box[-1]) + 1])

    # Plotting attacked image
    fig.add_subplot(rows, cols, 2)
    plt.imshow(attacked_img)
    plotType = 'Perturbed image'
    plt.title(plotType)
    for box in pred_attack:
        # Create new bounding box over detection
        # x,y,w,h
        rect = pac.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', fill=False)
        plt.gca().add_patch(rect)
        # h = box[3] - box[1]
        plt.text(x=box[0] + 20, y=box[1], s=classes[int(box[-1])] + f", {round(box[-2].item(), 3)}", c='black',
                 backgroundcolor='g')

    fig.add_subplot(rows, cols, 3)

    # Checking if we wish to plot result of just save in directory
    if plot_result:
        plt.imshow(noise)
        plt.title("Noise image")

    if save_result:
        save_result_image(plt, base_path, outputImageName, suffix='_bbox',
                          model_name=model_name)  # Save attack result image in directory

    # Showing all subfigures in a single plot
    if plot_result:
        plt.show()

    time.sleep(0.1)


def attack_with_white_noise_strength(X, model, normalize, base_path, results, imgPath, imgClass, iteration_num):
    # Attempt to add specific noise (with specified strength)

    strength = 100 + iteration_num * 10
    outputImgName = "\\" + imgPath.split('.')[0].split('\\')[-1]  # Get pure name of image
    # (without path and format of image)
    outputImgName += "_" + str(strength) + "_White_Noise"

    white_random_noise = np.random.random(X.shape) * strength
    delta1 = torch.from_numpy(white_random_noise)  # White noise (torch)
    inputs1 = normalize(X + delta1)  # version1 (original version)

    # inputs1 = X + delta1
    # inputs1 = inputs1 % 255  # Plotting results
    inputs1 = normalize(X + delta1)
    original_img = X[0].permute(1, 2, 0)
    attacked_img = inputs1[0].permute(1, 2, 0)
    noise1 = attacked_img - original_img

    # Rescale noise pixels
    if delta1.min() < 0:
        # normalize noise to fit inside range [0,255]
        noise1 = ((delta1[0].permute(1, 2, 0) + abs(delta1.min())) / abs(delta1.min())) * 255
    else:
        # noise1 = ((delta1[0].permute(1,2,0))/delta1.max())*255
        noise1 = (delta1[0].permute(1, 2, 0)) / 256

    original_img1 = original_img / 255
    attacked_img1 = attacked_img / 255
    output = model(np.asarray(inputs1[0].permute(1, 2, 0)))  # Run the model on the noised image

    return original_img1, attacked_img1, noise1, base_path, outputImgName, output  # return results for further handle


def attack_on_bounding_box(X, model, normalize, base_path, results, imgPath, imgClass, iteration_num):
    strength = 10 + 10 * iteration_num  # Noising strength (0-minimal, 255-maximal)

    outputImgName = "\\" + imgPath.split('.')[0].split('\\')[-1]  # Get pure name of image
    # (without path and format of image)
    outputImgName += "_" + str(strength) + "_Bounding_box_Noise"

    # Extract preds of original image ran on OD Neural Net
    result_preds = results.pred[0]  # [x_min,y_min,x_max,y_max,prob,c]

    # Extract width and height of all bounding boxes in the image
    w, h, x, y = [], [], [], []

    for box in result_preds:
        w.append(box[2] - box[0])
        h.append(box[3] - box[1])
        x.append(box[0])
        y.append(box[1])

    new_preds = [[x[i], y[i], w[i], h[i], result_preds[i, 4], result_preds[i, 5]] for i in
                 range(len(w))]  # [x_min,y_min,w,h,prob,c]

    # bounding_box_noise = np.random.random((1, 3, 640, 1280)) * (strength)
    bounding_box_noise = np.zeros(X.shape)

    # Adding targeted noise *inside* each bounding box
    for box in result_preds:
        x_min = int(box[0].item())
        x_max = int(box[2].item())
        y_min = int(box[1].item())
        y_max = int(box[3].item())
        bounding_box_noise[0, :, y_min:min(y_max, 640), x_min:min(x_max, 1280)] += np.random.random(
            (3, min(y_max, 640) - y_min, min(x_max, 1280) - x_min)) * strength

    delta1 = torch.from_numpy(bounding_box_noise)  # White noise (torch)

    # inputs1 = X + delta1
    # inputs1 = inputs1 % 255  # Plotting results
    inputs1 = normalize(X + delta1)
    original_img = X[0].permute(1, 2, 0)
    attacked_img = inputs1[0].permute(1, 2, 0)
    noise1 = attacked_img - original_img

    # Rescale noise pixels
    if delta1.min() < 0:
        # normalize noise to fit inside range [0,255]
        noise1 = ((delta1[0].permute(1, 2, 0) + abs(delta1.min())) / abs(delta1.min())) * 255
    else:
        # noise1 = ((delta1[0].permute(1,2,0))/delta1.max())*255
        noise1 = (delta1[0].permute(1, 2, 0)) / 256

    original_img1 = original_img / 255
    attacked_img1 = attacked_img / 255
    output = model(np.asarray(inputs1[0].permute(1, 2, 0)))  # Run the model on the noised image

    return original_img1, attacked_img1, noise1, base_path, outputImgName, output  # return results for further handle


# Function that creates a new directory for a new class for the attack output if doesnt exist
# In addition, it creates a new Test directory for current attack session
def createDir(path):
    # Trying to create a new directory in attack_output for the specified coco class (from path)
    os.makedirs(path, exist_ok=True)

    index = 1
    path_dir_test = path + "\\Test" + str(index)
    while os.path.exists(path_dir_test):
        index += 1
        path_dir_test = path + "\\Test" + str(index)
    os.mkdir(path_dir_test)
    return path_dir_test


# Function that finds the class of a given img path
# The function uses the img path name to determine the class
# The list of classes is found in cocoms_class.py file
def getClass(imgPath, classes=None):
    lst_classes = list(classes.values())
    img_class = [coco_class for coco_class in lst_classes if imgPath.__contains__(coco_class)]

    # Checking if a given image with class specified in the img path is found in the coco80 dataset classes
    if img_class != []:
        return img_class[0].capitalize()  # Return the corresponding coco80 class

    raise Exception(f"Invalid image with path: \"{imgPath}\" No class exists in coco dataset for this image!\n"
                    f"Please remove this image from directory and specify a new image from the coco80 dataset!")


# Function that adds noise to pixels inside each bounding box with Bernoulli distribution
# On each pixel inside the bounding boxes, we draw a Bernoulli random variable with probability *p* to be noised
def attack_on_bounding_box_Bernoulli(X, model, normalize, base_path, results, imgPath, imgClass, iteration_num):
    ############################################################################################
    # # Attempt to add specific noise (with specified strength)
    # imgClass = getClass(imgPath)
    # base_path += "Bernoulli_Bounding_Box_Attack\\" + imgClass + "\\Test"
    #
    # base_path = createDir(base_path)  # Create directory with specified Class if doesnt exist and return it's path

    strength = 255  # Noising strength (0-minimal, 255-maximal)
    p_noised = 0.01 * iteration_num  # Probability for each pixel to be noised with noise drawn from uniform
    # distribution of specified *strength* and current attack iteration

    # outputImgName = "\\" + imgClass + "_all"  # Name of the image to save in attack directory
    # outputImgName += "_" + str(strength) + "_Bernoulli_" + str(p_noised) + "_Bounding_box_Noise"
    outputImgName = "\\" + imgPath.split('.')[0].split('\\')[
        -1]  # Get pure name of image (without path and format of image)
    outputImgName += "_" + str(strength) + "_Bernoulli_" + str(p_noised) + "_Bounding_box_Noise"

    # Extract preds of original image ran on OD Neural Net
    result_preds = results.pred[0]  # [x_min,y_min,x_max,y_max,prob,c]

    # Extract width and height of all bounding boxes in the image
    w, h, x, y = [], [], [], []

    for box in result_preds:
        w.append(box[2] - box[0])
        h.append(box[3] - box[1])
        x.append(box[0])
        y.append(box[1])

    new_preds = [[x[i], y[i], w[i], h[i], result_preds[i, 4], result_preds[i, 5]] for i in
                 range(len(w))]  # [x_min,y_min,w,h,prob,c]

    # bounding_box_noise = np.random.random((1, 3, 640, 1280)) * (strength)
    bounding_box_noise = np.zeros(X.shape)

    # Adding targeted noise *inside* each bounding box
    for box in result_preds:
        x_min = int(box[0].item())
        x_max = int(box[2].item())
        y_min = int(box[1].item())
        y_max = int(box[3].item())
        # bounding_box_noise[0, :, y_min:min(y_max,640), x_min:min(x_max,1280)] +=
        # np.random.random((3, min(y_max,640)-y_min, min(x_max, 1280)-x_min))*strength

        bounding_box_shape = (3, min(y_max, 640) - y_min, min(x_max, 1280) - x_min)
        samples = np.random.binomial(size=bounding_box_shape, n=1,
                                     p=p_noised)  # Creating a (bounding box shape) matrix of 0/1 drawn from
        # Bernoulli distribution
        noise_pixels = np.random.random((3, min(y_max, 640) - y_min, min(x_max, 1280) - x_min)) * strength
        bounding_box_noise[0, :, y_min:min(y_max, 640),
        x_min:min(x_max, 1280)] += noise_pixels * samples  # Adding noise to specific pixels inside each bounding box
        # that the Bernoulli distribution returned 1

    delta1 = torch.from_numpy(bounding_box_noise)  # White noise (torch)

    inputs1 = X + delta1
    # inputs1 = inputs1 % 255  # Plotting results
    # inputs1 = normalize(X + delta1)
    original_img = X[0].permute(1, 2, 0)
    attacked_img = inputs1[0].permute(1, 2, 0)
    noise1 = attacked_img - original_img

    # Rescale noise pixels
    if delta1.min() < 0:
        # normalize noise to fit inside range [0,255]
        noise1 = ((delta1[0].permute(1, 2, 0) + abs(delta1.min())) / abs(delta1.min())) * 255
    else:
        # noise1 = ((delta1[0].permute(1,2,0))/delta1.max())*255
        noise1 = (delta1[0].permute(1, 2, 0)) / 256

    original_img1 = original_img / 255
    attacked_img1 = attacked_img / 255
    output = model(np.asarray(inputs1[0].permute(1, 2, 0)))  # Run the model on the noised image

    # plot_attacked_image_BOX(original_img1, attacked_img1, noise1, base_path, outputImgName, output.pred[0],
    #                         results.pred[0], plot_result=False, save_result=True)  # plot & save to output file

    return original_img1, attacked_img1, noise1, base_path, outputImgName, output  # return results for further handle


# Function that adds noise to pixels inside each bounding box with Bernoulli distribution
# On each pixel inside the bounding boxes, we draw a Bernoulli random variable with probability *p* to be noised
def attack_with_canny_and_Bernoulli(X, model, normalize, base_path, results, imgPath, imgClass, iteration_num):
    ############################################################################################
    # Attempt to add specific noise (with specified strength)

    strength = 255  # Noising strength (0-minimal, 255-maximal)
    p_noised = 0.01 * iteration_num  # Probability for each pixel to be noised with noise drawn from uniform
    # distribution of specified *strength* and current attack iteration

    outputImgName = "\\" + imgPath.split('.')[0].split('\\')[
        -1]  # Get pure name of image (without path and format of image)
    outputImgName += "_" + str(strength) + "_Bernoulli_" + str(p_noised) + "_Bounding_box_Noise"

    #########################################################################
    ### Bernoulli attack using Canny algorithm to extract edges of objects ###
    #########################################################################
    X1 = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)[:640, :1280, :]  # Read the image with maximum size (640,1280,3)
    # X1 = X[0].permute(1,2,0).numpy() # shape: (640,1280,3) --> this is required for the Canny algorithm
    canny_output = torch.tensor(cv2.Canny(X1, 100, 200))

    img_size = X1.shape
    noise = torch.tensor((np.random.random(size=img_size)) * strength)  # (640,1280,3)
    # noise = torch.tensor(np.zeros(img_size) * strength)  # (640,1280,3)  ### REMOVE FOR DEBUG!

    # entry==0 <=> no edge pixel, entry==1 <=> edge pixel
    canny_output_3_dim = torch.tensor(canny_output).repeat(3, 1, 1).permute(1, 2, 0) / 255
    Bernoulli_samples = np.random.binomial(size=canny_output_3_dim.shape, n=1,
                                           p=p_noised)  # Creating a (bounding box shape) matrix of 0/1 drawn from
    # Bernoulli distribution
    noised_edges = noise * canny_output_3_dim * Bernoulli_samples

    new_noised_image = X
    new_noised_image[0] += noised_edges.permute(2, 0, 1)
    inputs1 = new_noised_image

    delta1 = torch.from_numpy(noised_edges)  # noise (torch)

    original_img = X[0].permute(1, 2, 0)
    attacked_img = inputs1[0].permute(1, 2, 0)
    noise = attacked_img - original_img

    # Rescale noise pixels
    if delta1.min() < 0:
        noise1 = ((delta1[0].permute(1, 2, 0) + abs(delta1.min())) / abs(
            delta1.min())) * 255  # normalize noise to fit inside range [0,255]
    else:
        # noise1 = ((delta1[0].permute(1,2,0))/delta1.max())*255
        noise1 = (delta1[0].permute(1, 2, 0)) / 256

    original_img1 = original_img / 255
    attacked_img1 = attacked_img / 255
    output = model(np.asarray(inputs1[0].permute(1, 2, 0)))  # Run the model on the noised image

    noise1 = noised_edges / 255  #############################CHANGED ACCORDING TO CANNY!!!
    X2 = cv2.cvtColor(X1, cv2.COLOR_BGR2RGB)
    # plot_attacked_image_BOX(X2, attacked_img1, noise1, base_path, outputImgName, output.pred[0],
    #                         results.pred[0])  # plotting results and save them to output file
    # print()

    return original_img1, attacked_img1, noise1, base_path, outputImgName, output  # return results for further handle


def attack_on_bounding_box_center(X, model, normalize, base_path, results, imgPath, imgClass, iteration_num):
    ############################################################################################
    # Attempt to add specific noise (with specified strength)
    strength = 150  # Noising strength (0-minimal, 255-maximal)
    r = 50 + 10 * iteration_num
    outputImgName = "\\" + imgPath.split('.')[0].split('\\')[-1]  # Get pure name of image
    # (without path and format of image)
    outputImgName += "_" + str(strength) + "_" + str(r) + "_Bounding_box_Noise_Center"

    # Extract preds of original image ran on OD Neural Net
    result_preds = results.pred[0]  # [x_min,y_min,x_max,y_max,prob,c]

    # Extract width and height of all bounding boxes in the image and extract centers of bounding boxes
    center = []  # Each center is in the following format: ((x+w)/2,(y+h)/2)
    w_hat, h_hat, x_hat, y_hat = [], [], [], []

    for box in result_preds:
        w = int(box[2] - box[0])
        h = int(box[3] - box[1])
        x_min = int(box[0])
        y_min = int(box[1])

        # Center of a given bounding box
        cx = (x_min + w / 2)
        cy = (y_min + h / 2)
        center.append((cx, cy))

        # New bounding box, with specified radius r and center [x_hat,y_hat,w_hat,h_hat,prob,c]
        # I've produced the new centered bounding box, so it doesn't exceed the real bounding box (with min,max)
        x_hat.append(max(cx - r, x_min))
        y_hat.append(max(cy - r, y_min))
        w_hat.append(min(2 * r, 2 * (cx - x_min)))
        h_hat.append(min(2 * r, 2 * (cy - y_min)))
        # h_hat.append(4*r)

    new_preds = [[x_hat[i], y_hat[i], w_hat[i], h_hat[i], round(result_preds[i, 4].item(), 3),
                  round(result_preds[i, 5].item(), 3)] for i in range(result_preds.shape[0])]
    # [x_hat,y_hat,w_hat,h_hat,prob,c]

    # bounding_box_noise = np.random.random((1, 3, 640, 1280)) * (strength)
    bounding_box_noise = np.zeros(X.shape)

    # Adding targeted noise *inside* each bounding box center with specified radius size and center
    for box in new_preds:
        x_min = int(box[0])
        w = int(box[2])
        y_min = int(box[1])
        h = int(box[3])
        bounding_box_noise[0, :, y_min:min(y_min + h, 640), x_min:min(x_min + w, 1280)] += np.random.random(
            (3, max(min(y_min + h, 640) - y_min, 0), min(x_min + w, 1280) - x_min)) * strength

    delta1 = torch.from_numpy(bounding_box_noise)  # White noise (torch)

    inputs1 = X + delta1
    # inputs1 = inputs1 % 255  # Plotting results
    # inputs1 = normalize(X + delta1)
    original_img = X[0].permute(1, 2, 0)
    attacked_img = inputs1[0].permute(1, 2, 0)
    noise1 = attacked_img - original_img

    # Rescale noise pixels
    if delta1.min() < 0:
        # normalize noise to fit inside range [0,255]
        noise1 = ((delta1[0].permute(1, 2, 0) + abs(delta1.min())) / abs(delta1.min())) * 255
    else:
        # noise1 = ((delta1[0].permute(1,2,0))/delta1.max())*255
        noise1 = (delta1[0].permute(1, 2, 0)) / 256

    original_img1 = original_img / 255
    attacked_img1 = attacked_img / 255

    output = model(np.asarray(inputs1[0].permute(1, 2, 0)))  # Run the model on the noised image
    matplotlib.use('TkAgg')
    # plot_attacked_image_BOX(original_img1, attacked_img1, noise1, base_path, outputImgName, output.pred[0],
    #                         results.pred[0], plot_result=True, save_result=False, classes=classes_80)  # plot & save to output file
    # print()
    return original_img1, attacked_img1, noise1, base_path, outputImgName, output  # return results for further handle


def attack_with_chosen_noise_strength(X, model, normalize, base_path, results, imgPath, imgClass, iteration_num):
    ############################################################################################
    # Attempt to add specific noise (with specified strength)
    strength = 150  # Noising strength (0-minimal, 255-maximal)
    r = 50 + 10 * iteration_num
    outputImgName = "\\" + imgPath.split('.')[0].split('\\')[-1]  # Get pure name of image
    # (without path and format of image)
    outputImgName += "_" + str(strength) + "_Chosen_Noise"

    chosen_noise_strength = np.random.random(size=X.shape) * strength
    delta1 = torch.from_numpy(chosen_noise_strength)  # White noise (torch)

    # Plotting results
    inputs1 = normalize(X + delta1)
    original_img = X[0].permute(1, 2, 0)
    attacked_img = inputs1[0].permute(1, 2, 0)
    noise1 = attacked_img - original_img

    # Rescale noise pixels
    if delta1.min() < 0:
        # normalize noise to fit inside range [0,255]
        noise1 = ((delta1[0].permute(1, 2, 0) + abs(delta1.min())) / abs(delta1.min())) * 255
    else:
        # noise1 = ((delta1[0].permute(1,2,0))/delta1.max())*255
        noise1 = (delta1[0].permute(1, 2, 0)) / 256

    original_img1 = original_img / 255
    attacked_img1 = attacked_img / 255

    output = model(np.asarray(inputs1[0].permute(1, 2, 0)))  # Run the model on the noised image
    matplotlib.use('TkAgg')
    # plot_attacked_image_BOX(original_img1, attacked_img1, noise1, base_path, outputImgName, output.pred[0],
    #                         results.pred[0], plot_result=True, save_result=False, classes=classes_80)  # plot & save to output file
    # print()
    return original_img1, attacked_img1, noise1, base_path, outputImgName, output  # return results for further handle


# function that calculates IOU of two bounding boxes in the following format:
# OPTION1:
#  box: [[x_upper_left, y_upper_left],
#        [x_upper_right, y_upper_right],
#        [x_lower_right, y_lower_right],
#        [x_lower_left, y_lower_right]]
# OPTION2:
#  box: [x_min,y_min,x_max,y_max]
def calculate_iou(box1, box2):
    ###########
    # OPTION1 #
    ###########
    # box_1 = [[511, 41], [577, 41], [577, 76], [511, 76]]
    # box_2 = [[544, 59], [610, 59], [610, 94], [544, 94]]
    # poly_1 = Polygon(box1)
    # poly_2 = Polygon(box2)
    # iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    # return iou

    ###########
    # OPTION2 #
    ###########
    # box1 = torch.tensor([[511, 41, 577, 76]], dtype=torch.float)
    # box2 = torch.tensor([[544, 59, 610, 94]], dtype=torch.float)
    iou = bops.box_iou(box1, box2)
    return iou


############################################################
# Function that gets the model results and attack results  #
# Returns True <=> IOU attack was accomplished             #
############################################################
def check_IOU_attack(original_preds, attack_preds, IOU_threshold, classes):
    IOU_succ_attack_count = 0
    for i in range(min(original_preds.shape[0], attack_preds.shape[0])):
        bbox_original = torch.tensor([original_preds[i][:4].tolist()])  # [x_min, y_min, x_max, y_max]
        bbox_attack = torch.tensor([attack_preds[i][:4].tolist()])  # [x_min, y_min, x_max, y_max]

        ###################################################
        # TODO: Add option for several/all bounding boxes #
        ###################################################
        # check the attack achieved the specified IOU threshold at *ANY* bounding box
        current_bbox_ratio = calculate_iou(bbox_original, bbox_attack)
        print(f"#{i} bbox IOU ratio: {current_bbox_ratio}")
        if current_bbox_ratio <= IOU_threshold:
            print(
                colored(f"In bbox #{i} IOU attack succeeded!!! in class:"
                        f"{classes[int(original_preds[i][-1].item())]} ", "green"))
            IOU_succ_attack_count += 1

    # return false if *all* bounding boxes don't have less IOU ratio than the specified IOU_threshold
    return IOU_succ_attack_count, IOU_succ_attack_count != 0


#######################################################################
# Function that checks if original and attack preds' detection agree. #
# Returns True <=> detection is False Positive                        #
#######################################################################
def check_bbox_FP(original_pred, attack_pred):
    return original_pred[-1] != attack_pred[-1]


############################################################
# Function that gets the model results and attack results  #
# Returns True <=> IOU attack was accomplished             #
############################################################
def check_False_Positive_attack(original_preds, attack_preds, classes):
    FP_succ_attack_count = 1

    for i in range(min(original_preds.shape[0], attack_preds.shape[0])):
        # check the attack achieved the specified IOU threshold at *ANY* bounding box
        flag_bbox_FP = check_bbox_FP(original_preds[i], attack_preds[i])
        print(f"#{i} bbox FP: {flag_bbox_FP}")
        if flag_bbox_FP:
            print(colored(f"In bbox #{i} found False Positive detection:\n"
                          f"Original detection: {classes[int(original_preds[i][-1].item())]}\n"
                          f"Attack detection: {classes[int(attack_preds[i][-1].item())]}\n", "green"))
            FP_succ_attack_count += 1

    # return false if *all* bounding boxes detections are correct
    return FP_succ_attack_count, FP_succ_attack_count != 0


##############################################################################################
# Function that gets the model results, attack results and a specified target for the attack #
# Returns True <=> the target attack was accomplished                                        #
##############################################################################################
def check_attack_output(target, amount, model_results, output, classes):
    original_preds = model_results.pred[0]
    attack_preds = output.pred[0]
    original_pred_num = original_preds.shape[0]
    attack_pred_num = attack_preds.shape[0]

    if target == 'Missing Detection':
        # If the "amount" from config file is larger than the number of detections in the original image
        # this means that we would like to miss all detections in the output image
        missing_detection_count = min(model_results.pred[0].shape[0], amount)
        return original_pred_num - attack_pred_num, original_pred_num - attack_pred_num >= missing_detection_count

    elif target == 'IOU':
        ###################
        # TODO: Check IOU #
        ###################
        IOU_threshold = amount
        return check_IOU_attack(original_preds, attack_preds, IOU_threshold, classes=classes)
        pass

    elif target == 'False Positive':
        ########################################
        # TODO: Check false positive via preds #
        ########################################
        return check_False_Positive_attack(original_preds, attack_preds, classes=classes)


####################################################
# Function that returns the latest Test directory  #
####################################################
def getTestDir(base_path):
    return base_path + "\\" + os.listdir(base_path)[-1]


###################################################################
# Function that returns a function object of the specified attack #
###################################################################
def get_attack_function(noise_algorithm):
    if noise_algorithm == "Chosen_Noise_Attack":
        return attack_with_chosen_noise_strength
    elif noise_algorithm == "White_Noise_Attack":
        return attack_with_white_noise_strength
    elif noise_algorithm == "Bounding_Box_Attack":
        return attack_on_bounding_box
    elif noise_algorithm == "Bounding_Box_Center_Attack":
        return attack_on_bounding_box_center
    elif noise_algorithm == "Bernoulli_Bounding_Box_Attack":
        return attack_on_bounding_box_Bernoulli
    elif noise_algorithm == "Canny_Bernoulli_Attack":
        return attack_with_canny_and_Bernoulli


######################################################################################
# Function that gets kwargs with all information about model,attack outputs and more #
# Returns a specific message according to specified targeted attack                  #
######################################################################################
def get_message(**msg_args):
    if msg_args["target"] == 'Missing Detection':
        message = f"""
        ################################################################
        ### Attack Results On Image: \"{msg_args["imgPath"]}\":                                              
        ###    1. Success: {msg_args["success_color"]}   
        ###    2. Noise algorithm: {msg_args["noise_algorithm"]}
        ###    3. Attack target: {msg_args["target"]}
        ###    4. Missing detection attack success rate: {msg_args["attack_pred"].shape[0] / msg_args["original_pred"].shape[0]}                                                                                            
        ###    4. Image number in directory: {msg_args["image_index"]}                                    
        ###    5. Total number of iterations: {msg_args["iteration_num"]}                                    
        ###    6. Total number of detections in original image: {msg_args["original_pred"].shape[0]}       
        ###    7. Total number of detections in attacked image: {msg_args["attack_pred"].shape[0]} 
        ###    8. Total duration: {round(msg_args["ending_time"] - msg_args["starting_time"], 3)}s                       
        ################################################################
        """
    elif msg_args["target"] == 'IOU':
        message = f"""
        ################################################################
        ### Attack Results On Image: \"{msg_args["imgPath"]}\":                                              
        ###    1. Success: {msg_args["success_color"]}   
        ###    2. Noise algorithm: {msg_args["noise_algorithm"]}
        ###    3. Attack target: {msg_args["target"]}         
        ###    3. IOU threshold: {msg_args["amount"]}      
        ###    4. IOU attack success rate: {100 * msg_args["num_FP_or_IOU_Misses"] / msg_args["attack_pred"].shape[0]}%                                           
        ###    5. Image number in directory: {msg_args["image_index"]}                                    
        ###    6. Total number of iterations: {msg_args["iteration_num"]}                                    
        ###    7. Total duration: {round(msg_args["ending_time"] - msg_args["starting_time"], 3)}s                       
        ################################################################
        """

    elif msg_args["target"] == 'False Positive':
        message = f"""
        ################################################################
        ### Attack Results On Image: \"{msg_args["imgPath"]}\":                                              
        ###    1. Success: {msg_args["success_color"]}   
        ###    2. Noise algorithm: {msg_args["noise_algorithm"]}
        ###    3. Attack target: {msg_args["target"]}
        ###    4. False-Positive attack success rate: {100 * msg_args["num_FP_or_IOU_Misses"] / msg_args["attack_pred"].shape[0]}%                                                 
        ###    4. Image number in directory: {msg_args["image_index"]}                                    
        ###    5. Total number of iterations: {msg_args["iteration_num"]}                                    
        ###    6. Total duration: {round(msg_args["ending_time"] - msg_args["starting_time"], 3)}s                       
        ################################################################
        """
    return message


##################################################################################
# Function that runs the attack in iterations on a specific image specified by X #
##################################################################################
def main_attack(model, X, y, epsilon, alpha, num_restarts, max_attack_iter=10,
                multi_targeted=True, num_classes=10, use_adam=True,
                lower_limit=0.0, upper_limit=1.0, normalize=lambda x: x,
                initialization='uniform', results=None, imgPath=None, noise_algorithm=None,
                target='Missing at least one', image_index=1, amount=1):
    starting_time = time.time()  # Monitor attack time
    success = False  # check if attack succeeded after max_attack_iter (at most)
    success_color = None
    model_results = results

    #######################################################################
    # TODO: change classes_80 according to specified model (yolo/DETR...) #
    #######################################################################
    classes = classes_80
    imgClass = getClass(imgPath, classes)  # holds image class according to file path specified.
    # Searching class name in file path and checking if it
    # is one of the 80 classes in the coco80 dataset
    base_path = os.getcwd() + "\\Attack_Output\\" + noise_algorithm + "\\" + imgClass

    if image_index == 1:
        base_path = createDir(base_path)  # Create directory with specified Class if doesnt exist and return it's path
    else:
        base_path = getTestDir(base_path)  # Get latest Test directory to save results of latest image to same directory

    # Running several iterations on a given image specified by X.
    # In each iteration, we strengthen the attack if we haven't managed to produce
    # the wanted attack specified by "target" in the config.txt file
    iteration_num = 0
    for iteration_num in range(1, max_attack_iter + 1, 1):
        strength = 255
        # delta = np.random.random(size=X.shape) * strength
        delta = np.zeros(shape=X.shape) * strength
        inputs = normalize(X + delta)
        with torch.no_grad():
            # Getting the attack specified by noise_algorithm
            chosen_attack = get_attack_function(noise_algorithm)

            # running the attack with specified noise algorithm
            original_img, attacked_img, noise, base_path, outputImgName, attack_output = \
                chosen_attack(X, model, normalize, base_path, results, imgPath, imgClass, iteration_num)

            # checking if attack succeeded
            num_FP_or_IOU_Misses, flag_successful_attack = \
                check_attack_output(target, amount, model_results, attack_output, classes=classes)

            if flag_successful_attack:
                # Attack succeeded; Plotting & Saving results to directory
                plot_attacked_image_BOX(original_img, attacked_img, noise, base_path, outputImgName,
                                        attack_output.pred[0],
                                        # [Dx6] -> D is number of detections, 6 is [xmin, ymin, xmax, ymax, p, c]
                                        results.pred[0], classes=classes, plot_result=True, save_result=True,
                                        model_name=model._get_name())
                success = True
                break
            ####################################################################################################

    # If attack succeeded, the success message is green
    if success:
        success_color = colored(success, 'green')
    else:
        # If attack succeeded, the success message is red
        success_color = colored(success, 'red')

    ending_time = time.time()

    msg_args = {"imgPath": imgPath, "success_color": success_color,
                "noise_algorithm": noise_algorithm, "target": target,
                "image_index": image_index, "iteration_num": iteration_num,
                "original_pred": results.pred[0], "attack_pred": attack_output.pred[0],
                "starting_time": starting_time, "ending_time": ending_time,
                "amount": amount, "num_FP_or_IOU_Misses": num_FP_or_IOU_Misses}
    message = get_message(**msg_args)  # Getting result of attack message
    # message = f"""
    # ############################################################
    # ### Attack Results On Image: \"{imgPath}\":
    # ###    1. Success: {success_color}
    # ###    2. Noise algorithm: {noise_algorithm}
    # ###    3. Attack target: {target}
    # ###    4. Image number in directory: {image_index}
    # ###    5. Total number of iterations: {iteration_num}
    # ###    6. Total number of detections in original image: {results.pred[0].shape[0]}
    # ###    7. Total number of detections in attacked image: {attack_output.pred[0].shape[0]}
    # ###    8. Total duration: {round(ending_time - starting_time, 3)}s
    # ###########################################################
    # """

    return_attack_args = {'original_img': original_img, 'attacked_img': attacked_img, 'noise': noise,
                          'base_path': base_path, 'outputImgName': outputImgName, 'attack_output': attack_output,
                          'success': success, 'iteration_num': iteration_num, 'time': time, 'results': results,
                          'output_img_path': base_path + outputImgName + '_' + model._get_name() + imgPath[-4:],
                          'message': message}
    return return_attack_args


#######################################################
# TODO: Get rid of this function!!! Not needed at all #
#######################################################
def od_attack(dataset, model, x, max_eps, data_min, data_max, y=None, initialization="uniform", results=None,
              imgPath=None, noise_algorithm=None, target='Missing one', image_index=1, max_iter=10, amount=1):
    return main_attack(model, X=x, y=torch.tensor([y], device=x.device),
                       epsilon=float("inf"),
                       max_attack_iter=max_iter,
                       num_restarts=1,
                       upper_limit=data_max, lower_limit=data_min,
                       initialization=initialization,
                       results=results, imgPath=imgPath, alpha=0.01,
                       num_classes=80, use_adam=False, multi_targeted=True,
                       noise_algorithm=noise_algorithm, target=target, image_index=image_index, amount=amount)


def verify_with_other_models(args=None):
    # load DETR OD model

    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    # org_im = Image.open(os.path.join(args['base_path'], os.listdir(args['base_path'])[0]))
    org_im = args['original_img']
    org_img = transform(org_im.permute(2, 0, 1))
    org_img = org_img.unsqueeze(0)

    outputs = model.float()(org_img.float())

    # org_img = transform(org_im).unsqueeze(0)
    # outputs = model(org_img)
    detr_org_results = post_process_detr(im=org_im, output=outputs)

    # im = Image.open(args['output_img_path'])

    im = args['attacked_img']
    img = transform(im.permute(2, 0, 1))
    img = img.unsqueeze(0)
    outputs = model.float()(img.float())
    detr_attack_results = post_process_detr(im=im, output=outputs)
    plot_attacked_image_BOX(args['original_img'], args['attacked_img'], args['noise'], args['base_path'],
                            args['outputImgName'], detr_attack_results, detr_org_results, classes=classes_90,
                            plot_result=True, save_result=True, model_name=model._get_name())

    # load RetinaNet OD model
    model = retinanet_resnet50_fpn(pretrained=True)
    model.eval()
    predictions = model(img)


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_h, img_w, _ = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def post_process_detr(im=None, output=None):
    # standard PyTorch mean-std input image normalization

    # keep only predictions with 0.7+ confidence
    probas = output['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.75

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(output['pred_boxes'][0, keep], im.size())

    ret_det = []
    for p, (xmin, ymin, xmax, ymax) in zip(probas[keep], bboxes_scaled.tolist()):
        cl_idx = p.argmax()  # get index of detection calss
        pr = p[cl_idx]  # probability of this detection
        ret_det.append([xmin, ymin, xmax, ymax, pr, cl_idx])

    return torch.Tensor(ret_det)
