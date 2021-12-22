###########
# Imports #
###########
import time
import os

import cv2
import matplotlib
import matplotlib.patches as pac
import torch
import numpy as np
from cocoms_class import classes
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducibility


########################################################################################################################
# Function that saves results image to a given base path && the corresponding output image name
def save_result_image(plt, base_path, outputImageName):
    counter = 1
    while (True):
        newResult = base_path + outputImageName + str(counter) + ".jpg"
        if not os.path.exists(newResult):
            plt.savefig(newResult)
            time.sleep(0.1)
            break
        else:
            counter += 1


# Function that plots the results received from attack_pgd and saving result's image to directory
def plot_attacked_image_BOX(original_img, attacked_img, noise, base_path, outputImageName, pred_attack, pred_real):
    # Plotting configurations
    import matplotlib
    matplotlib.use('TkAgg')

    # create figure
    rows = 3
    cols = 1
    fig = plt.figure(figsize=(20, 20))

    # Plotting original image
    fig.add_subplot(rows, cols, 1)
    plt.imshow(original_img)
    plotType = 'Original image'
    plt.title(plotType)
    for box in pred_real:
        # Create new bounding box over detection
        # rect = pac.Rectangle((9.55873e+02, 4.57898e+02), 7.27803e+01, 6.16653e+01, linewidth=1, edgecolor='r', fill=False)
        # x,y,w,h
        rect = pac.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', fill=False)
        plt.gca().add_patch(rect)
        h = box[3] - box[1]

        plt.text(x=box[0] + 20, y=box[1], s=classes[int(box[-1]) + 1] + f", {round(box[-2].item(), 3)}", c='black',
                 backgroundcolor='g')
        # plt.text(x=box[0], y=box[1], s=classes[int(box[-1]) + 1])

    # Plotting attacked image
    fig.add_subplot(rows, cols, 2)
    plt.imshow(attacked_img)
    plotType = 'Perturbed image'
    plt.title(plotType)
    for box in pred_attack:
        # Create new bounding box over detection
        # rect = pac.Rectangle((9.55873e+02, 4.57898e+02), 7.27803e+01, 6.16653e+01, linewidth=1, edgecolor='r', fill=False)
        # x,y,w,h
        rect = pac.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                             fill=False)
        plt.gca().add_patch(rect)
        h = box[3] - box[1]
        plt.text(x=box[0] + 20, y=box[1], s=classes[int(box[-1]) + 1] + f", {round(box[-2].item(), 3)}", c='black',
                 backgroundcolor='g')

    fig.add_subplot(rows, cols, 3)
    plt.imshow(noise)
    plt.title("Noise image")

    outputImageName = outputImageName + "_box"
    save_result_image(plt, base_path, outputImageName)  ###################SAVE IMAGE IN ATTACK DIRECTORY

    # Showing all subfigures in a single plot
    plt.show()


def attack_with_white_noise_strength(X, model, normalize, base_path, outputImgName, results):
    # Attempt to add specific noise (with specified strength)
    base_path = "/home/avraham/alpha-beta-CROWN/complete_verifier/Attack_Output/White_Noise_Attack/Truck/"
    strength = 150
    outputImgName = "truck_all"  # Name of the image to save in attack directory

    white_random_noise = np.random.random(X.shape) * strength
    delta1 = torch.from_numpy(white_random_noise)  # White noise (torch)
    inputs1 = normalize(X + delta1)  # version1 (original version)

    inputs1 = X + delta1  # version2
    # inputs1 = inputs1 % 255  # version2 (new version)
    original_img = X[0].permute(1, 2, 0)
    attacked_img = inputs1[0].permute(1, 2, 0)
    noise = attacked_img - original_img

    # Rescale pixels
    # noise1 = (delta1[0].permute(1, 2, 0) + 255) / 256 ###### UNCOMMENT IF YOU PUT strength < 0
    noise1 = (delta1[0].permute(1, 2, 0)) / 256  ###### UNCOMMENT IF YOU PUT strength > 0
    original_img1 = original_img / 255
    attacked_img1 = attacked_img / 255
    output = model(np.asarray(inputs1[0].permute(1, 2, 0)))  # Run the model on the white noised image

    outputImgName += "_" + str(strength) + "_WHITE_NOISE"
    plot_attacked_image_BOX(original_img1, attacked_img1, noise1, base_path, outputImgName, output.pred[0],
                            results.pred[0])  # plotting results and save them to output file
    print()


def attack_on_bounding_box(X, model, normalize, base_path, outputImgName, results):
    ############################################################################################
    # Attempt to add specific noise (with specified strength)
    base_path = "/home/avraham/alpha-beta-CROWN/complete_verifier/Attack_Output/Bounding_Box_Attack/Horse/"
    strength = 100
    outputImgName = "horse_all"  # Name of the image to save in attack directory
    outputImgName += "_" + str(strength) + "_Bounding_box_Noise"

    # Extract preds of original image ran on OD Neural Net
    result_preds = results.pred[0]  # [x_min,y_min,x_max,y_max,prob,c]

    # Extract width and height of all bounding boxes in the image
    w = []
    h = []
    x = []
    y = []
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
    noise = attacked_img - original_img

    # Rescale pixels
    noise1 = (delta1[0].permute(1, 2, 0) + 255) / 256  ###### UNCOMMENT IF YOU PUT strength < 0
    # noise1 = (delta1[0].permute(1, 2, 0)) / 256  ###### UNCOMMENT IF YOU PUT strength > 0
    original_img1 = original_img / 255
    attacked_img1 = attacked_img / 255
    output = model(np.asarray(inputs1[0].permute(1, 2, 0)))  # Run the model on the noised image

    plot_attacked_image_BOX(original_img1, attacked_img1, noise1, base_path, outputImgName, output.pred[0],
                            results.pred[0])  # plotting results and save them to output file
    print()


# Function that adds noise to pixels inside each bounding box with Bernoulli distribution
# On each pixel inside the bounding boxes, we draw a Bernoulli random variable with probability *p* to be noised
def attack_on_bounding_box_Bernoulli(X, model, normalize, base_path, outputImgName, results):
    ############################################################################################
    # Attempt to add specific noise (with specified strength)
    base_path = "/home/avraham/alpha-beta-CROWN/complete_verifier/Attack_Output/Bernoulli_Bounding_Box_Attack/Truck/"
    strength = 255  # Noising strength (0-minimal, 255-maximal)
    p_noised = 0.1  # Probability for each pixel to be noised with noise drawn from uniform distribution of specified *strength*
    outputImgName = "truck_all"  # Name of the image to save in attack directory
    outputImgName += "_" + str(strength) + "_Bernoulli_" + str(p_noised) + "_Bounding_box_Noise"

    # Extract preds of original image ran on OD Neural Net
    result_preds = results.pred[0]  # [x_min,y_min,x_max,y_max,prob,c]

    # Extract width and height of all bounding boxes in the image
    w = []
    h = []
    x = []
    y = []
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
        # bounding_box_noise[0, :, y_min:min(y_max,640), x_min:min(x_max,1280)] += np.random.random((3, min(y_max,640)-y_min, min(x_max, 1280)-x_min))*strength
        ##########################################
        bounding_box_shape = (3, min(y_max, 640) - y_min, min(x_max, 1280) - x_min)
        samples = np.random.binomial(size=bounding_box_shape, n=1,
                                     p=p_noised)  # Creating a (bounding box shape) matrix of 0/1 drawn from Bernoulli distribution
        noise_pixels = np.random.random((3, min(y_max, 640) - y_min, min(x_max, 1280) - x_min)) * strength
        bounding_box_noise[0, :, y_min:min(y_max, 640),
        x_min:min(x_max, 1280)] += noise_pixels * samples  # Adding noise to specific pixels inside each bounding box
        # that the Bernoulli distribution returned 1
        ##########################################

    delta1 = torch.from_numpy(bounding_box_noise)  # White noise (torch)

    inputs1 = X + delta1
    inputs1 = inputs1 % 255  # Plotting results
    # inputs1 = normalize(X + delta1)
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

    plot_attacked_image_BOX(original_img1, attacked_img1, noise1, base_path, outputImgName, output.pred[0],
                            results.pred[0])  # plotting results and save them to output file
    print()


# Function that adds noise to pixels inside each bounding box with Bernoulli distribution
# On each pixel inside the bounding boxes, we draw a Bernoulli random variable with probability *p* to be noised
def attack_with_canny_and_Bernoulli(X, model, normalize, base_path, outputImgName, results, imgPath):
    ############################################################################################
    # Attempt to add specific noise (with specified strength)

    ##################################################################
    # TODO: Take the path from the config file (via config variable) #
    ##################################################################
    base_path = "C:\\Users\\assar\\PycharmProjects\\pythonProject1\\DL_Project\\Attack_Output\\Canny_Bernoulli_Attack\\Truck\\"
    strength = 255  # Noising strength (0-minimal, 255-maximal)
    p_noised = 1  # Probability for each pixel to be noised with noise drawn from uniform distribution of specified *strength*
    outputImgName = "truck_all"  # Name of the image to save in attack directory
    outputImgName += "_" + str(strength) + "_Canny_Bernoulli_" + str(p_noised) + "_Noise"

    # # Extract preds of original image ran on OD Neural Net
    # result_preds = results.pred[0] # [x_min,y_min,x_max,y_max,prob,c]
    #
    # # Extract width and height of all bounding boxes in the image
    # w = []
    # h = []
    # x = []
    # y = []
    # for box in result_preds:
    #     w.append(box[2] - box[0])
    #     h.append(box[3] - box[1])
    #     x.append(box[0])
    #     y.append(box[1])
    #
    # new_preds = [[x[i], y[i], w[i], h[i], result_preds[i, 4], result_preds[i, 5]] for i in range(len(w))]  # [x_min,y_min,w,h,prob,c]
    #
    # # bounding_box_noise = np.random.random((1, 3, 640, 1280)) * (strength)
    # bounding_box_noise = np.zeros(X.shape)
    #
    # # Adding targeted noise *inside* each bounding box
    # for box in result_preds:
    #     x_min = int(box[0].item())
    #     x_max = int(box[2].item())
    #     y_min = int(box[1].item())
    #     y_max = int(box[3].item())
    #     # bounding_box_noise[0, :, y_min:min(y_max,640), x_min:min(x_max,1280)] += np.random.random((3, min(y_max,640)-y_min, min(x_max, 1280)-x_min))*strength
    #     bounding_box_shape = (3, min(y_max, 640) - y_min, min(x_max, 1280) - x_min)
    #     samples = np.random.binomial(size=bounding_box_shape, n=1, p=p_noised)  # Creating a (bounding box shape) matrix of 0/1 drawn from Bernoulli distribution
    #     noise_pixels = np.random.random((3, min(y_max, 640) - y_min, min(x_max, 1280) - x_min)) * strength
    #     bounding_box_noise[0, :, y_min:min(y_max,640), x_min:min(x_max,1280)] += noise_pixels * samples  # Adding noise to specific pixels inside each bounding box
    #                                                                                                      # that the Bernoulli distribution returned 1
    # delta1 = torch.from_numpy(bounding_box_noise)  # White noise (torch)

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
                                           p=p_noised)  # Creating a (bounding box shape) matrix of 0/1 drawn from Bernoulli distribution
    noised_edges = noise * canny_output_3_dim * Bernoulli_samples

    new_noised_image = X
    new_noised_image[0] += noised_edges.permute(2, 0, 1)
    inputs1 = new_noised_image
    #########################################################################

    # inputs1 = X + delta1
    # inputs1 = inputs1 % 255  # Plotting results
    ## inputs1 = normalize(X + delta1)
    original_img = X[0].permute(1, 2, 0)
    attacked_img = inputs1[0].permute(1, 2, 0)
    noise = attacked_img - original_img

    # # Rescale noise pixels
    # if delta1.min() < 0:
    #     noise1 = ((delta1[0].permute(1, 2, 0) + abs(delta1.min())) / abs(
    #         delta1.min())) * 255  # normalize noise to fit inside range [0,255]
    # else:
    #     # noise1 = ((delta1[0].permute(1,2,0))/delta1.max())*255
    #     noise1 = (delta1[0].permute(1, 2, 0)) / 256

    original_img1 = original_img / 255
    attacked_img1 = attacked_img / 255
    output = model(np.asarray(inputs1[0].permute(1, 2, 0)))  # Run the model on the noised image

    noise1 = noised_edges / 255  #############################CHANEGD ACCORDING TO CANNY!!!
    X2 = cv2.cvtColor(X1, cv2.COLOR_BGR2RGB)
    plot_attacked_image_BOX(X2, attacked_img1, noise1, base_path, outputImgName, output.pred[0],
                            results.pred[0])  # plotting results and save them to output file
    print()


def attack_on_bounding_box_center(X, model, normalize, base_path, outputImgName, results):
    ############################################################################################
    # Attempt to add specific noise (with specified strength)
    base_path = "/home/avraham/alpha-beta-CROWN/complete_verifier/Attack_Output/Bounding_Box_Center_Attack/Bird/"
    strength = 255
    r = 100
    outputImgName = "bird_all"  # Name of the image to save in attack directory
    outputImgName += "_" + str(strength) + "_Bounding_Box_Center_radius_" + str(r) + "_Noise"

    # Extract preds of original image ran on OD Neural Net
    result_preds = results.pred[0]  # [x_min,y_min,x_max,y_max,prob,c]

    # Extract width and height of all bounding boxes in the image and extract centers of bounding boxes
    center = []  # Each center is in the following format: ((x+w)/2,(y+h)/2)
    w_hat = []
    h_hat = []
    x_hat = []
    y_hat = []
    for box in result_preds:
        w = int(box[2] - box[0])
        h = int(box[3] - box[1])
        x_min = int(box[0])
        y_min = int(box[1])

        # Center of a given bounding box
        cx = (x_min + w / 2)
        cy = (y_min + h) / 2
        center.append((cx, cy))

        # New bouding box, with specified radius r and center [x_hat,y_hat,w_hat,h_hat,prob,c]
        # I've produced the new centered bounding box, so it doesn't exceed the real bounding box (with min,max)
        x_hat.append(max(cx - r, x_min))
        y_hat.append(max(cy - r, y_min) + 50)
        w_hat.append(min(2 * r, 2 * (cx - x_min)))
        h_hat.append(min(2 * r, 2 * (cy - y_min)))
        # h_hat.append(4*r)

    new_preds = [[x_hat[i], y_hat[i], w_hat[i], h_hat[i], round(result_preds[i, 4].item(), 3),
                  round(result_preds[i, 5].item(), 3)] for i in
                 range(result_preds.shape[0])]  # [x_hat,y_hat,w_hat,h_hat,prob,c]

    # bounding_box_noise = np.random.random((1, 3, 640, 1280)) * (strength)
    bounding_box_noise = np.zeros(X.shape)

    # Adding targeted noise *inside* each bounding box center with specified radius size and center
    for box in new_preds:
        x_min = int(box[0])
        w = int(box[2])
        y_min = int(box[1])
        h = int(box[3])
        bounding_box_noise[0, :, y_min:min(y_min + h, 640), x_min:min(x_min + w, 1280)] += np.random.random(
            (3, min(y_min + h, 640) - y_min, min(x_min + w, 1280) - x_min)) * strength

    delta1 = torch.from_numpy(bounding_box_noise)  # White noise (torch)

    inputs1 = X + delta1
    inputs1 = inputs1 % 255  # Plotting results
    # inputs1 = normalize(X + delta1)
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

    # noise1 = (delta1[0].permute(1, 2, 0) + 255) / 256 ###### UNCOMMENT IF YOU PUT strength < 0
    # noise1 = (delta1[0].permute(1, 2, 0)) / 256  ###### UNCOMMENT IF YOU PUT strength > 0

    original_img1 = original_img / 255
    attacked_img1 = attacked_img / 255
    output = model(np.asarray(inputs1[0].permute(1, 2, 0)))  # Run the model on the noised image

    plot_attacked_image_BOX(original_img1, attacked_img1, noise1, base_path, outputImgName, output.pred[0],
                            results.pred[0])  # plotting results and save them to output file
    print()


def attack_with_chosen_noise_strength(X, model, normalize, base_path, outputImgName, results):
    ############################################################################################
    # Attemp to add specific noise (with specified strength)
    strength = 100
    chosen_noise_strength = np.zeros(X.shape) - (strength)
    delta1 = torch.from_numpy(chosen_noise_strength)  # White noise (torch)
    inputs1 = normalize(X + delta1)

    # Plotting results
    original_img = X[0].permute(1, 2, 0)
    attacked_img = inputs1[0].permute(1, 2, 0)
    noise = attacked_img - original_img

    # Rescale pixels
    noise1 = (delta1[0].permute(1, 2, 0) + 255) / 256
    original_img1 = original_img / 255
    attacked_img1 = attacked_img / 255
    output = model(np.asarray(inputs1[0].permute(1, 2, 0)))  # Run the model on the white noised image

    outputImgName += "_" + str(strength) + "_CHOSEN_NOISE"
    plot_attacked_image_BOX(original_img1, attacked_img1, noise1, base_path, outputImgName, output.pred[0],
                            results.pred[0])  # plotting results and save them to output file
    print()


# Function that plots the results received from attack_pgd and saving result's image to directory
def plot_attacked_image(original_img, attacked_img, noise, base_path, outputImageName, pred_attack, pred_real):
    # Plotting configurations
    import matplotlib.pyplot as plt
    plt.figure()
    matplotlib.use('TkAgg')

    # create figure
    fig = plt.figure(figsize=(20, 20))

    # setting values to rows and column variables
    rows = 3
    columns = 1

    # showing image
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    plt.imshow(original_img)
    plt.axis('off')
    plt.title("Original image")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    plt.imshow(attacked_img)
    plt.axis('off')
    plt.title("Perturbed image")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    plt.imshow(noise)
    plt.axis('off')
    plt.title("Noise image")

    # Save results to output file path specified by basePath + outputImageName without overriding images in directory
    # counter = 1
    # while(True):
    #     newResult = base_path + outputImageName + str(counter) + ".jpg"
    #     if not os.path.exists(newResult):
    #         plt.savefig(newResult)
    #         break
    #     else:
    #         counter += 1
    #

    # Save results to output file path specified by basePath + outputImageName without overriding images in directory
    save_result_image(plt, base_path, outputImageName)

    plt.show()
    print()


########################################################################################################################

def attack_pgd(model, X, y, epsilon, alpha, num_restarts, attack_iters=1,
               multi_targeted=True, num_classes=10, use_adam=True,
               lower_limit=0.0, upper_limit=1.0, normalize=lambda x: x,
               initialization='uniform', results=None, imgPath=None):
    # if initialization == 'osi':
    #     if multi_targeted:
    #         extra_dim = (num_restarts, num_classes - 1,)
    #     else:
    #         extra_dim = (num_restarts)
    #     X_init = OSI_init(model, X, y, epsilon, num_classes, extra_dim=extra_dim)
    #
    # best_loss = torch.empty(X.size(0), device=X.device).fill_(float("-inf"))
    # best_delta = torch.zeros_like(X, device=X.device)
    #
    # input_shape = X.size()
    # if multi_targeted:
    #     assert target is None  # Multi-targeted attack is for non-targed attack only.
    #     extra_dim = (num_restarts, num_classes - 1,)
    #     # Add two extra dimensions for targets. Shape is (batch, restarts, target, ...).
    #     X = X.unsqueeze(1).unsqueeze(1).expand(-1, *extra_dim, *(-1,) * (X.ndim - 1))
    #     # Generate target label list for each example.
    #     E = torch.eye(num_classes, dtype=X.dtype, device=X.device)
    #     c = E.unsqueeze(0) - E[y].unsqueeze(1)
    #     # remove specifications to self.
    #     I = ~(y.unsqueeze(1) == torch.arange(num_classes, device=y.device).unsqueeze(0))
    #     # c has shape (batch, num_classes - 1, num_classes).
    #     c = c[I].view(input_shape[0], num_classes - 1, num_classes)
    #     # c has shape (batch, restarts, num_classes - 1, num_classes).
    #     c = c.unsqueeze(1).expand(-1, num_restarts, -1, -1)
    #     target_y = y.view(-1, *(1,) * len(extra_dim), 1).expand(-1, *extra_dim, 1)
    #     num_restarts = 1
    # else:
    #     if target is not None:
    #         # An attack target for targeted attack, in dimension (batch, ).
    #         target = torch.tensor(target, device='cuda').view(-1, 1)
    #         target_index = target.view(-1, 1, 1).expand(-1, num_restarts, 1)
    #         # Add an extra dimension for num_restarts. Shape is (batch, num_restarts, ...).
    #         X = X.unsqueeze(1).expand(-1, num_restarts, *(-1,) * (X.ndim - 1))
    #         # Only run 1 restart, since we run all restarts together.
    #         extra_dim = (num_restarts,)
    #         num_restarts = 1
    #
    # # This is the maximal/minimal delta values for each sample, each element.
    # sample_lower_limit = torch.clamp(lower_limit - X, min=-epsilon)
    # sample_upper_limit = torch.clamp(upper_limit - X, max=epsilon)
    #
    # success = False
    #
    # for n in range(num_restarts):
    #     # one_label_loss = True if n % 2 == 0 else False  # random select target loss or marginal loss
    #     one_label_loss = False  # Temporarily disabled. Will do more tests.
    #     if early_stop and success:
    #         break
    #
    #     if initialization == 'osi':
    #         delta = (X_init - X).detach().requires_grad_()
    #     elif initialization == 'uniform':
    #         delta = (torch.empty_like(X).uniform_() * (
    #                     sample_upper_limit - sample_lower_limit) + sample_lower_limit).requires_grad_()
    #     elif initialization == 'none':
    #         delta = torch.zeros_like(X).requires_grad_()
    #     else:
    #         raise ValueError(f"Unknown initialization method {initialization}")
    #
    #     if use_adam:
    #         opt = AdamClipping(params=[delta], lr=alpha)
    #         scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)

    for _ in range(attack_iters):
        strength = 125
        delta = np.random.random(size=X.shape) * strength
        inputs = normalize(X + delta)
        with torch.no_grad():
            output = model(
                np.asarray(inputs[0].permute(1, 2, 0)))  # Make img to be of the wanted size: (640,1280,3)

            # Plotting results
            original_img = X[0].permute(1, 2, 0)
            attacked_img = inputs[0].permute(1, 2, 0)
            noise = attacked_img - original_img

            # Rescale pixels
            noise = (noise + 255) / 256
            original_img = original_img / 255
            attacked_img = attacked_img / 255

            # Run the plotting to see results and save them to output file specified by basePath + outputFileName
            base_path = "/home/avraham/alpha-beta-CROWN/complete_verifier/Attack_Output/"
            outputImgName = "truck_all"  # Name of the image to save in attack directory
            #                    plot_attacked_image(original_img, attacked_img, noise, base_path, outputImgName,
            #                                        output.pred[0])  # plotting results and save them to output file

            ####################################################################################################
            ############################ RUNNING ATTACKS ON VARIOUS NOISING ALGORITHMS ##########################
            # attack_with_chosen_noise_strength(X, model, normalize, base_path, outputImgName, results)
            # attack_with_white_noise_strength(X, model, normalize, base_path, outputImgName, results)
            # attack_on_bounding_box(X, model, normalize, base_path, outputImgName, results)
            # attack_on_bounding_box_center(X, model, normalize, base_path, outputImgName, results)
            # attack_on_bounding_box_Bernoulli(X, model, normalize, base_path, outputImgName, results)
            attack_with_canny_and_Bernoulli(X, model, normalize, base_path, outputImgName, results, imgPath)
            ####################################################################################################
            print("################## Plotted attack images! ##################")

        #     # output = model(inputs, size=640)
        # if not multi_targeted:
        #     # Not using the multi-targeted loss. Can be target or non-target attack.
        #     if target is not None:
        #         # Targeted attack. In this case we have an extra (num_starts, ) dimension.
        #         runnerup = output.scatter(dim=2, index=target_index, value=-float("inf")).max(2).values
        #         # t = output.gather(dim=2, index=target_index).squeeze(-1)
        #         t = output[:, :, target].squeeze(-1).squeeze(-1)
        #         loss = (t - runnerup)
        #     else:
        #         # Non-targeted attack.
        #         if one_label_loss:
        #             # Loss 1: simply reduce the loss groundtruth target.
        #             loss = -output.gather(dim=1, index=y.view(-1,1)).squeeze(1)  # -groundtruth
        #         else:
        #             # Loss 2: reduce the margin between groundtruth and runner up label.
        #             runnerup = output.scatter(dim=1, index=y.view(-1,1), value=-100.0).max(1).values
        #             groundtruth = output.gather(dim=1, index=y.view(-1,1)).squeeze(1)
        #             # Use the margin as the loss function.
        #             loss = (runnerup - groundtruth)
        #     loss.sum().backward()
        # else:
        #     # Non-targeted attack, using margins between groundtruth class and all target classes together.
        #     # loss = torch.einsum('ijkl,ijkl->ijk', c, output)
        #     loss = torch.einsum('ijkl,ijkl->', c, output)
        #     loss.backward()
        #
        # # print(loss.sum().item(), output.detach().cpu().numpy())
        # # print(loss[:, :, 5])
        #
        # with torch.no_grad():
        #     # Save the best loss so far.
        #     if not multi_targeted:
        #         # Not using multi-targeted loss.
        #         if target is not None:
        #             # Targeted attack, need to check if the top-1 label is target label.
        #             # Since we merged the random restart dimension, we need to find the best one among all random restarts.
        #             all_loss, indices = loss.max(1)
        #             # Gather the delta for the best loss in all random restarts.
        #             delta_best = delta.gather(dim=1, index=indices.view(-1,1,1,1,1).expand(-1,-1,*input_shape[1:])).squeeze(1)
        #             best_delta[all_loss >= best_loss] = delta_best[all_loss >= best_loss]
        #             best_loss = torch.max(best_loss, all_loss)
        #         else:
        #             # Non-targeted attack. Success when the groundtruth is not top-1.
        #             if one_label_loss:
        #                 runnerup = output.scatter(dim=1, index=y.view(-1,1), value=-100.0).max(1).values
        #                 groundtruth = output.gather(dim=1, index=y.view(-1,1)).squeeze(1)
        #                 # Use the margin as the loss function.
        #                 criterion = (runnerup - groundtruth)  # larger is better.
        #             else:
        #                 criterion = loss
        #             # Larger is better.
        #             best_delta[criterion >= best_loss] = delta[criterion >= best_loss]
        #             best_loss = torch.max(best_loss, criterion)
        #     else:
        #         # Using multi-targeted loss. Need to find which label causes the worst case margin.
        #         # Keep the one with largest margin.
        #         # Note that we recompute the runnerup label here - the runnerup label might not be the target label.
        #         # output has shape (batch, restarts, num_classes-1, num_classes).
        #         # runnerup has shape (batch, restarts, num_classes-1).
        #         runnerup = output.scatter(dim=3, index=target_y, value=-float("inf")).max(3).values
        #         # groundtruth has shape (batch, restarts, num_classes-1).
        #         groundtruth = output.gather(dim=3, index=target_y).squeeze(-1)
        #         # margins has shape (batch, restarts * num_classes), ).
        #         margins = (runnerup - groundtruth).view(groundtruth.size(0), -1)
        #         # all_loss and indices have shape (batch, ), and this is the best loss over all restarts and number of classes.
        #         all_loss, indices = margins.max(1)
        #         # delta has shape (batch, restarts, num_class-1, c, h, w). For each batch element, we want to select from the best over (restarts, num_classes-1) dimension.
        #         # delta_targeted has shape (batch, c, h, w).
        #         delta_targeted = delta.view(delta.size(0), -1, *input_shape[1:]).gather(dim=1, index=indices.view(-1,1,*(1,) * (len(input_shape) - 1)).expand(-1,-1,*input_shape[1:])).squeeze(1)
        #         best_delta[all_loss >= best_loss] = delta_targeted[all_loss >= best_loss]
        #         best_loss = torch.max(best_loss, all_loss)
        #
        #     if early_stop:
        #         if multi_targeted:
        #             # Must be a untargeted attack. If any of the target succeed, that element in batch is successfully attacked.
        #             # output has shape (batch, num_restarts, num_classes-1, num_classes,).
        #             if (output.view(output.size(0), -1, num_classes).max(2).indices != y.unsqueeze(1)).any(1).all():
        #                 print('pgd early stop.')
        #                 success = True
        #                 break
        #         elif target is not None:
        #             # Targeted attack, the top-1 label of every element in batch must match the target.
        #             # If any attack in some random restarts succeeds, the attack is successful.
        #             if (output.max(2).indices == target).any(1).all():
        #                 print('pgd early stop.')
        #                 success = True
        #                 break
        #         else:
        #             # Non-targeted attack, the top-1 label of every element in batch must not be the groundtruth label.
        #             if (output.max(1).indices != y).all():
        #                 print('pgd early stop.')
        #                 success = True
        #                 break
        #
        #     # Optimizer step.
        #     if use_adam:
        #         opt.step(clipping=True, lower_limit=sample_lower_limit, upper_limit=sample_upper_limit, sign=1)
        #         opt.zero_grad(set_to_none=True)
        #         scheduler.step()
        #     else:
        #         d = delta + alpha * torch.sign(delta.grad)
        #         d = torch.max(torch.min(d, sample_upper_limit), sample_lower_limit)
        #         delta.copy_(d)
        #         delta.grad = None

    attacked_images = None
    return attacked_images, None
    # return best_delta, delta


# class AdamClipping(Optimizer):
#     r"""Implements Adam algorithm, with per-parameter gradient clipping.
#     The function is from PyTorch source code.
#
#     It has been proposed in `Adam: A Method for Stochastic Optimization`_.
#
#     Arguments:
#         params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#         lr (float, optional): learning rate (default: 1e-3)
#         betas (Tuple[float, float], optional): coefficients used for computing
#             running averages of gradient and its square (default: (0.9, 0.999))
#         eps (float, optional): term added to the denominator to improve
#             numerical stability (default: 1e-8)
#         weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#         amsgrad (boolean, optional): whether to use the AMSGrad variant of this
#             algorithm from the paper `On the Convergence of Adam and Beyond`_
#             (default: False)
#
#     .. _Adam\: A Method for Stochastic Optimization:
#         https://arxiv.org/abs/1412.6980
#     .. _On the Convergence of Adam and Beyond:
#         https://openreview.net/forum?id=ryQu7f-RZ
#     """
#
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
#                  weight_decay=0, amsgrad=False):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#         if not 0.0 <= weight_decay:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         defaults = dict(lr=lr, betas=betas, eps=eps,
#                         weight_decay=weight_decay, amsgrad=amsgrad)
#         super().__init__(params, defaults)
#
#     def __setstate__(self, state):
#         super().__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('amsgrad', False)
#
#
#     @staticmethod
#     @torch.no_grad()
#     @torch.jit.script
#     def _clip_update(exp_avg : torch.Tensor, denom : torch.Tensor, step_size : float, clipping_step_eps : float, lower_limit : torch.Tensor, upper_limit : torch.Tensor, p : torch.Tensor):
#         # Compute the Adam update.
#         update = exp_avg / denom * step_size
#         # update = p.grad
#         # Linf norm, scale according to sign.
#         scaled_update = torch.sign(update) * clipping_step_eps
#         # Apply the update.
#         d = p.data + scaled_update
#         # Avoid out-of-boundary updates.
#         d = torch.max(torch.min(d, upper_limit), lower_limit)
#         p.copy_(d)
#
#     @torch.no_grad()
#     def step(self, clipping=None, lower_limit=None, upper_limit=None, sign=None, closure=None):
#         """Performs a single optimization step.
#
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()
#
#         # Currently we only deal with 1 parameter group.
#         assert len(self.param_groups) == 1
#         for group in self.param_groups:
#             for i, p in enumerate(group['params']):
#                 if p.grad is None:
#                     continue
#                 grad = p.grad
#                 if grad.is_sparse:
#                     raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
#                 amsgrad = group['amsgrad']
#
#                 state = self.state[p]
#
#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     # Exponential moving average of gradient values
#                     state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                     # Exponential moving average of squared gradient values
#                     state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                     if amsgrad:
#                         # Maintains max of all exp. moving avg. of sq. grad. values
#                         state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#
#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 if amsgrad:
#                     max_exp_avg_sq = state['max_exp_avg_sq']
#                 beta1, beta2 = group['betas']
#
#                 state['step'] += 1
#                 bias_correction1 = 1 - beta1 ** state['step']
#                 bias_correction2 = 1 - beta2 ** state['step']
#
#                 if group['weight_decay'] != 0:
#                     grad = grad.add(p, alpha=group['weight_decay'])
#
#                 # Decay the first and second moment running average coefficient
#                 exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
#                 if amsgrad:
#                     # Maintains the maximum of all 2nd moment running avg. till now
#                     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
#                     # Use the max. for normalizing running avg. of gradient
#                     denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
#                 else:
#                     denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
#
#                 step_size = group['lr'] / bias_correction1
#
#                 if clipping:
#                     assert sign == 1  # gradient ascent for adversarial attacks.
#                     self._clip_update(exp_avg, denom, step_size, step_size, lower_limit, upper_limit, p)
#                 else:
#                     # No clipping. Original Adam update.
#                     p.addcdiv_(exp_avg, denom, value=-step_size)
#
#         return loss
#


def pgd_attack(dataset, model, x, max_eps, data_min, data_max, y=None,
               initialization="uniform", results=None, imgPath=None):
    best_deltas, last_deltas = attack_pgd(model, X=x, y=torch.tensor([y], device=x.device),
                                          epsilon=float("inf"),
                                          attack_iters=1,
                                          num_restarts=1,
                                          upper_limit=data_max, lower_limit=data_min,
                                          initialization=initialization,
                                          results=results, imgPath=imgPath, alpha=0.01,
                                          num_classes=80, use_adam=False, multi_targeted=True)
    return None, None, None


def pgd_attack_original(dataset, model, x, max_eps, data_min, data_max, vnnlib=None, y=None,
                        target=None, only_target_attack=False, initialization="uniform", results=None, imgPath=None):
    r"""Interface to PGD attack.

    Args:
        dataset (str): The name of dataset. Each dataset might have different attack configurations.

        model (torch.nn.Module): PyTorch module under attack.

        x (torch.tensor): Input image (x_0).

        max_eps (float): Perturbation Epsilon. Assuming Linf perturbation for now. (e.g., 0.3 for MNIST)

        data_min (torch.tensor): Lower bounds of data input. (e.g., 0 for mnist)

        data_max (torch.tensor): Lower bounds of data input. (e.g., 1 for mnist)

        vnnlib (list, optional): VNNLIB specifications. It will be used to extract attack target.

        y (int, optional): Groundtruth label. If specified, vnnlib will be ignored.

    Returns:
        success (bool): True if attack is successful. Otherwise False.
        attack_images (torch.tensor): last adversarial examples so far, may not be a real adversarial example if attack failed
    """
    # assert arguments.Config["specification"]["norm"] == np.inf, print('We only support Linf-norm attack.')
    # if dataset in ["MNIST", "CIFAR"]:
    #     if y is not None and target is None:
    #         # Use y as the groundtruth label.
    #         pidx_list = ["all"]
    #         if data_max is None:
    #             # Adding noise (to a max_eps) to a given img (denoted by x)
    #             data_max = x + max_eps
    #             data_min = x - max_eps
    #         else:
    #             data_max = torch.min(x + max_eps, data_max)
    #             data_min = torch.max(x - max_eps, data_min)
    #     else:
    #         pidx_list = []
    #         if vnnlib is not None:
    #             # Extract attack target from vnnlib.
    #             for prop_mat, prop_rhs in vnnlib[0][1]:
    #                 if len(prop_rhs) > 1:
    #                     output = model(x).detach().cpu().numpy().flatten()
    #                     print('model output:', output)
    #                     vec = prop_mat.dot(output)
    #                     selected_prop = prop_mat[vec.argmax()]
    #                     y = int(np.where(selected_prop == 1)[0])  # true label, whatever in target attack
    #                     pidx = int(np.where(selected_prop == -1)[0])  # target label
    #                     only_target_attack = True
    #                 else:
    #                     assert len(prop_mat) == 1
    #                     y = np.where(prop_mat[0] == 1)[0]
    #                     if len(y) != 0:
    #                         y = int(y)
    #                     else:
    #                         y = None
    #                     pidx = int(np.where(prop_mat[0] == -1)[0])  # target label
    #                 if pidx == y:
    #                     raise NotImplementedError
    #                 pidx_list.append(pidx)
    #         elif target is not None:
    #             pidx_list.append(target)
    #         else:
    #             raise NotImplementedError
    #
    #     print('##### PGD attack: True label: {}, Tested against: {} ######'.format(y, pidx_list))
    #     if not isinstance(max_eps, float):
    #         max_eps = torch.max(max_eps).item()
    #
    #     if only_target_attack:
    #         # Targeted attack PGD.
    #         if arguments.Config["attack"]["pgd_alpha"] == 'auto':
    #             alpha = max_eps
    #         else:
    #             alpha = float(arguments.Config["attack"]["pgd_alpha"])
    #         best_deltas, last_deltas = attack_pgd(model, X=x, y=None, epsilon=float("inf"), alpha=alpha,
    #                                               attack_iters=arguments.Config["attack"]["pgd_steps"],
    #                                               num_restarts=arguments.Config["attack"]["pgd_restarts"],
    #                                               upper_limit=data_max, lower_limit=data_min,
    #                                               multi_targeted=False,
    #                                               lr_decay=arguments.Config["attack"]["pgd_lr_decay"],
    #                                               target=pidx_list[0], initialization=initialization,
    #                                               early_stop=arguments.Config["attack"]["pgd_early_stop"])
    #     else:
    #         # Untargeted attack PGD.
    #         if arguments.Config["attack"]["pgd_alpha"] == 'auto':
    #             alpha = max_eps / 4.0
    #         else:
    #             alpha = float(arguments.Config["attack"]["pgd_alpha"])
    #         '''
    #         best_deltas, last_deltas = attack_pgd(model, X=x, y=torch.tensor([y], device=x.device), epsilon=float("inf"), alpha=alpha,
    #                 attack_iters=arguments.Config["attack"]["pgd_steps"], num_restarts=arguments.Config["attack"]["pgd_restarts"], upper_limit=data_max, lower_limit=data_min,
    #                 multi_targeted=True, lr_decay=arguments.Config["attack"]["pgd_lr_decay"], target=None, initialization=initialization, early_stop=arguments.Config["attack"]["pgd_early_stop"])
    #         '''
    #         best_deltas, last_deltas = attack_pgd(model, X=x, y=torch.tensor([y], device=x.device),
    #                                               epsilon=float("inf"), alpha=alpha,
    #                                               attack_iters=arguments.Config["attack"]["pgd_steps"],
    #                                               num_restarts=1,
    #                                               upper_limit=data_max, lower_limit=data_min,
    #                                               multi_targeted=False,
    #                                               lr_decay=arguments.Config["attack"]["pgd_lr_decay"], target=None,
    #                                               initialization=initialization,
    #                                               early_stop=arguments.Config["attack"]["pgd_early_stop"],
    #                                               results=results, imgPath=imgPath)
    #
    #     if x.shape[0] == 1:
    #         attack_image = torch.max(torch.min(x + best_deltas, data_max), data_min)
    #         assert (attack_image >= data_min).all()
    #         assert (attack_image <= data_max).all()
    #         # assert (attack_image-x).abs().max() <= eps_temp.max(), f"{(attack_image-x).abs().max()} <= {eps_temp.max()}"
    #         attack_output = model(attack_image).squeeze(0)
    #
    #         # FIXME (10/02): This is not the best image for each attack target. We should save best attack delta for each target.
    #         all_targets_attack_image = torch.max(torch.min(x + last_deltas, data_max), data_min)
    #
    #         attack_label = attack_output.argmax()
    #         print("pgd prediction:", attack_output)
    #
    #         # FIXME (10/05): Cleanup.
    #         if only_target_attack:
    #             # Targeted attack, must have one label.
    #             attack_logit = attack_output.data[pidx_list[0]].item()
    #             attack_output.data[pidx_list[0]] = -float("inf")
    #             attack_margin = attack_output.max().item() - attack_logit
    #             print("attack margin", attack_margin)
    #             if attack_label == pidx_list[0]:
    #                 assert len(pidx_list) == 1
    #                 print("targeted pgd succeed, label {}, against label {}".format(y, attack_label))
    #                 # FIXME (10/05): Please check! attack_image is for one target only.
    #                 return True, attack_image.detach(), [attack_margin]
    #             else:
    #                 print(f"targeted pgd failed, margin {attack_margin}")
    #                 return False, attack_image.detach(), [attack_margin]
    #         else:
    #             # Untargeted attack, any non-groundtruth label is ok.
    #             groundtruth_logit = attack_output.data[y].item()
    #             attack_output.data[y] = -float("inf")
    #             attack_margin = groundtruth_logit - attack_output
    #             print("attack margin", attack_margin)
    #             # Untargeted attack, any non-groundtruth label is ok.
    #             if attack_label != y:
    #                 print("untargeted pgd succeed, label {}, against label {}".format(y, attack_label))
    #                 return True, all_targets_attack_image.detach().detach(), attack_margin.detach().cpu().numpy()
    #             else:
    #                 print("untargeted pgd failed")
    #                 return False, all_targets_attack_image.detach(), attack_margin.detach().cpu().numpy()
    #
    #     else:
    #         # FIXME (10/02): please remove duplicated code!
    #         attack_images = torch.max(torch.min(x + best_deltas, data_max), data_min)
    #         attack_output = model(attack_images).squeeze(0)
    #         # do in batch attack
    #         attack_label = attack_output.argmax(1)
    #
    #         if only_target_attack:
    #             # Targeted attack, must have one label.
    #             if (attack_label == pidx_list[0]).any():
    #                 # FIXME (10/02): remove duplicated code.
    #                 assert len(pidx_list) == 1
    #                 # print("targeted pgd succeed, label {}, against label {}".format(y, attack_label))
    #                 attack_logit = attack_output.data[:, pidx_list[0]].clone()
    #                 attack_output.data[:, pidx_list[0]] = -float("inf")
    #                 attack_margin = attack_output.max(1).values - attack_logit
    #                 return True, attack_images.detach(), attack_margin.detach().cpu().numpy()
    #             else:
    #                 attack_logit = attack_output.data[:, pidx_list[0]].clone()
    #                 attack_output.data[:, pidx_list[0]] = -float("inf")
    #                 attack_margin = attack_output.max(1).values - attack_logit
    #                 # print(f"targeted pgd failed, margin {attack_margin}")
    #                 return False, attack_images.detach(), attack_margin.detach().cpu().numpy()
    #         else:
    #             raise NotImplementedError
    #             # TODO support batch
    #             # Untargeted attack, any non-groundtruth label is ok.
    #             groundtruth_logit = attack_output.data[y].item()
    #             attack_output.data[y] = -float("inf")
    #             attack_margin = groundtruth_logit - attack_output
    #             # Untargeted attack, any non-groundtruth label is ok.
    #             if attack_label != y:
    #                 print("untargeted pgd succeed, label {}, against label {}".format(y, attack_label))
    #                 return True, attack_images.detach(), attack_margin.detach().cpu().numpy()
    #             else:
    #                 print("untargeted pgd failed")
    #                 return False, attack_images.detach(), attack_margin.detach().cpu().numpy()

    if False:
        pass
    else:
        print("pgd attack not supported for dataset", dataset)
        raise NotImplementedError
