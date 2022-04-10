import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt
import torch


# Blur a given image
def blur_image(image, ksize=(5, 5), min_threshold=128, max_threshold=255, plot=True):
    # image = cv2.imread('images/Airplane/airplane1.jpg')

    # Smoothing image
    blurred_img = cv2.GaussianBlur(image, ksize, 0)
    mask = np.zeros(image.shape, np.uint8)

    # Create gray scale of the given image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Applying cv2.THRESH_BINARY thresholding techniques
    # thresh = cv2.threshold(gray, min_threshold, max_threshold, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(gray, min_threshold, max_threshold, cv2.THRESH_BINARY)[1]

    # Extracting the contours from the given binary image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(mask, contours, -1, (255, 255, 255), 5)
    output = np.where(mask == np.array([255, 255, 255]), blurred_img, image)  # Apply blur only on contour,
    # and keep original pixels otherwise

    if plot:  # For Debugging!
        cv2.imshow("Blurred contour", output)
        cv2.waitKey(0)
    return output

def Canny_Blur_Image(image, low_threshold=0, max_low_Threshold=255,
                     window_name='Edge Map', kernel_size=(5, 5), plot=True):
    img_blur = cv2.blur(image, kernel_size)
    src_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(image.shape, np.uint8)

    detected_edges = cv2.Canny(img_blur, low_threshold, max_low_Threshold, kernel_size)   #### src_gray
    dst = image * (mask[:, :, None].astype(image.dtype))

    output = np.where(mask == np.array([255, 255, 255]), img_blur, image)  # Apply blur only on contour,
                                                                              # and keep original pixels otherwise
    if plot:
        cv2.imshow(window_name, output)
        cv2.waitKey(0)

    return output

# # Blur a given image
# image = cv2.imread('images/Airplane/airplane1.jpg')
# blurred_img = cv2.GaussianBlur(image, (21, 21), 0)
# mask = np.zeros(image.shape, np.uint8)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
# contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.drawContours(mask, contours, -1, (255, 255, 255), 5)
# output = np.where(mask == np.array([255, 255, 255]), blurred_img, image)
# cv2.imshow("Blurred contour", output)
# cv2.waitKey(0)

#
# # Find edges using Canny Algorithm
# imgPath = "images/Airplane/airplane1.jpg"
# img = cv2.imread(imgPath, cv2.COLOR_BGR2RGB)
# canny_output = torch.tensor(cv2.Canny(img, 1000, 1000))
#
# img = torch.tensor(img)
# # plt.imshow(img)
# # plt.show()
#
# strength = 255
# img_size = img.shape
# noise = torch.tensor((np.random.random(size=img_size)) * strength)
#
# canny_output_3_dim = torch.tensor(canny_output).repeat(3, 1, 1).permute(1, 2,
#                                                                         0) / 255  # Shape: (1024,1280,3) for truck.jpg
# noised_edges = noise * canny_output_3_dim
# plt.imshow(noised_edges / 255)
# plt.show()
#
#
# # Apply Gaussian blur, creating a new series of images according to different gaussian blur
# num_blurres = 4
# blurred = [skimage.filters.gaussian(img / 255, sigma=(sigma, sigma), truncate=3.5, multichannel=True) for sigma in
#            range(1, num_blurres + 1)]
# # display blurred image
# figure, axes = plt.subplots(1, num_blurres, figsize=(15, 8))
# sigma = 1
# for i in range(num_blurres):
#     axes[i].imshow(cv2.cvtColor(blurred[i], cv2.COLOR_BGR2RGB))
#     axes[i].set_title(f"Gaussian Blur with sigma={sigma}")
#     sigma += 1
# plt.show()
# print()


# cv2.imshow("Blurred Image ", blurred)
# cv2.waitKey(0)


# new_noised_image = img + noised_edges
# plt.imshow(new_noised_image/255)
# plt.show()


# titles = ['Original Image', 'Canny']
# images = [img,canny_output]
#
# for i in range(len(images)):
#     plt.subplot(1,len(images),i+1)
#     plt.imshow(images[i])
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
#
# plt.show()


# import cv2
# import numpy as np
# from scipy.ndimage.filters import convolve, gaussian_filter
# # from scipy.misc import imread, imshow
#
#
# def CannyEdgeDetector(im, blur=1, highThreshold=91, lowThreshold=31):
#     im = np.array(im, dtype=float)  # Convert to float to prevent clipping values
#
#     # Gaussian blur to reduce noise
#     im2 = gaussian_filter(im, blur)
#
#     # Use sobel filters to get horizontal and vertical gradients
#     im3h = convolve(im2, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     im3v = convolve(im2, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#
#     # Get gradient and direction
#     grad = np.power(np.power(im3h, 2.0) + np.power(im3v, 2.0), 0.5)
#     theta = np.arctan2(im3v, im3h)
#     thetaQ = (np.round(theta * (5.0 / np.pi)) + 5) % 5  # Quantize direction
#
#     # Non-maximum suppression
#     gradSup = grad.copy()
#     for r in range(im.shape[0]):
#         for c in range(im.shape[1]):
#             # Suppress pixels at the image edge
#             if r == 0 or r == im.shape[0] - 1 or c == 0 or c == im.shape[1] - 1:
#                 gradSup[r, c] = 0
#                 continue
#             tq = thetaQ[r, c] % 4
#
#             if tq == 0:  # 0 is E-W (horizontal)
#                 if grad[r, c] <= grad[r, c - 1] or grad[r, c] <= grad[r, c + 1]:
#                     gradSup[r, c] = 0
#             if tq == 1:  # 1 is NE-SW
#                 if grad[r, c] <= grad[r - 1, c + 1] or grad[r, c] <= grad[r + 1, c - 1]:
#                     gradSup[r, c] = 0
#             if tq == 2:  # 2 is N-S (vertical)
#                 if grad[r, c] <= grad[r - 1, c] or grad[r, c] <= grad[r + 1, c]:
#                     gradSup[r, c] = 0
#             if tq == 3:  # 3 is NW-SE
#                 if grad[r, c] <= grad[r - 1, c - 1] or grad[r, c] <= grad[r + 1, c + 1]:
#                     gradSup[r, c] = 0
#
#     # Double threshold
#     strongEdges = (gradSup > highThreshold)
#
#     # Strong has value 2, weak has value 1
#     thresholdedEdges = np.array(strongEdges, dtype=np.uint8) + (gradSup > lowThreshold)
#
#     # Tracing edges with hysteresis
#     # Find weak edge pixels near strong edge pixels
#     finalEdges = strongEdges.copy()
#     currentPixels = []
#     for r in range(1, im.shape[0] - 1):
#         for c in range(1, im.shape[1] - 1):
#             if thresholdedEdges[r, c] != 1:
#                 continue  # Not a weak pixel
#
#             # Get 3x3 patch
#             localPatch = thresholdedEdges[r - 1:r + 2, c - 1:c + 2]
#             patchMax = localPatch.max()
#             if patchMax == 2:
#                 currentPixels.append((r, c))
#                 finalEdges[r, c] = 1
#
#     # Extend strong edges based on current pixels
#     while len(currentPixels) > 0:
#         newPix = []
#         for r, c in currentPixels:
#             for dr in range(-1, 2):
#                 for dc in range(-1, 2):
#                     if dr == 0 and dc == 0: continue
#                     r2 = r + dr
#                     c2 = c + dc
#                     if thresholdedEdges[r2, c2] == 1 and finalEdges[r2, c2] == 0:
#                         # Copy this weak pixel to final result
#                         newPix.append((r2, c2))
#                         finalEdges[r2, c2] = 1
#         currentPixels = newPix
#
#     return finalEdges
#
#
# if __name__ == "__main__":
#
#     im = cv2.imread("/home/avraham/alpha-beta-CROWN/complete_verifier/images/Truck/truck.jpg", 0)  # Open image, convert to greyscale
#     finalEdges = CannyEdgeDetector(im)
#     plt.imshow(finalEdges)
#     plt.show()
