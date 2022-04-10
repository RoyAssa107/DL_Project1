from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import webcolors

# from scipy.spatial import KDTree
# from webcolors import (
#     CSS3_HEX_TO_NAMES,
#     hex_to_rgb,
# )
#
#
# def convert_rgb_to_names(rgb_tuple):
#     # a dictionary of all the hex and their respective names in css3
#     css3_db = CSS3_HEX_TO_NAMES
#     names = []
#     rgb_values = []
#     for color_hex, color_name in css3_db.items():
#         names.append(color_name)
#         rgb_values.append(hex_to_rgb(color_hex))
#
#     kdt_db = KDTree(rgb_values)
#     distance, index = kdt_db.query(rgb_tuple)
#     return f'{names[index]}'


def RGB2HEX(color, type_label="RGB"):
    if type_label == "RGB":
        rgb_color = [round(int(color[2])), round(int(color[1])), round(int(color[0]))]
        # return convert_rgb_to_names(rgb_color)
        return rgb_color
    else:
        hex_color = "#{:02x}{:02x}{:02x}".format(round(int(color[2])), round(int(color[1])), round(int(color[0])))
        return hex_color
        # return webcolors.hex_to_name(hex_color)


# Function that prints distribution of colors in a given image
def get_colors_distribution(img=None, imgPath=None, num_colors=3, show_chart=True, debug=False, seed=0):
    np.random.seed(seed)
    if debug is False:
        modified_image = cv2.resize(img, (600, 400), interpolation=cv2.INTER_AREA)
    else:
        modified_image = img
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    # Initiate KMEANS object
    clf = KMeans(n_clusters=num_colors)
    labels = clf.fit_predict(modified_image)

    # Extract count of each color appearance in img
    counts = Counter(labels)
    center_colors = clf.cluster_centers_

    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i], type_label="HEX") for i in counts.keys()]
    # rgb_colors = [ordered_colors[i] for i in counts.keys()]
    rgb_colors = [RGB2HEX(ordered_colors[i], type_label="RGB") for i in counts.keys()]

    # Plotting the results
    if show_chart:
        figure, axes = plt.subplots(1, 2, figsize=(15, 8))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Image Path: {imgPath}")

        axes[1].pie(counts.values(), labels=rgb_colors, colors=hex_colors)
        axes[1].set_title('Plotting distribution of colors in the format: [R,G,B]')


        # plt.figure(figsize=(8, 6))
        # plt.pie(counts.values(), labels=rgb_colors, colors=hex_colors)
        # plt.title(f"Image: {imgPath} \n\nPlotting distribution of colors in the format: [R,G,B]")
        plt.show()
        print()
    return rgb_colors


if __name__ == "__main__":
    np.random.seed(0)
    path = "images\\Bird\\blue.jpeg"
    # path = "images\\Bird\\bird1.jpg"
    # path = "images\\Airplane\\airplane1.jpg"

    image = cv2.imread(path, cv2.COLOR_BGR2RGB)
    # cv2.imshow("airplane", image)
    # cv2.waitKey(0)
    get_colors_distribution(img=image, imgPath=path, num_colors=8, show_chart=True)
    cv2.waitKey(0)
    print("Exit!")
