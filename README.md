# Adversarial Attacks on Object Detection Models 2021-2022

## Description
In our project, we have implemented 6 different attacks regarding the different type <br> of adversarial attack to indicate
that even a state-of-the-art OD model is vulnerable <br>
to the following adversarial attacks, including:
   1. White noise attack
   2. Edge noise attack
   3. Bounding box attack
   4. Rectangle noise attack
   5. Bounding box with Bernoulli sampling attack
   6. Ellipse Bounding Box Attack 

In addition, we have added support for different attack purposes such as:
   1. Missing Detection
   2. False Positive
   3. IoU Threshold

We have created a dedicated interface via **config.txt** file containing several key configuration <br>
settings which control the following arguments:

### General Arguments:
* **_device_** - Type of device to run the code on.
* **_model_** - Type of model to explore.
* **_conf_model_** - Confidence level of the specified model for creating designed for an object's bbox. 
* **_iou_thresh_model_** - IoU threshold target for the attack.

### Dataset Arguments:
* **_relative_path_** - Relative path of the base directory of test images.

### Attack Arguments:
  * **_max_iter_** - Maximal number of iteration for the specified attack in each image. 
  * **_max_strength_** - Maximal strength of noising per pixel. 
  * **_target_** - Purpose of the attack. 
  * **_noise_algorithm_** - Type of noising algorithm to apply on test images.
  * **_max_prob_** - Maximal probability for Bernoulli sampling attack.
  * **_upper_IoU_** - Upper bound for IoU attack. 
  * **_lower_IoU_** - Lower bound for IoU attack. 


##
| device | model | conf_model | iou_thresh_model | relative_path | max_iter | max_strength |      target       |        noise_algorithm        | max_prob | upper_IoU | lower_IoU |
|:------:|:-----:|:----------:|:----------------:|:-------------:|:--------:|:------------:|:-----------------:|:-----------------------------:|:--------:|:---------:|:---------:|
|  cpu   | yolo  |   [0,1]    |      [0,1]       |   ./images/   |    0     |      0       | Missing Detection |      White_Noise_Attack       |  [0,1]   |   [0,1]   |   [0,1]   |
|  cuda  |       |            |                  |               |    1     |      1       |  False Positive   | Bounding_Box_Attack_Rectangle |          |           |           |
|        |       |            |                  |               |    2     |      2       |   IoU Threshold   |  Bounding_Box_Center_Attack   |          |           |           |
|        |       |            |                  |               |   ...    |     ...      |                   | Bernoulli_Bounding_Box_Attack |          |           |           |
|        |       |            |                  |               |          |     255      |                   |    Canny_Bernoulli_Attack     |          |           |           |
|        |       |            |                  |               |          |              |                   |  Ellipse_Bounding_Box_Attack  |          |           |           |



## Contributors:
  1. *Roy Assa*
  2. *Avraham Raviv*
  3. *Itav Nissim*