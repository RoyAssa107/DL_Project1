# Adversarial Attacks on Object Detection Models 2021-2022

## Description:
_In our project, we have implemented 6 different attacks regarding the different type <br>
of adversarial attack to indicate that even a state-of-the-art OD model is vulnerable <br>
to the following adversarial attacks, including:_
   1. **_White noise attack_**
   2. **_Edge noise attack_**
   3. **_Bounding box attack_**
   4. **_Rectangle noise attack_**
   5. **_Bounding box with Bernoulli sampling attack_**
   6. **_Ellipse Bounding Box Attack_** 

_In addition, we have added support for different attack purposes such as:_
   1. **_Missing Detection_**
   2. **_False Positive_**
   3. **_IoU Threshold_**

_We have created a dedicated interface via **config.txt** file containing several key configuration <br>
settings which control the following arguments:_

### General Parameters:
* **_device_** - Type of device to run the code on.
* **_model_** - Type of model to explore.
* **_conf_model_** - Confidence level of the specified model for creating designed for an object's bbox. 
* **_iou_thresh_model_** - IoU threshold target for the attack.

### Dataset Parameters:
* **_relative_path_** - Relative path of the base directory of test images.

### Attack Parameters:
  * **_max_iter_** - Maximal number of iteration for the specified attack in each image. 
  * **_max_strength_** - Maximal strength of noising per pixel. 
  * **_target_** - Purpose of the attack. 
  * **_noise_algorithm_** - Type of noising algorithm to apply on test images.
  * **_max_prob_** - Maximal probability for Bernoulli sampling attack.
  * **_upper_IoU_** - Upper bound for IoU attack. 
  * **_lower_IoU_** - Lower bound for IoU attack. 


### Summarization of parameters and their possible values:
| device | model | conf_model | iou_thresh_model | relative_path | max_iter | max_strength |      target       |        noise_algorithm        | max_prob | upper_IoU | lower_IoU |
|:------:|:-----:|:----------:|:----------------:|:-------------:|:--------:|:------------:|:-----------------:|:-----------------------------:|:--------:|:---------:|:---------:|
|  cpu   | yolo  |   [0,1]    |      [0,1]       |   ./images/   |    0     |      0       | Missing Detection |      White_Noise_Attack       |  [0,1]   |   [0,1]   |   [0,1]   |
|  cuda  |       |            |                  |               |    1     |      1       |  False Positive   | Bounding_Box_Attack_Rectangle |          |           |           |
|        |       |            |                  |               |    2     |      2       |   IoU Threshold   |  Bounding_Box_Center_Attack   |          |           |           |
|        |       |            |                  |               |   ...    |     ...      |                   | Bernoulli_Bounding_Box_Attack |          |           |           |
|        |       |            |                  |               |          |     255      |                   |    Canny_Bernoulli_Attack     |          |           |           |
|        |       |            |                  |               |          |              |                   |  Ellipse_Bounding_Box_Attack  |          |           |           |


## Installations:
1. _Git must be installed on your machine. If you haven't installed git on your machine,<br>
    please refer to the following link for further details:_ https://github.com/git-guides/install-git <br><br>
2. _Clone the repo to your local directory using the following command:_ <br> `$ git clone https://github.com/RoyAssa107/DL_Project1.git` <br><br>  

3. _Install all required packages by run the following command in the terminal:_ <br>`$ pip install -r requirements.txt `


## How to run the code:
_After you have cloned the project's repo, installed all the packages, configures all parameters as explained above <br>
and set up the directories, please follow these instructions:_
        
        1. Run UI_support_attack_OOP.py  

        2. If attack succeded, you will be able to find adversarial attack results 
           under the current directory named Attack_Output.

        3. Check the message in the terminal for summariazation of the attack, including several statistics.



## Contributors:
  1. *Roy Assa*
  2. *Avraham Raviv*
  3. *Itav Nissim*