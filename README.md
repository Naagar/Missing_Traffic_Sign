## [CHALLENGE: Missing Traffic Sign](https://mobility.iiit.ac.in/ncvpripg2023/c4mtschallenge/)
NCVPRIPG 2023 CHALLENGE ON CATEGORIZING MISSING

### [Paper- C4MTS: Challenge on Categorizing Missing Traffic Signs from Contextual Cues](https://link.springer.com/chapter/10.1007/978-981-97-5212-6_16)

Dataset: IDD, IIIT-Hyderabad https://idd.insaan.iiit.ac.in/
## OVERVIEW
The international Autonomous Vehicle (AV) market is estimated to reach around two thousand billion by 2030. Millions of lives are lost yearly in road accidents, and traffic violations cause a significant percentage of AV accidents. The traffic signs are generally installed at the side of the road to control traffic flow or pass information about the road environment to Vulnerable Road Users (VRUs). Often, the information is also available in the form of cues present in the context around the traffic signs in the cues away from it, which we refer to as contextual cues.

## CHALLENGE
The C4MTS challenge of Categorizing Missing Traffic Signs provides 200 real scenes from the Missing Traffic Sign Video Dataset (MTSVD) uniformly distributed over four types of missing traffic signs: left-hand-curve, right-hand-curve, gap-in-median, and side-road-left. The traffic signs are common but individually observed with contextual cues. In the given examples, contextual cues like rumble strips, side roads, etc., are present, but the traffic signs are missing due to improper planning, lack of budget, etc. To learn the relationship between the traffic signs and the contextual cues, 2000 training images, each containing one of the four traffic signs (commonly and individually visible with contextual cues) and corresponding bounding boxes, are also provided. Two tasks are proposed for the challenge: i) Object Detection, wherein model training happens using bounding box annotations provided with the data, and ii) Missing Traffic Sign Scene Categorization, wherein model training happens using road scene images with in-painted traffic signs provided with the data. The models for the second task will be tested on a mixture of images with in-painted traffic signs and images with missing traffic signs to encourage inpainting agnostic real-life solutions.

#### Dataset and test images: [https://idd.insaan.iiit.ac.in/](https://idd.insaan.iiit.ac.in/dataset/download/) .

### Task 1:

To train the detection model or test using pre-trained weights for the  YoloV8 detection model, run the:

` python m4mts_task_1.py
`
or 
`cls_c4mts_task_2.py
`


### Task 2:

To train the classification model or test using pre-trained weights for the YoloV8 detection model, run the 
      ` python class_yolo8.py`




`Baseline code available at: https://github.com/vibhugupta1/C4MTS-Task2-Baseline`
`Baseline code available at https://github.com/ananditajam/C4MTS-task-1-baseline`


## Cite: 
    @inproceedings{gupta2023c4mts,
      title={C4MTS: Challenge on Categorizing Missing Traffic Signs from Contextual Cues},
      author={Gupta, Varun and Nagar, Sandeep and Choudhury, Suman Paul and Singh, Rohit and Jamwal, Anandita and Gupta, Vibhu and Subramanian, Anbumani and Jawahar, CV and Saluja, Rohit},
      booktitle={National Conference on Computer Vision, Pattern Recognition, Image Processing, and Graphics},
      pages={141--154},
      year={2023},
      organization={Springer}
    }
