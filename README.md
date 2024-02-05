# PedVision-YOLOX-DeepSort-MTAN
## Description:
Welcome to PedVision, an advanced computer vision project for pedestrian tracking and attribute recognition. Leveraging the power of YOLOX for precise object detection, DeepSort for robust tracking and a MTANet for PAR.
**Dataset:** https://mivia.unisa.it/par2023/  
**Paper:** [PedVisionPaper.pdf](https://github.com/Bobolsky/PedVision-YOLOX-DeepSort-MTAN/files/14166879/PedVisionPaper.pdf)

## Key Features:

- Detection: YOLOX Object Detector  
- Tracking: DeepSort Algorithm  
- Attribute Recognition: MultiTaskAttentionNetwork (MTAN)  
  - Backbone: Resenet 50  

## Attributes Recognized:

1. **Gender:** Gender: the considered values are male and female, represented in this order with the values [0,1].
2. **Upper & Lower Color:** Color of the clothes (upper and lower): the considered values are black, blue,     
     brown, gray, green, orange, pink, purple, red, white, and yellow and are represented, in this order, with 
     the labels [1,2,3,4,5,6,7,8,9,10,11].
3. **Hat Presence:** We consider the absence or presence of a hat, representing it with the values [0,1].
4. **Bag Presence:** We consider the absence or presence of a bag, representing it with the values [0,1].

Other than those attributes we have the ROIs logic. In our system we keep track and display some general info about those:
- ROI 1 Passages: it is the total number of people passages in ROI 1
- ROI 2 Passage: it is the total number of people passages in ROI 2
- People in ROI: it is the total number of people currently present in ROI 1 and ROI 2.
- Total Person: it is the total number of people in the scene

## Usage
```cmd
python group08.py --video test.mp4 --configuration configuration.json --results results.txt
```
The above command sees the necessity of specifying three parameters: --video , which delineates the path
to the video file designated for testing; --configuration ,
which points to the JSON file containing configurations for
ROIs; and --results , which identifies the file path where
the generated outputs are to be stored.

## Results
![image](https://github.com/Bobolsky/PedVision-YOLOX-DeepSort-MTAN/assets/36935215/0d857bd8-f63a-4373-b247-98b52d437d7e)  
*Results on the training set (Legend: U = Upper Color, L = Lower Color, G = Gender, B = Bag, H = Hat).*  
![image](https://github.com/Bobolsky/PedVision-YOLOX-DeepSort-MTAN/assets/36935215/52383326-6ab4-4d0d-adbc-f3402445351d)  
*Results on the validation set (Legend: U = Upper Color, L = Lower Color, G = Gender, B = Bag, H = Hat).*  
### Example  
![image](https://github.com/Bobolsky/PedVision-YOLOX-DeepSort-MTAN/assets/36935215/1b8072d9-040f-4163-802d-fcd7d6d597a6)  
*Example of application of the system*



