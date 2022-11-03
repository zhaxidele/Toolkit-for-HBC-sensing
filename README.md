# Toolkit-for-HBC-sensing
This repository supplyies the basic concept of human body capacitance sening and the hardware platform, the firmware, the collected datasets from experiments that combines HBC sensing and IMU sensing for human activity recognition, and the used machine learning algorithms for human activity classification. 

## Hardware

<img width="266" alt="Board_1" src="https://user-images.githubusercontent.com/12549420/136019960-b8bdf3e0-5ddf-455c-8c63-4a171c2dda4a.png" >

The hardware combines IMU and HBC sensing modality on a cutomised board, with ESP32 as the controller, and a 3.7V battery as the power source. 


## GymExercise

<img width="450" alt="Calos_Gym" src="https://user-images.githubusercontent.com/12549420/136020221-a32689d2-7c01-42a5-8b1c-68694b6b18b8.png" >

11 gym exercises: Adductor, Armcurl, Benchpress, Legcurl, Legpress, Riding, Ropeskipping, Running, Squat, Stairsclimber and Walking

## TV-Wall Assembling and Dissembling

<img width="450" alt="Screenshot 2021-10-05 at 14 17 18" src="https://user-images.githubusercontent.com/12549420/136020676-b11e646e-c842-406d-bfa7-7066ad29c571.png"  >

A TV-Wall assembling and disassembling activity including both collaborative and non-collaborative activities.


## Classifcation Result Example

### Gym Exercise Recognition: Random Forest with 615 abstracted features

<img width="900" alt="Screenshot 2021-10-05 at 14 22 43" src="https://user-images.githubusercontent.com/12549420/136021710-d045e318-1fff-47cc-8626-19b3f9f696c9.png" >

Gym Exercise classification result with sensing units on different body locations(pocket, leg, and wrist).


### Gym Exercise Recognition: Deep models result

<img width="741" alt="Screenshot 2022-06-15 at 11 14 01" src="https://user-images.githubusercontent.com/12549420/173791120-9f0ee57e-f1d6-4ffa-ab9e-3f6f92593887.png">


### Collaborative Activity Recognition (TV-Wall)

<img width="500" alt="Screenshot 2021-10-05 at 14 23 21" src="https://user-images.githubusercontent.com/12549420/136021780-874b06c1-089b-4d96-9c78-fdc670fe4bfe.png" >

TV-Wall building activities classification result with different configuration of sensing modalities.




### More detailed description could be found in our published paper:



1， A systematic study of the influence of various user specific and environmental factors on wearable Human Body Capacitance sensing

Available at: https://link.springer.com/chapter/10.1007/978-3-030-95593-9_20

Share and cite:

@inproceedings{bian2021systematic,
  title={A systematic study of the influence of various user specific and environmental factors on wearable human body capacitance sensing},
  author={Bian, Sizhen and Lukowicz, Paul},
  booktitle={EAI International Conference on Body Area Networks},
  pages={247--274},
  year={2021},
  organization={Springer}
}


2, Using Human Body Capacitance Sensing to Monitor Leg Motion Dominated Activities with a Wrist Worn Device

Available at: https://link.springer.com/chapter/10.1007/978-981-19-0361-8_5

Share and cite: 

@incollection{bian2022using,
  title={Using human body capacitance sensing to monitor leg motion dominated activities with a wrist worn device},
  author={Bian, Sizhen and Yuan, Siyu and Rey, Vitor Fortes and Lukowicz, Paul},
  booktitle={Sensor-and Video-Based Activity and Behavior Computing},
  pages={81--94},
  year={2022},
  publisher={Springer}
}


3, The Contribution of Human Body Capacitance/Body-Area Electric Field To Individual and Collaborative Activity Recognition

Available at: https://arxiv.org/abs/2210.14794

Share and cite: 

@article{bian2022contribution,
  title={The Contribution of Human Body Capacitance/Body-Area Electric Field To Individual and Collaborative Activity Recognition},
  author={Bian, Sizhen and Rey, Vitor Fortes and Yuan, Siyu and Lukowicz, Paul},
  journal={arXiv preprint arXiv:2210.14794},
  year={2022}
}


