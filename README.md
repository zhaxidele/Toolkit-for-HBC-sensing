# Toolkit-for-HBC-sensing
This repository supplyies the basic concept of human body capacitance sening and the hardware platform, the firmware, the collected datasets from experiments that combines HBC sensing and IMU sensing for human activity recognition, and the used machine learning algorithms for human activity classification. 
More detailed description could be found in our published paper:

1, Using Human Body Capacitance Sensing to Monitor Leg Motion Dominated Activities with a Wrist Worn Device
Available at: https://abc-research.github.io/

Share and cite: 

2, Human Body Capacitance: a Novel Wearable MotionSensing Approach Besides IMU and its Contributionto Human Activity Recognition
Available at: https://www.computer.org/csdl/magazine/pc

Share and cite: 


# selected figures:

---Hardware

<img width="266" alt="Board_1" src="https://user-images.githubusercontent.com/12549420/136019960-b8bdf3e0-5ddf-455c-8c63-4a171c2dda4a.png" >

The hardware combines IMU and HBC sensing modality on a cutomised board, with ESP32 as the controller, and a 3.7V battery as the power source. 

---GymExercise

<img width="450" alt="Calos_Gym" src="https://user-images.githubusercontent.com/12549420/136020221-a32689d2-7c01-42a5-8b1c-68694b6b18b8.png" >

11 gym exercises: Adductor, Armcurl, Benchpress, Legcurl, Legpress, Riding, Ropeskipping, Running, Squat, Stairsclimber and Walking

--- TV-Wall Assembling and Dissembling

<img width="450" alt="Screenshot 2021-10-05 at 14 17 18" src="https://user-images.githubusercontent.com/12549420/136020676-b11e646e-c842-406d-bfa7-7066ad29c571.png"  >

A TV-Wall assembling and disassembling activity including both collaborative and non-collaborative activities.


--- Classifcation Result Example

<img width="600" alt="Screenshot 2021-10-05 at 14 22 43" src="https://user-images.githubusercontent.com/12549420/136021710-d045e318-1fff-47cc-8626-19b3f9f696c9.png" >

Gym Exercise classification result with sensing units on different body locations.

<img width="292" alt="Screenshot 2021-10-05 at 14 23 21" src="https://user-images.githubusercontent.com/12549420/136021780-874b06c1-089b-4d96-9c78-fdc670fe4bfe.png" >

TV-Wall building activities classification result with different configuration of sensing modalities.


