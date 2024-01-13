# Real-Time-Number-Plate-Recognition
Author : Deepak Pant 22M0035@iitb.ac.in

# **Problem Statement:**
Real-Time Number Plate Detection and Vehicle Tracking using OpenCV

Ensuring road safety and compliance with traffic regulations requires efficient monitoring and identification of vehicles, especially at critical points such as traffic signals and highways. This project addresses this need by developing a real-time number plate detection system using OpenCV, a computer vision library. The project aims to create a robust solution that identifies and tracks vehicles, captures number plates, and provides motion prediction within dynamic road environments.

# **Project Objectives:**

The central objective of this project is to create a real-time number plate detection system capable of tracking vehicles and capturing number plates using OpenCV. By combining computer vision techniques and motion prediction, the project aims to achieve the following specific objectives:

1. **Real-Time Detection:** Develop a Python code using OpenCV to enable real-time identification and capture of car number plates. Utilize image processing techniques to isolate and extract number plate regions from live video streams.

2. **Vehicle Tracking:** Implement tracking mechanisms to acknowledge and follow vehicles as they move through the monitored area. This involves employing object tracking algorithms to maintain the continuity of tracking.

3. **Motion Prediction:** Incorporate motion prediction algorithms to estimate the trajectory of vehicles, enabling the system to predict their future positions based on their current velocities.

4. **Speed Detection:** Integrate speed detection mechanisms that calculate and report the speed of vehicles based on the time taken to traverse specific distances. This enables identification of over-speeding vehicles.

# **Importance and Implications:**

The successful completion of this project holds several significant implications:

- **Enhanced Road Safety:** The real-time number plate detection system aids in monitoring and enforcing traffic regulations, contributing to safer road environments.
- **Traffic Management:** By capturing number plates and tracking vehicle movements, the system assists traffic authorities in managing traffic flow and identifying potential congestion points.
- **Law Enforcement:** The system's capability to detect over-speeding vehicles empowers law enforcement agencies to take timely action against traffic rule violators.
- **Surveillance and Monitoring:** The project provides a valuable tool for surveillance and monitoring of road activities, helping authorities address security concerns and ensure compliance.

# **Applicability and Deployment:**

The project's outcome is applicable for deployment at critical road points, such as traffic signals, highways, and checkpoints. It provides real-time insights into vehicle movements, speed violations, and potential risks, enabling prompt response and action. The system's accuracy and real-time capabilities make it a valuable asset for traffic management and law enforcement agencies.

By leveraging OpenCV and computer vision techniques, this project aims to contribute to road safety, efficient traffic management, and improved law enforcement practices. The system's ability to identify vehicles, capture number plates, track movement, and predict motion holds the potential to revolutionize real-time monitoring in road environments.


## Tech Stack
* [openCV](https://opencv.org/): It is a library mainly used at real-time computer vision.
* [Tensorflow](https://github.com/tensorflow/models) : Here I used Tensorflow object detection Model (SSD MobileNet V2 FPNLite 320x320) to detect the plate trained on a Kaggle Dataset.
* Python Libraries: Most of the libraries are mentioned in [requirements.txt](https://github.com/harshitkd/Real-Time-Number-Plate-Recognition/blob/main/requirements.txt) but some of the libraries and requirements depends on the user's machines, whether its installed or not and also the libraries for Tensorflow Object Detection (TFOD) consistently change.
# Steps
These outline the steps I used to go through in order to get up and running with ANPR. 

### Install and Setup :

<b>Step 1.</b> Clone this repository: https://github.com/Pantd007/Car-Number-Plate-Detection
<br/><br/>
<b>Step 2.</b> Create a new virtual environment 
<pre>
python -m venv arpysns
</pre> 
<br/>
<b>Step 3.</b> Activate your virtual environment
<pre>
source tfod/bin/activate # Linux
.\arpysns\Scripts\activate # Windows 
</pre>
<br/>
<b>Step 4.</b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=anprsys
</pre>
<br/>

# Dataset: 
Used the [Car License Plate Detection](https://www.kaggle.com/andrewmvd/car-plate-detection) kaggel dataset and manually divided the collected images into two folders train and test so that all the images and annotations will be split among these two folders.

### Training Object Detection Model
I used pre-trained state-of-the-art model and just fine tuned it on our particular specific use case.Begin the training process by opening [Real Time Number Plate Detection](https://github.com/harshitkd/Real-Time-Number-Plate-Recognition/blob/main/Real%20Time%20Number%20Plate%20Detection.ipynb) and installed the Tensoflow Object Detection (TFOD) 

![68747470733a2f2f692e696d6775722e636f6d2f465351466f31362e706e67](https://user-images.githubusercontent.com/56076028/145552503-b3a442a4-03bf-467e-af74-3e218c949dad.png)

In the below image you will see the object detection model which is now trained. I have decided to train it on the terminal because the training inside a separate terminal on a windows machine displays live loss metrics.

![Screenshot (72)](https://user-images.githubusercontent.com/56076028/145536355-94f60307-3632-4bd4-9eb7-02b9c875471d.png)

* Visualization of Loss Metric, learning rate and number of steps:

<pre>
tensorboard --logdir=.
</pre>

![tensorboard loss](https://user-images.githubusercontent.com/56076028/145684910-d237be53-88d4-45fa-b36e-dd9a52daf8e1.jpg)

![tensorboard learning and steps](https://user-images.githubusercontent.com/56076028/145684923-36a95279-5b27-4f25-bd2d-ea58eaa82075.jpg)

### Detecting License Plates

![Screenshot 2021-12-10 130124](https://user-images.githubusercontent.com/56076028/145536393-986af131-ce84-4d4c-8174-735ed492a45b.jpg)


### Apply OCR to text

<pre>
import easyocr
detection_threshold=0.7
image = image_np_with_detections
scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
boxes = detections['detection_boxes'][:len(scores)]
classes = detections['detection_classes'][:len(scores)]
</pre>

![Screenshot 2021-12-10 125508](https://user-images.githubusercontent.com/56076028/145536427-d27c0fdc-cd30-446b-9b16-6408fdb4efcd.jpg)

### Results

Used this in real time to detect the license plate and stored the text in .csv file and images in the Detection_Images folder.

### Object Detection Metric:
![evaluation metric](https://user-images.githubusercontent.com/56076028/145684944-29306983-8396-47a2-9a08-f13a86d56f08.jpg)

![evaluation metric detail](https://user-images.githubusercontent.com/56076028/145684945-7f17e0b6-e623-4a71-b163-388a84d713fd.jpg)

<pre>
tensorboard --logdir=.
</pre>

![mAP](https://user-images.githubusercontent.com/56076028/145684953-51fc55d3-c9cd-4789-807e-0cfa0196000c.jpg)

![AR](https://user-images.githubusercontent.com/56076028/145684962-3236958f-4354-4230-b8d2-c59d18665b31.jpg)

