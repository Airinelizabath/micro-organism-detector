# micro-organism-detector

This repo contains code to train and run a CNN to be able to detect the precense of malaria in blood smears.
We used  tensorflow and opencv to do the above mentioned  project.

File nn3.py contains code to train and save a model using the cell images found at the following link.
Dataset : https://lhncbc.nlm.nih.gov/LHC-downloads/dataset.html

File nn4.py can be used to load the stored model file and then we can use it to test input images. Sample images have been provided. Please edit the path to model and images before running code. 

Our final aim was to implement and run this project on a raspberry pi. The code to run in on a pi is found in file final.py. 

Detailed explaination can be found in our paper 'Convolutional Neural Network Based Mobile Microorganism Detector' by Shaji, Airin Elizabath (et al.), Proceedings of International Conference on Intelligent Computing, Information and Control Systems, ICICCS 2020, Pages 509-519. Link: https://www.springer.com/in/book/9789811584428
