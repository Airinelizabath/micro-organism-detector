import os
import cv2
import math
import tflearn
import numpy as np
import tensorflow as tf
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from RPLCD import CharLCD
from RPi import GPIO
from gpiozero import Button
from time import sleep
from gpiozero import LED
lcd = CharLCD(numbering_mode=GPIO.BOARD, cols=16, rows=2, pin_rs=37, pin_e=35, pins_data=[33, 31, 29, 23])
button = Button(2)
led = LED(3)

while(True):
    lcd.clear()
    sleep(1)
    lcd.write_string('Welcome.')
    led.on()
    while True:
        if button.is_pressed:
            break
        
    lcd.clear()
    sleep(1)
    led.off()
    lcd.write_string('Processing..')
    #delete old files
    folder = '/home/pi/Documents/project/slide/cropped'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    folder = '/home/pi/Documents/project/slide/cropped/new100'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    #camera
    cap=cv2.VideoCapture(0)
    ret,raw_image=cap.read()
    cap.release()
    
    #contouring code
    bilateral_filtered_image = cv2.bilateralFilter(raw_image, 5, 175, 175)
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (len(approx) < 23) & (area > 170)):
            contour_list.append(contour)
    i=1
    for cnt in contour_list:
        mask = np.zeros(raw_image.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
        pixelpoints = np.transpose(np.nonzero(mask))
        pp0len=pixelpoints.shape[0]
        j=0
        x,y,w,h = cv2.boundingRect(cnt)
        reimg = np.zeros([w+10,h+10,3], dtype=int)
        while(j<pp0len):
            reimg[pixelpoints[j][1]-x+5,pixelpoints[j][0]-y+5,0]=raw_image[pixelpoints[j][0],pixelpoints[j][1],0]
            reimg[pixelpoints[j][1]-x+5,pixelpoints[j][0]-y+5,1]=raw_image[pixelpoints[j][0],pixelpoints[j][1],1]
            reimg[pixelpoints[j][1]-x+5,pixelpoints[j][0]-y+5,2]=raw_image[pixelpoints[j][0],pixelpoints[j][1],2]
            j+=1    
        if(i%2==0):
            cv2.imwrite('/home/pi/Documents/project/slide/cropped/cellCroped'+str((int)(i/2))+'.png',reimg)
        i=i+1

    #above new100 folder il ile ela pic ne 100x100 size akum
    os.chdir('/home/pi/Documents/project/slide/cropped')
    i=0
    for f in os.listdir():
        file_name,ext=os.path.splitext(f)
        if(ext=='.png'):
            i=i+1
            print(i)
            file_name+=ext
            img = cv2.imread(file_name,1)
            height, width = img.shape[:2]
            if(height>width):
                ph=100/height
                h=height*ph
                w=width*ph
                hfloor=math.floor(h)
                hceil=math.ceil(h)
                if(hfloor==100):
                    h=hfloor
                    w=math.floor(w)
                else:
                    h=hceil
                    w=math.ceil(w)   
                h=int(h)
                w=int(w)         
                reimg = cv2.resize(img,(w,h))
                reimg1 = np.zeros([100,100,3], dtype=int)
                reimg1[0:reimg.shape[0], 0:reimg.shape[1]]=reimg
                cv2.imwrite('/home/pi/Documents/project/slide/cropped/new100/'+file_name,reimg1)
            else:
                pw=100/width
                h=height*pw
                w=width*pw
                wfloor=math.floor(w)
                wceil=math.ceil(w)
                if(wfloor==100):
                    w=wfloor
                    h=math.floor(h)
                else:
                    w=wceil
                    h=math.ceil(h)   
                h=int(h)
                w=int(w)         
                reimg = cv2.resize(img,(w,h))
                reimg1 = np.zeros([100,100,3], dtype=int)
                reimg1[0:reimg.shape[0], 0:reimg.shape[1]]=reimg
                cv2.imwrite('/home/pi/Documents/project/slide/cropped/new100/'+file_name,reimg1)

    #cnn define and load daved model
    IMG_SIZE = 100
    LR = 1e-3

    def create_model():
        tf.reset_default_graph()
        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
        model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=3)
        return model

    model=create_model()
    model.load('/home/pi/Documents/project/model1/model.tfl')

    #all 100x100 size pics ne pass akum through cnn. if any one is infected then infected display akum
    os.chdir('/home/pi/Documents/project/slide/cropped/new100')
    flag=0
    for f in os.listdir():
        file_name,ext=os.path.splitext(f)
        if(ext=='.png'):
            file_name += ext
            img_data = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
            model_out = model.predict([data]) [0]
            if np.argmax(model_out) == 0:
                flag=1	
    
    #lcd display code
    lcd.clear()
    sleep(1)
    if(flag==0):
        lcd.write_string('Not Infected.')
    else:
        lcd.write_string('Malaria Detected!')
    sleep(10)
    lcd.clear()
    sleep(1)