import numpy as np 
import cv2, math
from os import mkdir, path
from random import randint, uniform, shuffle
import matplotlib.pyplot as plt
import csv
import os, sys

# folder = 'ADCB'
# folder = 'DABC'
# folder = 'DCAB'
my_dict = {'A':0, 'B':1, 'C':2, 'D':3}


def noisy(image, noise_typ, var, amount):
    if noise_typ == "gauss":
        row,col= image.shape
        ch = 1
        mean = 0
        #var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        ch = 1
        s_vs_p = var
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
        out[coords] = 1
        
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
        out[coords] = 0

        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy



def makedata(folder,cur=0):
    tot_imgs = []

    folder = folder+str(cur+1)
    os.mkdir('frames/'+folder,0755)

    for i in np.arange(0,1000):
        fonts = [cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_SCRIPT_COMPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC]
        font = fonts[randint(0,len(fonts)-1)]
        color_in = randint(0,100)
        color_out = randint(200,255)
        pt1_x = randint(2,8)
        pt1_y = randint(12,18)
        var_sp = np.random.uniform(0.3,0.7)
        var_gauss = np.random.uniform(0.0,0.2)
        amount = np.random.uniform(0.001,0.01)
        image = np.ones((32,32), np.uint8)
        image *= color_out
        cv2.putText(image, folder[0], (pt1_x, pt1_y), font, uniform(0.6,0.9), color_in, randint(1,3))
        image = noisy(image, 'gauss', var_gauss, amount)
        cv2.imwrite('frames/'+folder+'/' + str(i).zfill(5)  +".png", image)
        tot_imgs.append(['frames/'+folder+'/'+ str(i).zfill(5)  +".png", my_dict[folder[0]]])
    for i in np.arange(1000,2000):
        fonts = [cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_SCRIPT_COMPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC]
        font = fonts[randint(0,len(fonts)-1)]
        color_in = randint(0,100)
        color_out = randint(200,255)
        pt1_x = randint(2,8)
    	pt1_y = randint(12,18)
    	var_sp = np.random.uniform(0.3,0.7)
    	var_gauss = np.random.uniform(0.0,0.2)
    	amount = np.random.uniform(0.001,0.01)
    	image = np.ones((32,32), np.uint8)
    	image *= color_out
    	cv2.putText(image, folder[1], (pt1_x, pt1_y), font, uniform(0.6,0.9), color_in, randint(1,3))
    	image = noisy(image, 'gauss', var_gauss, amount)
    	cv2.imwrite('frames/'+folder+'/' + str(i).zfill(5)  +".png", image)
    	tot_imgs.append(['frames/'+folder+'/' + str(i).zfill(5)  +".png", my_dict[folder[1]]])
    for i in np.arange(2000,3000):
        fonts = [cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_SCRIPT_COMPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC]
        font = fonts[randint(0,len(fonts)-1)]
        color_in = randint(0,100)
        color_out = randint(200,255)
        pt1_x = randint(2,8)
        pt1_y = randint(12,18)
        var_sp = np.random.uniform(0.3,0.7)
        var_gauss = np.random.uniform(0.0,0.2)
        amount = np.random.uniform(0.001,0.01)
        image = np.ones((32,32), np.uint8)
        image *= color_out
        cv2.putText(image, folder[2], (pt1_x, pt1_y), font, uniform(0.6,0.9), color_in, randint(1,3))
        image = noisy(image, 'gauss', var_gauss, amount)
        cv2.imwrite('frames/'+folder[2]+'/' + str(i).zfill(5)  +".png", image)
        tot_imgs.append(['frames/'+folder[2]+'/' + str(i).zfill(5)  +".png", my_dict[folder[2]]])
    for i in np.arange(3000,4000):
        fonts = [cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_SCRIPT_COMPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC]
        font = fonts[randint(0,len(fonts)-1)]
        color_in = randint(0,100)
        color_out = randint(200,255)
        pt1_x = randint(2,8)
        pt1_y = randint(12,18)
        var_sp = np.random.uniform(0.3,0.7)
        var_gauss = np.random.uniform(0.0,0.2)
        amount = np.random.uniform(0.001,0.01)
        image = np.ones((32,32), np.uint8)
        image *= color_out
        cv2.putText(image, folder[3], (pt1_x, pt1_y), font, uniform(0.6,0.9), color_in, randint(1,3))
        image = noisy(image, 'gauss', var_gauss, amount)
        cv2.imwrite('frames/'+folder+'/' + str(i).zfill(5)  +".png", image)
        tot_imgs.append(['frames/'+folder+'/' + str(i).zfill(5)  +".png", my_dict[folder[3]]])
    w_train = csv.writer(open('frames/train'+folder+'.txt', 'wb'), delimiter=' ')
    w_test = csv.writer(open('frames/test'+folder+'.txt', 'wb'), delimiter=' ')
    for t in tot_imgs[:2000]:
        w_train.writerow(t)
    for t in tot_imgs[2000:]:
        w_test.writerow(t)

folder = 'ABCD'

for make in np.arange(0,2):
    makedata(folder,make)