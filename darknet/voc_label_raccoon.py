import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
import random
import numpy as np
import pandas as pd 

classes = ["raccoon"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

files = glob.glob("/home/vicente/datasets/raccoon/annotations_voc/*.xml")

for file in files:
	print (file)
	fname =	(file.split("/")[-1]).split(".")[0]

	out_file = open('/home/vicente/datasets/raccoon/annotations/%s.txt'%(fname), 'w')

	tree=ET.parse(file)
	root = tree.getroot()
	size = root.find('size')
	w = int(size.find('width').text)
	h = int(size.find('height').text)

	for obj in root.iter('object'):
		cls = obj.find('name').text
		if cls not in classes:
		    continue
		cls_id = classes.index(cls)
		xmlbox = obj.find('bndbox')
		b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
		bb = convert((w,h), b)

		out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

files = glob.glob("/home/vicente/datasets/raccoon/images/*.jpg")
np_files = np.array(files)
np.random.shuffle(np_files)

train = np_files[0:150]
val = np_files[150:175]
test = np_files[175:200]

np.savetxt('train.txt', train, delimiter=" ", fmt="%s") 
np.savetxt('val.txt', val, delimiter=" ", fmt="%s") 
np.savetxt('test.txt', test, delimiter=" ", fmt="%s") 







