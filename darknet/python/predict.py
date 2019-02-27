from ctypes import *
import math
import random
import cv2
from darknet import *

net = load_net("../cfg/yolov3_googleship.cfg", "../backup/yolov3_googleship_300.weights", 0)
meta = load_meta("../cfg/googleship.1.data")
#detections = detect(net, meta, "../data/100000068.jpg")
#print detections[:10]


f = open('../test.txt', "r")
lines = f.readlines()
f.close()

for line in lines:
    print("proccesing..." + line + "*")
    line = line.replace('\n', '')
    detections = detect(net, meta, line)
    img = cv2.imread(line)
    name = line.split('/')[-1]
    name = name.split('.')[-2]

    # estearchivo contendra todos los bbox predecidos 
    f = open('../predicted/' + name + '.txt', "w")
    for det in detections:
        label = det[0]
        accuracy = det[1]
        box = det[2]

        width = int(box[2])
        height = int(box[3])
        xmin = int(box[0]) - width / 2
        ymin = int(box[1]) - height / 2
        xmax = int(box[0]) + width / 2
        ymax = int(box[1]) + height / 2

        f.write(str(label) + " " + str(accuracy) + " " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n")
        
        cv2.rectangle(img, (xmin, ymin), (xmin + width, ymin + height), (0, 0, 255), 2, 2)

    f.close()

    cv2.imwrite("../predicted/" + name + '.jpg', img)
    cv2.namedWindow('win',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('win', 600,600)
    cv2.imshow("win", img)
    cv2.waitKey(30)
	
    

