import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import collections
from collections import defaultdict
import random
import csv
import re


all_images_path="./all_images"
faces_path="./faces"

if os.path.exists(all_images_path) == False:
	os.mkdir("./all_images")

if os.path.exists(faces_path) == False:
	os.mkdir("./faces")

part1_path="./part1"
part2_path="./part2"
part3_path="./part3"
age_path="./age_labels.csv"
gender_path="./gender_labels.csv"

print(len(os.listdir(part1_path)))
print(len(os.listdir(part2_path)))
print(len(os.listdir(part3_path)))

#UNCOMMENT THE FOLLOWING SCETION TO COPY ALL IMAGES TO A COMMON FOLDER
# src1=os.listdir(part1_path)
# for idx,file_names in enumerate(os.listdir(part1_path)):
# 	path_name=os.path.join(part1_path,file_names)
# 	if os.path.isfile(path_name):
# 		shutil.copy(path_name,all_images_path)

# src2=os.listdir(part2_path)
# for idx,file_names in enumerate(os.listdir(part2_path)):
# 	path_name=os.path.join(part2_path,file_names)
# 	if os.path.isfile(path_name):
# 		shutil.copy(path_name,all_images_path)

# src3=os.listdir(part3_path)
# for idx,file_names in enumerate(os.listdir(part3_path)):
# 	path_name=os.path.join(part3_path,file_names)
# 	if os.path.join(path_name):
# 		shutil.copy(path_name,all_images_path)


assert len(os.listdir(all_images_path))==len(os.listdir(part2_path))+len(os.listdir(part1_path))+len(os.listdir(part3_path))

#SEGMENT_FACES_FUNCTION
CASCADE_PATH='haarcascade_frontalface_default.xml'
cascade_object=cv2.CascadeClassifier(CASCADE_PATH)

# for i,j in enumerate(os.listdir(all_images_path)):
# 	path=os.path.join(all_images_path,j)
# 	original_image=cv2.imread(path).astype(np.uint8)
# 	original_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
# 	gray_image=cv2.cvtColor(original_image,cv2.COLOR_RGB2GRAY)
# 	faces=cascade_object.detectMultiScale(gray_image,scaleFactor=1.1,minNeighbors=10)
# 	#cv2.imwrite("./faces/image%d.jpg" % i,original_image)
# 	for (x, y, w, h) in faces:
# 		r = min(w, h) / 2
# 		centerx=x+w/ 2
# 		centery=y+h/2
# 		nx=int(centerx-r)
# 		ny=int(centery-r)
# 		nr=int(r * 2)
# 		faceimg=original_image[ny+1:ny+nr-1, nx+1:nx+nr-1]
# 		lastimg=cv2.resize(faceimg, (224,224))
# 		cv2.imwrite("./faces/"+j, cv2.cvtColor(lastimg,cv2.COLOR_RGB2BGR))
# assert len(os.listdir(faces_path)) == len(os.listdir(all_images_path))

# cnt1=0
# cnt2=0
# bad_images_paths=[]
# for i,j in enumerate(os.listdir(faces_path)):
# 	path=os.path.join(faces_path,j)
# 	image=cv2.imread(path).astype(np.float32)
# 	h,w,ch=image.shape
# 	if (h!=224 and w!=224):
# 		cnt2+=1
# 		bad_images_paths.append(path)
# 	else:
# 		cnt1+=1

# assert (len(bad_images_paths) == (cnt2))
# print("Number of Bad Images:",cnt2)
# print("Number of good Images:",cnt1)

def age_to_cat(y):
    if y>=0 and y<=4:
        return(0)
    if y>=5 and y<=12:
        return(1)
    if y>=13 and y<=20:
        return(2)
    if y>=21 and y<=24:
        return(3)
    if y>=25 and y<=30:
        return(4)
    if y>=31 and y<=38:
        return(5)
    if y>=39 and y<=45:
        return(6)
    if y>=46 and y<=54:
        return(7)
    if y>=55 and y<=65:
        return(8)
    if y>=65:
        return(9)

age_labels=collections.defaultdict(lambda:())
gender_labels=collections.defaultdict(lambda:())
for i,j in enumerate(os.listdir(faces_path)):
	path=os.path.join(faces_path,j)
	age=j.split("_")[0]
	age=int(age)
	age_label=age_to_cat(age)
	age_labels[j,age_label]
	gender=j.split("_")[1].split("_")[0]
	gender_labels[j,gender]
	#print(path)

print(age_labels)

with open("age_labels.csv","w") as f1:
	writer=csv.writer(f1)
	for key in age_labels:
		writer.writerow([key[0],key[1]])
		#f1.write("%s %s \n"%(key,age_labels[key]))

with open("gender_labels.csv","w") as f2:
	writer=csv.writer(f2)
	for key in gender_labels.keys():
		writer.writerow([key[0],key[1]])
		#f1.write("%s %s \n"%(key,age_labels[key])

