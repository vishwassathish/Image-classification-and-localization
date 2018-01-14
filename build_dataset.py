import os
import xml.etree.ElementTree as ET
import numpy as np
from pickle import dump
root = 'C:/Users/Varun/Desktop/VOCdevkit/VOC2010' #THE LINE YOU SHOULD CHANGE
ommited_classes = {'sheep', 'sofa', 'bus', 'diningtable', 'pottedplant'}


one_hot = {'person': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'tvmonitor': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'bird': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'train': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'aeroplane': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'horse': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'cow': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 'cat': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), 'dog': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), 'boat': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 'car': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]), 'bicycle': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 'motorbike': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), 'bottle': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 'chair': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])}
inv_hot = {0: 'person', 1: 'tvmonitor', 2: 'bird', 3: 'train', 4: 'aeroplane', 5: 'horse', 6: 'cow', 7: 'cat', 8: 'dog', 9: 'boat', 10: 'car', 11: 'bicycle', 12: 'motorbike', 13: 'bottle', 14: 'chair'}

ann_dir = os.path.join(root, 'Annotations')
ann_file_list = os.listdir(ann_dir)
class_dataset = []
box_dataset = []
master_dataset = []
for file in ann_file_list :
    e = ET.parse(root + '/Annotations/' + file).getroot()
    class_name = ""
    big_box = 0
    objcount = 0
    bndbox = []
    for child in e :
        if child.tag == 'object' :
            objcount += 1 
            for grandchild in child :
                if grandchild.tag == 'name' :
                    name = grandchild.text
                if grandchild.tag == 'bndbox' :
                    d = {}
                    for greatgrandchild in grandchild :
                        d[greatgrandchild.tag] = int(greatgrandchild.text)
                    bndbox = [d["xmin"], d["ymin"], d["xmax"], d["ymax"]]
                    class_name = name
    if objcount != 1 or  class_name in ommited_classes:
        continue
    file = file[:-3] + 'jpg'
    k = [file, tuple(bndbox), class_name]
    class_dataset.append([k[0]] + [k[2]]) #imgname, class
#    k = [file, tuple(bndbox)]
    box_dataset.append([k[0]] + [k[1]])	#imgname, boundbox
    master_dataset.append(k[:]) #imgname, boundbox tuple, class
    del k

print('works')
print(len(master_dataset))
print(master_dataset[0])
f1 = open('imgname_classname.txt', 'wb')
f2 = open('imgname_bndbox.txt', 'wb')
f3 = open('imgname_bndbox_classname.txt', 'wb')
dump(class_dataset, f1)
dump(box_dataset, f2)
dump(master_dataset, f3)
f1.close()
f2.close()
f3.close()

print("first set loaded")

from PIL import Image, ImageDraw

root_dir = 'C:/Users/Varun/Desktop/VOCdevkit/VOC2010'
img_dir = os.path.join(root_dir, 'JPEGImages')
ann_dir = os.path.join(root_dir, 'Annotations')

def scale_bnd_box_cord(p, dim) :
    #print(p, type(p))
    scale_fac = 128 / max(dim[0], dim[1])
    rv = [int(x * scale_fac) for x in p]
    return np.array(rv)

def load_img(img_filename, bnd_box):
    MAX_X = MAX_Y = 128
    
    tmp_img = Image.open(os.path.join(img_dir, img_filename))
    bnd_box = scale_bnd_box_cord(bnd_box, tmp_img.size)
    tmp_img.thumbnail((128, 128), Image.ANTIALIAS)
    
    #print(tmp_img.size)
    
    img_arr = np.array(tmp_img)
    
    img_x, img_y = img_arr.shape[0], img_arr.shape[1]
    #print(img_arr.shape)
    pad_x = (MAX_X - img_x)
    pad_y = (MAX_Y - img_y)
    odd_x = pad_x & 1
    odd_y = pad_y & 1
    pad_x //= 2
    pad_y //= 2
    
    bnd_box[1] += pad_x + odd_x
    bnd_box[3] += pad_x + odd_x
    bnd_box[0] += pad_y + odd_y
    bnd_box[2] += pad_y + odd_y
    
    img_arr = np.pad(img_arr, [(pad_x + odd_x, pad_x), (pad_y + odd_y, pad_y), (0, 0)], mode = 'constant', constant_values = 0)
    
    img_arr.astype('float32')
    img_arr = img_arr/255.0
    
    return (img_arr, bnd_box)
    
def draw_bnd_box(img_arr, p) :
    im = Image.fromarray(img_arr, 'RGB')
    draw = ImageDraw.Draw(im)
    draw.rectangle([(p[0], p[1]), (p[2], p[3])], outline = 'green')
    im.show()

class_dataset_arr = []
box_dataset_arr = []
master_dataset_arr = []
# for item in master_dataset :
# 	print(item)

print("building mainset")
for imgname, bndbox, classname in master_dataset :
	img_array, bndbox = load_img(imgname, bndbox)
	master_dataset_arr.append([img_array][:] + [bndbox][:] + [one_hot[classname]])
	class_dataset_arr.append([img_array][:] + [one_hot[classname]])
	box_dataset_arr.append([img_array][:] + [bndbox][:])

print(len(master_dataset_arr), len(class_dataset_arr), len(box_dataset_arr))
print("master", master_dataset_arr[0])
print("class", class_dataset_arr[0])
print("box", box_dataset_arr[0])

f1 = open('imgarr_classonehot.txt', 'wb')
f2 = open('imgarr_bndbox.txt', 'wb')
f3 = open('imgarr_bndbox_classonehot.txt', 'wb')
dump(class_dataset_arr, f1)
dump(box_dataset_arr, f2)
dump(master_dataset_arr, f3)
f1.close()
f2.close()
f3.close()