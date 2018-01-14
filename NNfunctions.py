from pickle import load
import numpy as np
from PIL import Image, ImageDraw
import os
import keras.backend as K
one_hot = {'person': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'tvmonitor': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'bird': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'train': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'aeroplane': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'horse': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'cow': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 'cat': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), 'dog': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), 'boat': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 'car': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]), 'bicycle': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 'motorbike': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), 'bottle': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 'chair': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])}
inv_hot = {0: 'person', 1: 'tvmonitor', 2: 'bird', 3: 'train', 4: 'aeroplane', 5: 'horse', 6: 'cow', 7: 'cat', 8: 'dog', 9: 'boat', 10: 'car', 11: 'bicycle', 12: 'motorbike', 13: 'bottle', 14: 'chair'}

def get_dataset(root_dir) : 
	f = open(os.path.join(root_dir, 'imgarr_bndbox_classonehot.txt'), 'rb')
	imgarr_bndbox_classonehot = load(f)
	#3 cols-> image np array of dim 128, 128, 3 and np array 4-tuple of bounded box and np array of one hot vector(15)
	f.close()

	dataset = np.array(imgarr_bndbox_classonehot)
	indices = np.arange(dataset.shape[0])
	np.random.shuffle(indices)
	dataset = dataset[indices]

	x_master, y_master, z_master = [], [], []
	#image np array of dim 128, 128, 3 and np array 4-tuple of bounded box and np array of one hot vector(15)
	for x, y, z in dataset :
	    x_master.append(x); y_master.append(y); z_master.append(z)
	print(list(map(len, [x_master, y_master, z_master])))

	del dataset, imgarr_bndbox_classonehot
	x_master = np.array(x_master)
	y_master = np.array(y_master)
	z_master = np.array(z_master)
	return x_master, y_master, z_master

#code for splitting data


def get_one_hot(category) :	
	return one_hot[category]

def get_inv_hot(index) :	
	return inv_hot[index]


# BOUNDING BOX FUNCTIONS
def draw_bnd_box(img_arr, p) :
    tmp_img_arr = 255 * img_arr
    tmp_img_arr = np.int_(tmp_img_arr)
    tmp_img_arr = np.asarray(tmp_img_arr, dtype = 'uint8')
    im = Image.fromarray(tmp_img_arr, 'RGB')
    draw = ImageDraw.Draw(im)
    draw.rectangle([(p[0], p[1]), (p[2], p[3])], outline = 'green')
    im.show()
    

#load_img

def load_img(img_filename, bnd_box = (0, 0, 0, 0)):
    MAX_X = MAX_Y = 128
    
    tmp_img = Image.open(img_filename)
    bnd_box = scale_bnd_box_cord(bnd_box, tmp_img.size)
    tmp_img.thumbnail((128, 128), Image.ANTIALIAS)
    
    img_arr = np.array(tmp_img)
    
    img_x, img_y = img_arr.shape[0], img_arr.shape[1]
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
    img_arr = img_arr / 255.0
    
    return (img_arr, bnd_box)
    
def predict_bnd_box(img_filename, model) :
    img_arr, dummy = load_img(img_filename)
    ans = model.predict(np.array([img_arr]))
    print(ans[0])
    ans[0] = list(map(int, ans[0]))
    draw_bnd_box(img_arr, ans[0])
    
def scale_bnd_box_cord(p, dim) :
    scale_fac = 128 / max(dim[0], dim[1])
    rv = [int(x * scale_fac) for x in p]
    return np.array(rv)

def predict_class(model, filename) :
    img_arr, dummy = load_img(filename)
    l = [img_arr]
    l = np.array(l)
    ans = model.predict(l)
    print(ans[0])
    return inv_hot[np.argmax(ans[0])]

#CONFUSION MATRIX

num_class = 15
def make_confusion_matrix(model, x_val, y_val) :
    cols = ['Predicted "' + y  + '"' for y in one_hot.keys()]
    rows = ['Expected "' + y  + '"' for y in one_hot.keys()]
    print('CONFUSION MATRIX')
    print(np.array(list(range(15))))
    matrix =  [[0 for x in range(num_class)] for y in range(num_class)]
    #rows are expected, cols are predicted
    n = len(x_val)
    for i in range(n) :
        tmp = [x_val[i]]
        tmp = np.array(tmp)
#         y_val[i] = np.array([y_val[i]])
        exp = np.argmax(y_val[i])
        pre = np.argmax(model.predict(tmp))
        matrix[exp][pre] += 1

    # for i in range(num_class) :
    #     print(i, matrix[i], sep = "\t")
    # #print('\t\t', list(one_hot.keys()))
    print(np.matrix(matrix))
    metrics = [[0 for x in range(3)] for y in range(15)]
    #print(np.matrix(metrics))
    k = []
    for i in range(num_class) :
        metrics[i][0] = "{0:.4f}".format(get_accuracy(matrix, i))
    #print('ACCURACY : ', k)
    
    k = []
    for i in range(num_class) :
        metrics[i][1] = "{0:.4f}".format(get_precision(matrix, i))
    #print('PRECISION : ', k)
    
    k = []
    for i in range(num_class) :
        metrics[i][2] = "{0:.4f}".format(get_recall(matrix, i))
    #print('RECALL : ', k)
    print(inv_hot)
    q = ['ACCURACY', 'PRECISION', 'RECALL']
    metrics.insert(0, q)
    print(np.matrix(metrics))

def get_accuracy(matrix, item) :
    correct = matrix[item][item]
    total = 0
    for i in range(num_class) :
        total += matrix[i][item]
    if total == 0 :
        return 0
    return correct/total
def  get_recall(matrix, item):
    correct = matrix[item][item]
    total = 0
    for i in range(num_class) :
        total += matrix[i][i]
    if total == 0 :
        return 0
    return correct /  total

def get_precision(matrix, item) :
    correct = matrix[item][item]
    total = sum(matrix[item])
    if total == 0 :
        return 0
    return correct/total

def precision(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1))) 
    precision = true_positives / (predicted_positives + K.epsilon()) 
    return precision

def recall(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) 
    recall = true_positives / (possible_positives + K.epsilon()) 
    return recall 

'''
accuracy = correct predictions / total expected A
recall = correct predictions / (all correct predictions i.e. diagonals)  TP / (TP + FN)
precision = correct predictions / total predictions  TP / (TP + FP)
'''
