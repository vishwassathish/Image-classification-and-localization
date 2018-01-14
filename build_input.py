from pickle import load, dump

#Import the dataset in whatever format you need
f = open('imgname_classname.txt', 'rb')
imgname_classname = load(f) #2 cols-> JPG image name and class name
f.close()


f = open('imgname_bndbox.txt', 'rb')
imgname_bndbox = load(f) #2 cols -> JPG image name and 4-tuple of bounded box
f.close()

f = open('imgname_bndbox_classname.txt', 'rb')
imgname_bndbox_classname = load(f) #3 cols -> JPG image name, 4-tuple of bounded box and class name
f.close()

f = open('imgarr_classonehot.txt', 'rb')
imgarr_classonehot = load(f) #2 cols -> image np array of dim 128, 128, 3 and np array of one hot vector (15)
f.close()

f = open('imgarr_bndbox.txt', 'rb')
imgarr_bndbox = load(f) #2 cols-> image np array of dim 128, 128, 3 and np array 4-tuple of bounded box
f.close()

f = open('imgarr_bndbox_classonehot.txt', 'rb')
imgarr_bndbox_classonehot = load(f) #3 cols-> image np array of dim 128, 128, 3 and np array 4-tuple of bounded box and np array of one hot vector(15)
f.close()

print(list(map(len, [imgname_classname, imgname_bndbox, imgname_bndbox_classname, imgarr_classonehot, imgarr_bndbox, imgarr_bndbox_classonehot])))

for item in [imgname_classname, imgname_bndbox, imgname_bndbox_classname, imgarr_classonehot, imgarr_bndbox, imgarr_bndbox_classonehot] :
	print(item[0])

# x_master = []
# y_master = []
# z_master = []
# for x, y, z in imgarr_bndbox_classonehot :
# 	x_master.append(x)
# 	y_master.append(y)
# 	z_master.append(z)

# dataset = [x_master, y_master, z_master]
# f = open('dataset.txt', 'wb')
# dump(dataset, f)
# f.close()