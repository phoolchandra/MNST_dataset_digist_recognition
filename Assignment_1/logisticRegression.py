import time
stime = time.time()

import struct as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import array 


train_file = {'train_images' : 'train-images.idx3-ubyte' ,'train_labels' : 'train-labels.idx1-ubyte', 'test_images' : 'test-images.idx3-ubyte' ,'test_labels' : 'test-labels.idx1-ubyte'}

train_labels_array = np.array([])

data_types = {
        0x08: ('ubyte', 'B', 1),
        0x09: ('byte', 'b', 1),
        0x0B: ('>i2', 'h', 2),
        0x0C: ('>i4', 'i', 4),
        0x0D: ('>f4', 'f', 4),
        0x0E: ('>f8', 'd', 8)}

for name in train_file.keys():
	if name == 'train_images':
		train_imagesfile = open(train_file[name],'r+')
	if name == 'train_labels':
		train_labelsfile = open(train_file[name],'r+')
	if name == 'test_images':
		test_imagesfile=open(train_file[name],'r+')
	if name == 'test_labels':
		test_labelsfile = open(train_file[name],'r+')


train_imagesfile.seek(0)
magic = st.unpack('>4B',train_imagesfile.read(4))
if(magic[0] and magic[1]) or (magic[2] not in data_types):
	raise ValueError("File Format not correct")

test_imagesfile.seek(0)
magic2 = st.unpack('>4B',test_imagesfile.read(4))
if(magic2[0] and magic2[1]) or (magic2[2] not in data_types):
        raise ValueError("File Format not correct")

nDim = magic[3]
print ("Data is ",nDim,"-D")
nDim2 = magic2[3]
print ("Data is ",nDim2,"-D")

#offset = 0004 for number of images
#offset = 0008 for number of rows
#offset = 0012 for number of columns
#32-bit integer (32 bits = 4 bytes)
train_imagesfile.seek(4)
nImg = st.unpack('>I',train_imagesfile.read(4))[0] #num of images/labels
nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of columns
nBytes = nImg*nR*nC
train_labelsfile.seek(8) #Since no. of items = no. of images and is already read
print ("no. of trainimages :: ",nImg)
print ("no. of trainrows :: ",nR)
print ("no. of traincolumns :: ",nC)

test_imagesfile.seek(4)
nImg2 = st.unpack('>I',test_imagesfile.read(4))[0] #num of images/labels
nR2 = st.unpack('>I',test_imagesfile.read(4))[0] #num of rows
nC2 = st.unpack('>I',test_imagesfile.read(4))[0] #num of columns
nBytes2 = nImg2*nR2*nC2
test_labelsfile.seek(8) #Since no. of items = no. of images and is already read
print ("no. of testimages :: ",nImg2)
print ("no. of testrows :: ",nR2)
print ("no. of testcolumns :: ",nC2)

#Read all data bytes at once and then reshape
train_images_array = 255 - np.asarray(st.unpack('>'+'B'*nBytes,train_imagesfile.read(nBytes))).reshape((nImg,nR*nC))
train_labels_array = np.asarray(st.unpack('>'+'B'*nImg,train_labelsfile.read(nImg)))

test_images_array = 255 - np.asarray(st.unpack('>'+'B'*nBytes2,test_imagesfile.read(nBytes2))).reshape((nImg2,nR2*nC2))
test_labels_array = np.asarray(st.unpack('>'+'B'*nImg2,test_labelsfile.read(nImg2)))

# c=0;
# for i in test_labels_array:
# 	c=c+1
# 	print(i)
# 	if(c==5):
# 		break

# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(test_images_array[0:5], test_labels_array[0:5])):
#     plt.subplot(1, 5, index + 1)
#     plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
#     plt.title('Training: %i\n' % label, fontsize = 20)

# plt.savefig("plot", dpi=150)

# for (image, label) in enumerate(zip(train_images_array, train_labels_array)):
# 	c=c+1
# 	print(image,label)
# 	if(c==5):
# 		break



print (train_labels_array)
print (train_labels_array.shape)
print (train_images_array.shape)

print (test_labels_array)
print (test_labels_array.shape)
print (test_images_array.shape)



x = list()
y=list()


v=0
for i in range(1,50,10):
	x.append(i)
	logisticRegr = LogisticRegression(C=i)
	logisticRegr.fit(train_images_array, train_labels_array)
	score = logisticRegr.score(test_images_array,test_labels_array)
	y.append(score)



plt.ylabel('Accuracy')
plt.xlabel('C')
plt.plot(x,y)
plt.title('C vs Accuracy')
plt.savefig('CvsAccuracy_0to100.png')
plt.close()

for i in y:
	print(i)



# # for i in range(10000):
# # 	result=logisticRegr.predict(test_images_array[i].reshape(1,-1))
# # 	if(result == test_labels_array[i]):
# # 		c=c+1;

# # print("Accuracy -",c)
	

# #print(logisticRegr.predict(test_images_array[30].reshape(1,-1)))
# score = logisticRegr.score(test_images_array,test_labels_array)

# print("Score",score)

# # label=test_labels_array[0]
# # print(label)

# # plt.figure(figsize=(20,4))

# # # for index, (image, label) in enumerate(zip(test_images_array[0], test_labels_array[0])):
# #     # plt.subplot(1, 5, index + 1)

# plt.imshow(np.reshape(test_images_array[30], (28,28)), cmap=plt.cm.gray)
# plt.title('Training: %i\n' % test_labels_array[30], fontsize = 20)
# # plt.savefig('plot.png')
# plt.savefig("plot", dpi=150)


print ("Time of execution : %s seconds" % str(time.time()-stime))