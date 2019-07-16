import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import os

# # using mpl
# img = mpimg.imread("English/Fnt/Sample001/img001-00001.png")
# img = 1-img
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()
# 
# use opencv
# img = cv2.imread("English/Fnt/Sample001/img001-00001.png")
# cv2.imshow("image",img)
# cv2.waitKey(0)
# # resize and save image
# img_resized = cv2.resize(img, (28,28))
# cv2.imshow("image resized", img_resized)
# cv2.waitKey(0)   
# cv2.imwrite("resized.png", img_resized)

# resizing all image in directory
# path = "English/Fnt/Sample001"
# for infile in glob.glob( os.path.join(path, '*.png') ):
#     print "current file is: " + infile
#     img = cv2.imread(infile)
#     img_resized = cv2.resize(img, (28,28))
#     img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(infile, img_gray)


# path = "English/Fnt/Sample001"
# for infile in glob.glob( os.path.join(path, '*.png') ):
#     print "current file is: " + infile
#     img = mpimg.imread(infile)
#     print img.shape


# iterate through all files in the directory to resize
# and convert image to grayscale
# dir = "English/Fnt"
# subdirs = [x[0] for x in os.walk(dir)]                                                                            
# for subdir in subdirs:                                                                                            
#         files = os.walk(subdir).next()[2]                                                                             
#         if (len(files) > 0):                                                                                          
#             for file in files:
#                 print file
#                 img = cv2.imread(subdir + "/" + file)
#                 img_resized = cv2.resize(img, (28,28))
#                 img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
#                 cv2.imwrite(subdir + "/" + file, img_gray)

def resize_and_gray(path):
    subdirs = [x[0] for x in os.walk(path)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:
                print file
                img = cv2.imread(subdir + "/" + file)
                img_resized = cv2.resize(img, (15,15))
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(subdir + "/" + file, img_gray)

# saving image to numpy array
# path = "English/Fnt/Sample001"
# f=open('img.txt','ab')
# for infile in glob.glob( os.path.join(path, '*.png') ):
#     print "current file is: " + infile
#     img = mpimg.imread(infile)
#     img = img.flatten()
#     print img.shape
#     np.savetxt(f,img[None])
# f.close()
                
def img_to_array(img_path, save_name ):
    '''
    A function to convert the png format into numpy array and 
    save them onto hard drive as .txt file
    '''
    f=open(save_name,'ab')
    for infile in glob.glob( os.path.join(img_path, '*.png') ):
        print "current file is: " + infile
        img = mpimg.imread(infile)
        img = img.flatten()
        np.savetxt(f,img[None])
    f.close()

# img_to_array("English/Fnt/Sample001", "images_001.txt")

# for all the folder, convert all the pics to array and save to hard disk
# dir = "English/Fnt"
# subdirs = [x[0] for x in os.walk(dir)]                                                                            
# for subdir in subdirs:                                                                                            
#         files = os.walk(subdir).next()[2]        
#         img_to_array(subdir,subdir + ".txt")

sample = np.loadtxt("English/Fnt/Sample001.txt")
print( sample.shape)
    

        

    
    