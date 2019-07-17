from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import cv2

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

zero_number = np.loadtxt("char74k/Sample001.txt")
one_number = np.loadtxt("char74k/Sample002.txt")
two_number = np.loadtxt("char74k/Sample003.txt")
three_number = np.loadtxt("char74k/Sample004.txt")
four_number = np.loadtxt("char74k/Sample005.txt")
five_number = np.loadtxt("char74k/Sample006.txt")
six_number = np.loadtxt("char74k/Sample007.txt")
seven_number = np.loadtxt("char74k/Sample008.txt")
eight_number = np.loadtxt("char74k/Sample009.txt")
nine_number = np.loadtxt("char74k/Sample010.txt")
A_letter = np.loadtxt("char74k/Sample011.txt")
B_letter = np.loadtxt("char74k/Sample012.txt")
C_letter = np.loadtxt("char74k/Sample013.txt")
D_letter = np.loadtxt("char74k/Sample014.txt")
E_letter = np.loadtxt("char74k/Sample015.txt")
F_letter = np.loadtxt("char74k/Sample016.txt")
G_letter = np.loadtxt("char74k/Sample017.txt")
H_letter = np.loadtxt("char74k/Sample018.txt")
J_letter = np.loadtxt("char74k/Sample020.txt")
K_letter = np.loadtxt("char74k/Sample021.txt")
L_letter = np.loadtxt("char74k/Sample022.txt")
M_letter = np.loadtxt("char74k/Sample023.txt")
N_letter = np.loadtxt("char74k/Sample024.txt")
P_letter = np.loadtxt("char74k/Sample026.txt")
Q_letter = np.loadtxt("char74k/Sample027.txt")
R_letter = np.loadtxt("char74k/Sample028.txt")
S_letter = np.loadtxt("char74k/Sample029.txt")
T_letter = np.loadtxt("char74k/Sample030.txt")
U_letter = np.loadtxt("char74k/Sample031.txt")
V_letter = np.loadtxt("char74k/Sample032.txt")
W_letter = np.loadtxt("char74k/Sample033.txt")
X_letter = np.loadtxt("char74k/Sample034.txt")
Y_letter = np.loadtxt("char74k/Sample035.txt")
Z_letter = np.loadtxt("char74k/Sample036.txt")
 
 
zero_label = np.zeros(1016)
one_label = np.zeros(1016)
two_label = np.zeros(1016)
three_label = np.zeros(1016)
four_label = np.zeros(1016)
five_label = np.zeros(1016)
six_label = np.zeros(1016)
seven_label = np.zeros(1016)
eight_label = np.zeros(1016)
nine_label = np.zeros(1016)
 
A_label = np.zeros(1016)
B_label = np.zeros(1016)
C_label = np.zeros(1016)
D_label = np.zeros(1016)
E_label = np.zeros(1016)
F_label = np.zeros(1016)
G_label = np.zeros(1016)
H_label = np.zeros(1016)
J_label = np.zeros(1016)
K_label = np.zeros(1016)
L_label = np.zeros(1016)
M_label = np.zeros(1016)
N_label = np.zeros(1016)
P_label = np.zeros(1016)
Q_label = np.zeros(1016)
R_label = np.zeros(1016)
S_label = np.zeros(1016)
T_label = np.zeros(1016)
U_label = np.zeros(1016)
V_label = np.zeros(1016)
W_label = np.zeros(1016)
X_label = np.zeros(1016)
Y_label = np.zeros(1016)
Z_label = np.zeros(1016)
 
zero_label.fill(0)
one_label.fill(1)
two_label.fill(2)
three_label.fill(3)
four_label.fill(4)
five_label.fill(5)
six_label.fill(6)
seven_label.fill(7)
eight_label.fill(8)
nine_label.fill(9)
A_label.fill(10)
B_label.fill(11)
C_label.fill(12)
D_label.fill(13)
E_label.fill(14)
F_label.fill(15)
G_label.fill(16)
H_label.fill(17)
J_label.fill(18)
K_label.fill(19)
L_label.fill(20)
M_label.fill(21)
N_label.fill(22)
P_label.fill(23)
Q_label.fill(24)
R_label.fill(25)
S_label.fill(26)
T_label.fill(27)
U_label.fill(28)
V_label.fill(29)
W_label.fill(30)
X_label.fill(31)
Y_label.fill(32)
Z_label.fill(33)
 
 
X_train = np.vstack((zero_number[:700], one_number[:700], two_number[:700], three_number[:700], four_number[:700], five_number[:700], six_number[:700], seven_number[:700], eight_number[:700], nine_number[:700],
                    A_letter[:700], B_letter[:700], C_letter[:700], D_letter[:700], E_letter[:700], F_letter[:700], G_letter[:700], H_letter[:700], J_letter[:700], K_letter[:700], L_letter[:700], M_letter[:700], N_letter[:700],
                    P_letter[:700], Q_letter[:700], R_letter[:700], S_letter[:700], T_letter[:700], U_letter[:700], V_letter[:700], W_letter[:700], X_letter[:700], Y_letter[:700], Z_letter[:700]))
 
X_test = np.vstack((zero_number[700:], one_number[700:], two_number[700:], three_number[700:], four_number[700:], five_number[700:], six_number[700:], seven_number[700:], eight_number[700:], nine_number[700:],
                    A_letter[700:], B_letter[700:], C_letter[700:], D_letter[700:], E_letter[700:], F_letter[700:], G_letter[700:], H_letter[700:], J_letter[700:], K_letter[700:], L_letter[700:], M_letter[700:], N_letter[700:],
                    P_letter[700:], Q_letter[700:], R_letter[700:], S_letter[700:], T_letter[700:], U_letter[700:], V_letter[700:], W_letter[700:], X_letter[700:], Y_letter[700:], Z_letter[700:]))
 
Y_train = np.hstack((zero_label[:700], one_label[:700], two_label[:700], three_label[:700], four_label[:700], five_label[:700], six_label[:700], seven_label[:700], eight_label[:700], nine_label[:700],
                    A_label[:700], B_label[:700], C_label[:700], D_label[:700], E_label[:700], F_label[:700], G_label[:700], H_label[:700], J_label[:700], K_label[:700], L_label[:700], M_label[:700], N_label[:700],
                    P_label[:700], Q_label[:700], R_label[:700], S_label[:700], T_label[:700], U_label[:700], V_label[:700], W_label[:700], X_label[:700], Y_label[:700], Z_label[:700]))
 
Y_test = np.hstack((zero_label[700:], one_label[700:], two_label[700:], three_label[700:], four_label[700:], five_label[700:], six_label[700:], seven_label[700:], eight_label[700:], nine_label[700:],
                    A_label[700:], B_label[700:], C_label[700:], D_label[700:], E_label[700:], F_label[700:], G_label[700:], H_label[700:], J_label[700:], K_label[700:], L_label[700:], M_label[700:], N_label[700:],
                    P_label[700:], Q_label[700:], R_label[700:], S_label[700:], T_label[700:], U_label[700:], V_label[700:], W_label[700:], X_label[700:], Y_label[700:], Z_label[700:]))
 
 
# print X_train.shape
# print X_test.shape
# print Y_train.shape
# print Y_test.shape
X_train = 1-X_train
X_test = 1-X_test
 
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
 
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

model = Sequential()

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32*196, input_shape=(128,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(34))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

# def plotLearningCurve(figPath):
#     fig=plt.figure(0, figsize=(10,8) )
#     fig.clf()
#     plt.ioff()
#     plt.subplot(211)
#     plt.grid(True)
#     plt.plot(trn_error, label='Training Set Error', linestyle="--", linewidth=2)
#     plt.plot(tst_error, label='Validation Set Error', linewidth=2)
#     plt.title('CE Error')
#     plt.xlabel('Epoch')
#     plt.ylabel('Error')
#     plt.legend()
       
#     plt.subplot(212)
#     plt.grid(True)
#     plt.plot(trn_class_accu, label='Training Set Accuracy', linestyle="--", linewidth=2)
#     plt.plot(tst_class_accu, label='Validation Set Accuracy', linewidth=2)
#     plt.ylabel('Percent')
#     plt.xlabel('Epoch')
#     plt.title('Classification Accuracy')
#     plt.legend(loc=4)
       
#     plt.tight_layout(pad=2.1)
#     plt.savefig(figPath)

tstErrorCount = 0
oldtstError = 0
trn_error=[]
tst_error=[]
trn_class_accu=[]
tst_class_accu=[]
patience = 100
figPath = "Figure_CNN/ErrorGraph"
trnErrorPath='Figure_CNN/trn_error'
tstErrorPath='Figure_CNN/tst_error'
trnClassErrorPath='Figure_CNN/trn_ClassAccu'
tstClassErrorPath='Figure_CNN/tst_ClassAccu'
 
while (tstErrorCount<patience): 
    # train model for 1 epoch
    log = model.fit(X_train, Y_train, batch_size=128, verbose=0, validation_data=(X_test, Y_test), epochs=20)
    # monitor train loss
    print(log.history.keys())

    trnError = log.history["loss"]
    # trnAcc = log.history["acc"]
    # monitor val loss
    tstError = log.history["val_loss"]
    # tstAcc = log.history["val_acc"] 
    # append the loss and accuracy
    # trn_class_accu.append(trnAcc)
    # tst_class_accu.append(tstAcc)
    trn_error.append(trnError)
    tst_error.append(tstError)
    
    np.savetxt(trnErrorPath, trn_error)
    np.savetxt(tstErrorPath, tst_error)
    # np.savetxt(trnClassErrorPath, trn_class_accu)
    # np.savetxt(tstClassErrorPath, tst_class_accu)
     
    if(oldtstError==0):
        oldtstError = tstError
     
    if(oldtstError<tstError):
        tstErrorCount = tstErrorCount+1
        print( 'No Improvement, count=%d' % tstErrorCount)
        # print( 'Trn Acc:', trnAcc )
        # print( 'Tst Acc:', tstAcc)
#         print( '    Old Validation Error:', oldtstError )
#         print( 'Current Validation Error:', tstError)
                                                                                                                                                 
    if(oldtstError>tstError):
        print( 'Improvement made!')
        # print( 'Trn Acc:', trnAcc )
        # print( 'Tst Acc:', tstAcc)
#         print '    Old Validation Error:', oldtstError 
#         print 'Current Validation Error:', tstError
        tstErrorCount=0
        oldtstError = tstError
        model.save('model.h5')
        model.save_weights("cnn.gz", overwrite=True)
        # plotLearningCurve(figPath)
        
print( "Patience elapsed! Stopping.")
model.summary()

# model.load_weights("cnn.gz")
# im = mpimg.imread('testimage/Selection_007.png')

# print( im.shape)
# print( type(im))

# im = im.reshape((28,28))
# print( im.shape)
# print( X_test[:1].shape)

# # print( model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0))
# # print( X_test.shape)

# # print( model.predict_classes(X_test[:1] ))
# print( model.predict_classes(im))

