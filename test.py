from keras.models import load_model
import matplotlib.image as mpimg

model = load_model("cnn.gz")

model.summary()

im = mpimg.imread('testimage/Selection_007.png')

print( im.shape)
print( type(im))

im = im.reshape((28,28))
print( im.shape)
print( X_test[:1].shape)

# print( model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0))
# print( X_test.shape)

# print( model.predict_classes(X_test[:1] ))
print( model.predict_classes(im))
