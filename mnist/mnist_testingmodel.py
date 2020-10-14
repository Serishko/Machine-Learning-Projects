import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

model = keras.models.load_model(r"C:\Users\santo\PycharmProjects\MachineLearning\my models\mnist model\mnist")
model.load_weights(r"C:\Users\santo\PycharmProjects\MachineLearning\my models\mnist model\mnist_w")
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

#This image contains hand-written numbers for test purpose
image = cv2.imread(r'C:\Users\santo\Desktop\numbers.jpg')
grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grey.copy(), 78, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
preprocessed_digits = []; rect_area = []; k = []; dimension = []; new_processed_digits = []; l = []; modified_processed_digits = []; a = []


for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    rect_area.append(w*h)
    dimension.append(x)

    # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    # Cropping out the digit from the image corresponding to the current contours in the for loop
    digit = thresh[y:y + h, x:x + w]

    # Resizing that digit to (18, 18)
    resized_digit = cv2.resize(digit, (18, 18))

    # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
    padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

    # Adding the preprocessed digit to the list of preprocessed digits
    preprocessed_digits.append(padded_digit)

sorted_dim = [j for j in sorted(dimension)]
for i in range(len(dimension)):
    for j in range(len(dimension)):
        if sorted_dim[i] == dimension[j]:
            l.append(j)

for i in l:
    new_processed_digits.append(preprocessed_digits[i])

max_area = max(rect_area)
for index, j in enumerate(rect_area):
    if j<(max_area/2):
        k.append(index)

preprocessed_digits = [j for index, j in enumerate(preprocessed_digits) if index not in k]

for j in new_processed_digits:
    for i in preprocessed_digits:
        if np.array_equal(j, i):
            modified_processed_digits.append(j)


plt.imshow(image, cmap="gray")
plt.show()

inp = np.array(modified_processed_digits)

for digit in modified_processed_digits:
    prediction = model.predict(digit.reshape(1, 28, 28, 1))
    # plt.imshow(digit.reshape(28, 28), cmap="gray")
    # plt.show()
    # print("\n\nFinal Output: {}".format(np.argmax(prediction)))
    a.append(np.argmax(prediction))

for i in a:
    print("The predicted numbers are: ", i)
