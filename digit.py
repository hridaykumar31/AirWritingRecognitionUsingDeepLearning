import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# # Load your dataset (use appropriate dataset)
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()  # Use EMNIST for letters

# # Preprocess the data
# X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
# X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# # Build the model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     MaxPooling2D((2, 2)),
#     Dropout(0.2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Dropout(0.2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(10, activation='softmax')  # 10 classes for MNIST digits, adjust for EMNIST
# ])
# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

def plot_sample_images(X_train, y_train):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.show()

plot_sample_images(X_train, y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')  
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_test, y_pred_classes, target_names=[str(i) for i in range(10)])
print("Classification Report:\n")
print(report)

conf_matrix = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save("F:/model/cnn_model.h5")

# Arrays to handle color points
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]


blue_index = 0
green_index = 0
red_index = 0

kernel = np.ones((5, 5), np.uint8)


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
colorIndex = 0


line_thickness = 12

# Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
cv2.putText(paintWindow, "PREDICT", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "CLEAR", (510, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ret = True
while ret:
    
    ret, frame = cap.read()

    x, y, c = frame.shape

    
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (500, 1), (585, 65), (0, 0, 0), 2)
    cv2.putText(frame, "PREDICT", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "CLEAR", (510, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    
    result = hands.process(framergb)

    
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)
        if thumb[1] - center[1] < 30:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
        elif center[1] <= 65:
            if 40 <= center[0] <= 140:  
                
                drawing_area_gray = cv2.imread("drawing_area_gray.png", cv2.IMREAD_GRAYSCALE)
                if np.any(drawing_area_gray):
                    contours, _ = cv2.findContours(drawing_area_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        cropped_image = drawing_area_gray[y + 1:y + h - 1, x + 1:x + w - 1]
                        resized_image = cv2.resize(cropped_image, (28, 28))

                        
                        if np.mean(resized_image) >= 250:
                            print("Background is full white. Skipping prediction.")
                        else:
                            
                            plt.figure()
                            plt.imshow(resized_image, cmap='gray')
                            plt.title('Cropped Grayscale Image')
                            plt.axis('off')
                            plt.show()

                            
                            resized_image = resized_image.astype('float32') / 255.0
                            resized_image = resized_image.reshape(1, 28, 28)  

                           
                            predictions = model.predict(resized_image)
                            predicted_digit = np.argmax(predictions)

                            
                            char_mapping = {
                                0: 'Zero',
                                1: 'One',
                                2: 'Two',
                                3: 'Three',
                                4: 'Four',
                                5: 'Five',
                                6: 'Six',
                                7: 'Seven',
                                8: 'Eight',
                                9: 'Nine'
                            }
                            predicted_char = char_mapping.get(predicted_digit, 'Unknown')

                            
                            print(f"Predicted Digit: {predicted_digit} ({predicted_char})")

                           
                            cv2.putText(frame, f"Predicted Digit: {predicted_digit} ({predicted_char})", (10, 400),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            elif 160 <= center[0] <= 255:
                colorIndex = 0  # Blue
            elif 275 <= center[0] <= 370:
                colorIndex = 1  # Green
            elif 390 <= center[0] <= 485:
                colorIndex = 2  # Red
            elif 500 <= center[0] <= 585: 
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:] = 255
        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)

    
    points = [bpoints, gpoints, rpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], line_thickness)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], line_thickness)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)


    drawing_area = paintWindow[67:, :]

   
    cv2.imshow("Drawing Area", drawing_area)
    drawing_area_resized = cv2.resize(drawing_area, (200, 200))

    
    drawing_area_resized = np.uint8(drawing_area_resized)
    
    
    drawing_area_gray = cv2.cvtColor(drawing_area_resized, cv2.COLOR_BGR2GRAY)
    
   
    inverted_gray = 255 - drawing_area_gray
    cv2.imshow("Resized Grayscale Image", inverted_gray)
    
    cv2.imwrite("drawing_area_gray.png", inverted_gray)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('='):  
        line_thickness += 1
    elif key == ord('-'): 
        line_thickness = max(1, line_thickness - 1)  


cap.release()
cv2.destroyAllWindows()
