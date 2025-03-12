"""
Demonstrate face detection.

Face detection finds the location of an image in an image.
This can be a probability over each pixel, or a predicted
bounding box. OpenCV's haarcascade classifier predicts
bounding boxes, and these are displayed in real time.

Download the model:
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

"""
import cv2



# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + \
                                     'haarcascade_frontalface_default.xml')

# Start video capture (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Convert the frame to grayscale (Haar cascades work on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30))
    
    # Draw a bounding box around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the frame with the bounding box
    cv2.imshow("Face Detection", frame)
    
    # Exit the loop if the user presses the q key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
