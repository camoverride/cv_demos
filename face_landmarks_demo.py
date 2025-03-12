"""
Use mediapipe's face mesh to show face landmarks.
"""
import cv2
import mediapipe as mp



# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Set up the webcam
cap = cv2.VideoCapture(0)

# Initialize the face mesh model
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Flip the frame horizontally for a more natural viewing
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get face landmarks
        results = face_mesh.process(rgb_frame)

        # Draw landmarks if detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the landmarks
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

        # Display the frame with landmarks
        cv2.imshow('Face Landmarks', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
