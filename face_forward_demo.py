"""
Detect if a face is looking forward.

Use landmarks returned by mediapipe's facemesh API to determine
which way the face is looking. Print out True if the face is looking
forward, otherwise false (displayed alongside video playback.)
"""
import mediapipe as mp
import cv2
import numpy as np



# Initialize MediaPipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

# Set up the webcam
cap = cv2.VideoCapture(0)


# Main event loop.
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # Process the image.
    results = face_mesh.process(image)

    # Get image size information.
    img_h , img_w, img_c = image.shape

    # Collect the 2D and 3D landmarks.
    face_2d = []
    face_3d = []

    # If there are landmarks (face detected), continue
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                    if idx ==1:
                        nose_2d = (lm.x * img_w,lm.y * img_h)
                        nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                    x,y = int(lm.x * img_w),int(lm.y * img_h)

                    face_2d.append([x,y])
                    face_3d.append(([x,y,lm.z]))


            # Get 2D coordinates
            face_2d = np.array(face_2d, dtype=np.float64)

            # Get 3D coordinates
            face_3d = np.array(face_3d,dtype=np.float64)

            # Calculate the orientation of the face.
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length,0,img_h/2],
                                [0,focal_length,img_w/2],
                                [0,0,1]])
            distortion_matrix = np.zeros((4,1),dtype=np.float64)

            success, rotation_vec, translation_vec = \
                cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

            # Get the rotational vector of the face.
            rmat, jac = cv2.Rodrigues(rotation_vec)

            angles, mtxR, mtxQ ,Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Check which way the face is oriented.
            if y < -3: # Looking Left
                looking_forward = False

            elif y > 3: # Looking Right
                looking_forward = False

            elif x < -3: # Looking Down
                looking_forward = False

            elif x > 7: # Looking Up
                looking_forward = False

            else: # Looking Forward
                looking_forward = True

    # Annotate the frame.
    cv2.putText(frame, f"{looking_forward}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 0),
                2,
                cv2.LINE_AA)
    
    # Display the frame.
    cv2.imshow("Is the face looking forward?", frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
