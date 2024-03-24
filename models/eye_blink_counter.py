
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
# import wget
# import time
import dlib
import cv2
import os


import cv2
import dlib
from imutils import face_utils
import random

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to generate a random number of blinks between 4 and 7
def generate_random_blink_count():
    return random.randint(4, 7)

# Function to perform blink verification
def verify_blink(cap, detector, predictor):
    required_blinks = generate_random_blink_count()
    blink_count = 0
    consecutive_frames = 0
    EYE_AR_CONSEC_FRAMES = 3  # Adjust as needed

    print("Please perform {} random blinks for verification.".format(required_blinks))

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Accessing facial landmarks using dlib's default indices
            leftEye = shape[36:42]
            rightEye = shape[42:48]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                consecutive_frames += 1
                if consecutive_frames >= EYE_AR_CONSEC_FRAMES:
                    blink_count += 1
                    consecutive_frames = 0
            else:
                consecutive_frames = 0

            cv2.putText(frame, "Blink count: {}/{}".format(blink_count, required_blinks), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Blink Verification", frame)
        key = cv2.waitKey(1) & 0xFF

        if blink_count >= required_blinks:
            print("Employee verified!")
            break

        if key == ord("q"):
            break

    return blink_count >= required_blinks

# Main function
def main():
    global EYE_AR_THRESH
    EYE_AR_THRESH = 0.26
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(0)

    if verify_blink(cap, detector, predictor):
        print("Verification successful!")
    else:
        print("Verification failed. Please try again.")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
