import face_recognition
import cv2
# import numpy as np
from src.mysql_operations import fetch_employee_image, mysql_config, get_employee_name
from src.eye_blink_detection import eye_aspect_ratio   
import mysql.connector
from imutils import face_utils
import time
from logger_config import logger
import os
import cv2
from imutils import face_utils
from flask import render_template,current_app
import socket
from app import ppoNo



mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'jay@gajanan123',
    'database': 'liveness_detection'
}

conn = mysql.connector.connect(
    host=mysql_config['host'],
    user=mysql_config['user'],
    password=mysql_config['password'],
    database=mysql_config['database']
)

cursor = conn.cursor()

def get_ip_address():
    try:
        # Create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Connect to a remote server (doesn't matter which)
        s.connect(("8.8.8.8", 80))
        
        # Get the local IP address
        ip_address = s.getsockname()[0]
        
        # Close the socket
        s.close()
        
        return ip_address
    except Exception as e:
        print("Error:", e)
        return None
    
ip_address=get_ip_address()
# def get_video_capture():
#     try:
#         cap = cv2.VideoCapture(get_local_ip())  # Adjust the index based on your camera configuration
#         if not cap.isOpened():
#             raise RuntimeError("Could not open camera. Please make sure the camera is connected.")
#         return cap
#     except Exception as e:
#         print(f"Error initializing camera: {str(e)}")
#         return None
    
def get_video_capture(ppoNo, ip_address, port=8000):
    try:
        
        # Construct the URL for the remote camera feed
        camera_url = f"http://{ip_address}:{port}/{ppoNo}/video"
        
        # Initialize VideoCapture with the remote camera URL
        cap = cv2.VideoCapture(camera_url)
        
        # Check if the camera is opened successfully
        if not cap.isOpened():
            raise RuntimeError("Could not open camera. Please make sure the camera is connected.")
        
        return cap
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
        return None  

    
    
# def release_camera():
#     cap = get_video_capture(ppoNo, ip_address, port=8000)
#     if cap is not None:
#         release_camera()



# def recognize_employee(verified_frame, known_face, EmpCode, mysql_config):
#     # Fetch the known face image from the database
#     known_face = fetch_employee_image(EmpCode, mysql_config)

#     # Check if known_face is None (i.e., face image not found in the database)
#     if known_face is None:
#         logger.error("Face image not found in the database for employee: %s", EmpCode)
#         return [], None  # Return empty list and None for matches and face_dist

#     # Log the shape of the known face image
#     logger.info("Known face shape: %s", known_face.shape)

#     # Get face locations and encodings for the known face image
#     face_known_frame = face_recognition.face_locations(known_face)
#     encode_known_face = face_recognition.face_encodings(known_face, face_known_frame)

#     # Check if no faces are found in the known face image
#     if not encode_known_face:
#         logger.error("No faces found in the known face image for employee: %s", EmpCode)
#         return [], None  # Return empty list and None for matches and face_dist

#     # Resize and convert the current frame for processing
#     img_s = cv2.resize(verified_frame, (0, 0), None, 0.25, 0.25)
#     img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

#     # Get face locations and encodings for the current frame
#     face_cur_frame = face_recognition.face_locations(img_s)
#     encode_cur_frame = face_recognition.face_encodings(img_s, face_cur_frame)

#     # Check if no faces are detected in the current frame
#     if not encode_cur_frame:
#         logger.warning("No faces detected in the current frame for employee: %s", EmpCode)
#         return [], None  # Return empty list and None for matches and face_dist

#     for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
#         # Compare face encodings and calculate face distances
#         matches = face_recognition.compare_faces(encode_known_face, encode_face, tolerance=0.5)
#         face_dist = face_recognition.face_distance(encode_known_face, encode_face)

#         logger.info("matches: %s", matches)
#         logger.info("data type of matches: %s", type(matches[0]))
#         logger.info("face_dist: %s", face_dist)

#         if matches and matches[0]:
#             # Face match found
#             # Perform further actions or return relevant information
#             logger.info("Face match found for employee: %s", EmpCode)
#             break

#     # Check if no matches are found
#     if not any(matches):
#         logger.warning("No face matches found for employee: %s", EmpCode)

#     return matches, face_dist

# def recognize_employee(verified_frame, known_face, EmpCode, mysql_config):
#     # Fetch the known face image from the database
#     known_face = fetch_employee_image(EmpCode, mysql_config)

#     # Check if known_face is None (i.e., face image not found in the database)
#     if known_face is None:
#         logger.error("Face image not found in the database for employee: %s", EmpCode)
#         return [], None  # Return empty lists for matches and face_dist
        
        

#     # Log the shape of the known face image
#     logger.info("Known face shape: %s", known_face.shape)

#     # Get face locations and encodings for the known face image
#     face_known_frame = face_recognition.face_locations(known_face)
#     encode_known_face = face_recognition.face_encodings(known_face, face_known_frame)

#     # Check if no faces are found in the known face image
#     if not encode_known_face:
#         logger.error("No faces found in the known face image for employee: %s", EmpCode)
#         return [], None  # Return empty lists for matches and face_dist

#     # Check if verified_frame is None or empty
#     if verified_frame is None or verified_frame.size == 0:
#         logger.error("Verified frame is empty")
#         return [], None  # Return empty lists for matches and face_dist

#     # Resize and convert the current frame for processing
#     img_s = cv2.resize(verified_frame, (0, 0), None, 0.25, 0.25)
#     img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

#     # Get face locations and encodings for the current frame
#     face_cur_frame = face_recognition.face_locations(img_s)
#     encode_cur_frame = face_recognition.face_encodings(img_s, face_cur_frame)

#     # Check if no faces are detected in the current frame
#     if not encode_cur_frame:
#         logger.warning("No faces detected in the current frame for employee: %s", EmpCode)
#         return [], None  # Return empty lists for matches and face_dist

#     matches = []
#     face_dist = []

#     for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
#         # Compare face encodings and calculate face distances
#         matches = face_recognition.compare_faces(encode_known_face, encode_face, tolerance=0.5)
#         face_dist = face_recognition.face_distance(encode_known_face, encode_face)

#         logger.info("matches: %s", matches)
#         logger.info("data type of matches: %s", type(matches[0]))
#         logger.info("face_dist: %s", face_dist)

#         if matches and matches[0]:
#             # Face match found
#             # Perform further actions or return relevant information
#             logger.info("Face match found for employee: %s", EmpCode)
#             break

#     # Check if no matches are found
#     if not any(matches):
#         logger.warning("No face matches found for employee: %s", EmpCode)

#     return matches, face_dist


def recognize_employee(verified_frame, known_face, EmpCode, mysql_config):
    # Fetch the known face image from the database
    known_face = fetch_employee_image(EmpCode, mysql_config)

    # Check if known_face is None (i.e., face image not found in the database)
    if known_face is None:
        logger.error("Face image not found in the database for employee: %s", EmpCode)
        return [], None  # Return empty lists for matches and face_dist
        

    # Log the shape of the known face image
    logger.info("Known face shape: %s", known_face.shape)

    # Get face locations and encodings for the known face image
    face_known_frame = face_recognition.face_locations(known_face)
    encode_known_face = face_recognition.face_encodings(known_face, face_known_frame)

    # Check if no faces are found in the known face image
    if not encode_known_face:
        logger.error("No faces found in the known face image for employee: %s", EmpCode)
        return [], None  # Return empty lists for matches and face_dist

    # Check if verified_frame is None or empty
    if verified_frame is None or verified_frame.size == 0:
        logger.error("Verified frame is empty")
        return [], None  # Return empty lists for matches and face_dist

    # Resize and convert the current frame for processing
    img_s = cv2.resize(verified_frame, (0, 0), None, 0.25, 0.25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    # Get face locations and encodings for the current frame
    face_cur_frame = face_recognition.face_locations(img_s)
    encode_cur_frame = face_recognition.face_encodings(img_s, face_cur_frame)

    # Check if no faces are detected in the current frame
    if not encode_cur_frame:
        logger.warning("No faces detected in the current frame for employee: %s", EmpCode)
        return [], None  # Return empty lists for matches and face_dist

    matches = []
    face_dist = []

    for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
        # Compare face encodings and calculate face distances
        matches = face_recognition.compare_faces(encode_known_face, encode_face, tolerance=0.5)
        face_dist = face_recognition.face_distance(encode_known_face, encode_face)

        logger.info("matches: %s", matches)
        logger.info("data type of matches: %s", type(matches[0]))
        logger.info("face_dist: %s", face_dist)

        if matches and matches[0]:
            # Face match found
            # Perform further actions or return relevant information
            logger.info("Face match found for employee: %s", EmpCode)
            break

    # Check if no matches are found
    if not any(matches):
        logger.warning("No face matches found for employee: %s", EmpCode)
        # Return empty lists for matches and face_dist
        return [], None

    return matches, face_dist




    
def generate_frames(detector, predictor, EmpCode, mysql_config):
    
    required_blinks = 3
    consecutive_frames = 0
    blink_count = 0
    EYE_AR_CONSEC_FRAMES = 5
    EYE_AR_THRESH = 0.2
    BLINK_DELAY = 1
    last_blink_time = time.time()

    ear = 0.0  # Initialize ear variable
    verified_frame = None  # Initialize the variable to store the verified frame
    cap = get_video_capture()

    match_found = False  # Initialize match_found before the loop

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

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
                if consecutive_frames >= EYE_AR_CONSEC_FRAMES and time.time() - last_blink_time > BLINK_DELAY:
                    blink_count += 1
                    consecutive_frames = 0
                    last_blink_time = time.time()
            else:
                consecutive_frames = 0

        # Display blink count in each frame
        cv2.putText(frame, f"Blink count: {blink_count}/{required_blinks}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        # Break the loop if the required blinks are achieved
        if blink_count >= required_blinks:
            verified_frame = frame
            break

    cap.release()
    known_face = fetch_employee_image(EmpCode, mysql_config)

    if known_face is None:
        return "error", "Employee image not found in the database."

    consecutive_matches_threshold = 1   
    consecutive_matches_count = 0  # Initialize consecutive_matches_count

    for _ in range(5):
        matches, face_dist = recognize_employee(verified_frame, known_face, EmpCode, mysql_config)

        # Check if at least one face is detected
        if len(matches) > 0 and face_dist[0] < 0.55:
            consecutive_matches_count += 1

            if consecutive_matches_count == consecutive_matches_threshold:
                employee_name = get_employee_name(EmpCode, mysql_config)
                match_found = True
                break
        else:
            consecutive_matches_count = 0

    if match_found:
        if len(matches) > 1:
            # Multiple faces detected
            return "error", "Multiple faces detected. Please ensure only one face is visible for verification.", None
        else:
            folder_path = r"C:\Users\gssaw\inten\finalnew\static\images"
            image_filename = f"{EmpCode}_{int(time.time())}.jpg"
            image_path = os.path.join(folder_path, image_filename)
            cv2.imwrite(image_path, verified_frame)

            insert_query = """
            INSERT INTO blink_result (EmpCode, FirstName, LastName, ver_img, ver_time)
            VALUES (%s, %s, %s, %s, NOW())
            """
            cursor.execute(insert_query, (EmpCode, employee_name.split(" ")[0], employee_name.split(" ")[1], image_path))
            conn.commit()

            print(f"Employee verification successful for {employee_name}")
            return "success", employee_name
            
    else:
        cap.release()
        # release_camera()
        # return render_template('result.html', error_msg='Facial recognition failed')
        return "error",'Facial recognition failed'





if __name__=="__main__":
    cap = cv2.VideoCapture(0)
   
    recognize_employee(cap,mysql_config)  # Replace mysql_config with your actual MySQL configuration




