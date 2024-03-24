import face_recognition
import cv2
# import numpy as np
from src.mysql_operations import fetch_employee_image, mysql_config, get_employee_name
from src.eye_blink_detection import eye_aspect_ratio   # EYE_AR_THRESH, generate_random_blink_count,verify_blink
import mysql.connector
from imutils import face_utils
import time
from logger_config import logger

# from mysql_operations import fetch_employee_image, mysql_config, get_employee_name
import dlib
import os
import cv2
from imutils import face_utils
from flask import render_template,current_app
# from app import app



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


def get_video_capture():
    try:
        cap = cv2.VideoCapture(0)  # Adjust the index based on your camera configuration
        if not cap.isOpened():
            raise RuntimeError("Could not open camera. Please make sure the camera is connected.")
        return cap
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
        return None
    
    
def initialize_video_capture():
    # Replace '0' with the appropriate camera index or video file path
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Error: Could not open video capture device.")

    # Replace 'path/to/shape_predictor_68_face_landmarks.dat' with the path to your shape predictor file
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

    return detector, predictor, cap   

    
    
def release_camera():
    cap = get_video_capture()
    if cap is not None:
        release_camera()


def recognize_employee(verified_frame, known_face, EmpCode, mysql_config):
    # Fetch the known face image from the database
    known_face = fetch_employee_image(EmpCode, mysql_config)

    # Log the shape of the known face image
    logger.info("Known face shape: %s", known_face.shape)

    # Get face locations and encodings for the known face image
    face_known_frame = face_recognition.face_locations(known_face)
    encode_known_face = face_recognition.face_encodings(known_face, face_known_frame)

    # Check if no faces are found in the known face image
    if not encode_known_face:
        logger.error("No faces found in the known face image for employee: %s", EmpCode)
        return [], []

    # Resize and convert the current frame for processing
    img_s = cv2.resize(verified_frame, (0, 0), None, 0.25, 0.25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    # Get face locations and encodings for the current frame
    face_cur_frame = face_recognition.face_locations(img_s)
    encode_cur_frame = face_recognition.face_encodings(img_s, face_cur_frame)

    # Check if no faces are detected in the current frame
    if not encode_cur_frame:
        logger.warning("No faces detected in the current frame for employee: %s", EmpCode)
        return [], []

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
    # if not any(matches):
    #     logger.warning("No face matches found for employee: %s", EmpCode)

    return matches, face_dist


def process_frame(frame, detector, predictor):
    if len(frame.shape) > 2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    rects = detector(gray, 0)

    ear = 0.0

    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[36:42]
        rightEye = shape[42:48]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

    return gray, ear



def detect_blink(ear, EYE_AR_THRESH, consecutive_frames, last_blink_time, BLINK_DELAY):
    EYE_AR_CONSEC_FRAMES=3
    if ear < EYE_AR_THRESH:
        consecutive_frames += 1
        if consecutive_frames >= EYE_AR_CONSEC_FRAMES and time.time() - last_blink_time > BLINK_DELAY:
            return True, consecutive_frames, time.time()
    else:
        consecutive_frames = 0

    return False, consecutive_frames, last_blink_time


def capture_frame(cap):
    ret, frame = cap.read()
    return ret, frame


def convert_to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def encode_frame(frame):
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()


def display_text_on_frame(frame, text, position, font_size=0.7, color=(0, 0, 255), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)


def detect_blinks(ear, threshold, consecutive_frames, last_blink_time, delay):
    blink_detected, consecutive_frames, last_blink_time = detect_blink(
        ear, threshold, consecutive_frames, last_blink_time, delay
    )
    return blink_detected, consecutive_frames, last_blink_time


def process_and_yield_frame(detector, predictor, cap, required_blinks, EmpCode, mysql_config):
    consecutive_frames = 0
    blink_count = 0
    EYE_AR_CONSEC_FRAMES = 5
    EYE_AR_THRESH = 0.2
    BLINK_DELAY = 1
    last_blink_time = time.time()

    ear = 0.0
    verified_frame = None
    match_found = False

    while True:
        ret, frame = capture_frame(cap)
        gray_frame = convert_to_gray(frame)

        processed_frame, ear = process_frame(gray_frame, detector, predictor)
        blink_detected, consecutive_frames, last_blink_time = detect_blinks(
            ear, EYE_AR_THRESH, consecutive_frames, last_blink_time, BLINK_DELAY
        )

        display_text_on_frame(processed_frame, f"Blink count: {blink_count}/{required_blinks}", (10, 30))
        display_text_on_frame(processed_frame, "EAR: {:.2f}".format(ear), (300, 30))

        frame_bytes = encode_frame(frame)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n'
        )

        if blink_detected:
            blink_count += 1

        if blink_count >= required_blinks:
            verified_frame = processed_frame
            match_found = True
            break

    cap.release()
    return verified_frame, match_found


def fetch_and_validate_employee_image(EmpCode, mysql_config):
    known_face = fetch_employee_image(EmpCode, mysql_config)

    if known_face is None or known_face[0] is None:
        return "error", "Employee image not found in the database.", None

    return None, known_face



def recognize_and_verify_employee(verified_frame, known_face, EmpCode, mysql_config):
    consecutive_matches_threshold = 1
    consecutive_matches_count = 0
    match_found = False

    for _ in range(5):
        matches, face_dist = recognize_employee(
            verified_frame, known_face, EmpCode, mysql_config
        )

        if len(matches) > 0 and face_dist[0] < 0.55:
            consecutive_matches_count += 1

            if consecutive_matches_count == consecutive_matches_threshold:
                match_found = True
                break
        else:
            consecutive_matches_count = 0

    return match_found


def save_verified_image_and_database_record(EmpCode, employee_name, verified_frame):
    folder_path = r"C:\Users\gssaw\inten\finalnew\static\images"
    image_filename = f"{EmpCode}_{int(time.time())}.jpg"
    image_path = os.path.join(folder_path, image_filename)
    cv2.imwrite(image_path, verified_frame)

    insert_query = """
    INSERT INTO blink_result (EmpCode, FirstName, LastName, ver_img, ver_time)
    VALUES (%s, %s, %s, %s, NOW())
    """
    cursor.execute(
        insert_query,
        (
            EmpCode,
            employee_name.split(" ")[0],
            employee_name.split(" ")[1],
            image_path,
        ),
    )
    conn.commit()

    print(f"Employee verification successful for {employee_name}")
    return "success", employee_name


def generate_frames(detector, predictor, EmpCode, mysql_config):
    required_blinks = 3
    cap = get_video_capture()

    verified_frame, match_found = process_and_yield_frame(detector, predictor, cap, required_blinks, EmpCode, mysql_config)

    if not match_found:
        cap.release()
        return "error", 'Facial recognition failed'

    error, error_message, _ = fetch_and_validate_employee_image(EmpCode, mysql_config)
    if error is not None:
        cap.release()
        return error, error_message

    match_found = recognize_and_verify_employee(verified_frame, error, EmpCode, mysql_config)
    if not match_found:
        cap.release()
        return "error", "Facial recognition failed"

    return save_verified_image_and_database_record(EmpCode, error, verified_frame)



def process_and_yield_frame(detector, predictor, cap, required_blinks, EmpCode, mysql_config):
    consecutive_frames = 0
    blink_count = 0
    EYE_AR_CONSEC_FRAMES = 5
    EYE_AR_THRESH = 0.2
    BLINK_DELAY = 1
    last_blink_time = time.time()

    ear = 0.0
    verified_frame = None
    match_found = False

    while True:
        ret, frame = capture_frame(cap)
        gray_frame = convert_to_gray(frame)

        processed_frame, ear = process_frame(gray_frame, detector, predictor)
        blink_detected, consecutive_frames, last_blink_time = detect_blinks(
            ear, EYE_AR_THRESH, consecutive_frames, last_blink_time, BLINK_DELAY
        )

        display_text_on_frame(processed_frame, f"Blink count: {blink_count}/{required_blinks}", (10, 30))
        display_text_on_frame(processed_frame, "EAR: {:.2f}".format(ear), (300, 30))

        frame_bytes = encode_frame(processed_frame)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n'
        )

        if blink_detected:
            blink_count += 1

        if blink_count >= required_blinks:
            verified_frame = processed_frame
            break

    cap.release()
    yield verified_frame, match_found


def fetch_and_validate_employee_image(EmpCode, mysql_config):
    known_face = fetch_employee_image(EmpCode, mysql_config)

    if known_face is None or known_face[0] is None:
        return "error", "Employee image not found in the database.", None

    return None, known_face


def recognize_and_insert_result(verified_frame, EmpCode, mysql_config):
    match_found = False
    consecutive_matches_threshold = 1
    consecutive_matches_count = 0

    known_face = fetch_employee_image(EmpCode, mysql_config)

    if known_face is None:
        return "error", "Employee image not found in the database.", None

    for _ in range(5):
        matches, face_dist = recognize_employee(verified_frame, known_face, EmpCode, mysql_config)

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
        release_camera()
        return "error", 'Facial recognition failed', None
