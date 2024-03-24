from flask import Flask, render_template, jsonify, Response, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from src.facial_recog_working import release_camera, get_video_capture, recognize_employee
from src.mysql_operations import fetch_employee_image, mysql_config, get_employee_name
from src.eye_blink_detection import eye_aspect_ratio
import cv2
import dlib
from imutils import face_utils
import mysql.connector
import time
import os
import warnings
from logger_config import logger
app = Flask(__name__)
socketio = SocketIO(app)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def initialize_camera():
    return get_video_capture()



# Mock function for successful verification handling
def handle_successful_verification(ppoNo, verified_frame, longitude, latitude, mysql_config):
    employee_name = get_employee_name(ppoNo, mysql_config)
    emp_name = f"{employee_name[2]} {employee_name[3]}"
    
    # Store the verification image in a folder
    folder_path = r"C:\Users\gssaw\inten\finalnew\static\images"
    image_filename = f"{ppoNo}_{int(time.time())}.jpg"
    image_path = os.path.join(folder_path, image_filename)
    cv2.imwrite(image_path, verified_frame)
    
    ver_status = 'success'
    
    # Insert verification result into the database
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()

    insert_query = """
    INSERT INTO verification_result (`pensionId`, `ppoNo`, `firstName`, `lastName`, `middlestName`, `ver_img_path`, `ver_time`, `longitude`, `latitude`,`ver_status`)
    VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s,%s)
    """
    cursor.execute(insert_query,(employee_name[0], ppoNo, employee_name[2], employee_name[3], employee_name[4], image_path, longitude, latitude,ver_status))

    cursor.close()
    connection.commit()
    connection.close()

    success_message = f'Verification successful for {emp_name}'
    print(success_message)
    logger.info(f'Verification successful for {emp_name}')
    verification_result = {'result_status': 'success', 'message': success_message, 'employee_name': employee_name}

    return verification_result 
verified_frame=None
def perform_verification(ppoNo, verified_frame, mysql_config):
    """
    Perform the face verification process and return the result.
    """
    # Retrieve employee details for given ppoNo from the database
    employee_name = get_employee_name(ppoNo, mysql_config)
    
    if employee_name is None:
        return None, None, None
    
    emp_name = f"{employee_name[2]} {employee_name[3]}"

    # Fetch the known face image from the database
    known_face = fetch_employee_image(ppoNo, mysql_config)

    # Perform face recognition
    matches, face_dist = recognize_employee(verified_frame, known_face, ppoNo, mysql_config)

    return matches, face_dist, emp_name

@socketio.on('start_verification')
def handle_start_verification(data):
    ppoNo = data.get('ppoNo')
    print("Starting verification for employee:", ppoNo)
    
    # Initialize camera
    cap = initialize_camera()
    
    # Perform face verification
    matches, face_dist, emp_name = perform_verification(ppoNo, cap, detector, predictor, mysql_config)
    longitude=123.456
    latitude=123.456
    # Send verification result to client
    if matches and matches[0] and face_dist and face_dist[0] < 0.55:
        verification_result = handle_successful_verification(ppoNo, verified_frame, longitude, latitude, mysql_config)
        last_verification_time[ppoNo] = time.time()  # Update last verification time
        # Emit event to inform client of successful verification and include verification result
        socketio.emit('verification_success', {'result_status': 'Verification successful for ' + emp_name, 'verification_result': verification_result}, broadcast=True)
    else:
        verification_result = handle_failed_verification(ppoNo, verified_frame, mysql_config)
        # Emit event to inform client of failed verification and include verification result
        socketio.emit('verification_failed', {'result_status': 'Verification failed for ' + emp_name, 'verification_result': verification_result}, broadcast=True)

# Mock function for failed verification handling
def handle_failed_verification(ppoNo, verified_frame, mysql_config):
    employee_name = get_employee_name(ppoNo, mysql_config)
    emp_name = f"{employee_name[2]} {employee_name[3]}"

    # Store the verification image in a folder
    folder_path = r"C:\Users\gssaw\inten\finalnew\static\images"
    image_filename = f"{ppoNo}_{int(time.time())}.jpg"
    image_path = os.path.join(folder_path, image_filename)
    cv2.imwrite(image_path, verified_frame)
    
    ver_status = 'fail'
    longitude=123.456
    latitude=123.456
    # Insert verification result into the database
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()

    insert_query = """
    INSERT INTO verification_result (`pensionId`, `ppoNo`, `firstName`, `lastName`, `middlestName`, `ver_img_path`, `ver_time`, `longitude`, `latitude`, `ver_status`)
    VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s)
    """
    cursor.execute(insert_query, (employee_name[0], ppoNo, employee_name[2], employee_name[3], employee_name[4], image_path, longitude, latitude, ver_status))

    cursor.close()
    connection.commit()
    connection.close()

    failure_message = f'Verification failed for {emp_name}'
    logger.info(f'Verification failed for {emp_name}')
    print(failure_message)
    verification_result = {'result_status': 'fail', 'message': failure_message, 'employee_name': emp_name}

    return verification_result

last_verification_time = {}

# Route to render the verification page
@app.route('/video_feed')
def video_feed():
    return render_template("verification.html")

@app.route('/')
def index_n():
    return render_template('form.html')

# Client requests the verification result
@app.route('/result/<int:ppoNo>')
def result(ppoNo):
    return jsonify({'message': 'Verification result for employee {ppoNo}'})

if __name__ == '__main__':
    app.config['SECRET_KEY'] = 'xyz'
    socketio.run(app, debug=True, host='0.0.0.0', port=8000)
