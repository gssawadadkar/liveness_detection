from flask import Flask, render_template, jsonify, Response
from flask_cors import CORS
from flask_restful import Api
from src.facial_recog_working import get_video_capture, recognize_employee,ip_address
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
import socket

warnings.filterwarnings("ignore")

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)
api = Api(app)

mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'jay@gajanan123',
    'database': 'liveness_detection'
}


# def get_video_capture(ppono, ip_address, port=8000):
#     try:
#         # Construct the URL for the remote camera feed
#         camera_url = f"http://{ip_address}:{port}/{ppono}/video"
        
#         # Initialize VideoCapture with the remote camera URL
#         cap = cv2.VideoCapture(camera_url)
        
#         # Check if the camera is opened successfully
#         if not cap.isOpened():
#             raise RuntimeError("Could not open camera. Please make sure the camera is connected.")
        
#         return cap
#     except Exception as e:
#         print(f"Error initializing camera: {str(e)}")
#         return None  


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
def should_attempt_verification(ppoNo):
    """
    Check if enough time has passed since the last successful verification for the given employee.
    """
    global last_verification_time
    last_time = last_verification_time.get(ppoNo, 0)
    current_time = time.time()
    return current_time - last_time >= VERIFICATION_COOLDOWN_PERIOD

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

def should_attempt_verification(ppoNo):
    """
    Check if enough time has passed since the last successful verification for the given employee.
    """
    global last_verification_time
    last_time = last_verification_time.get(ppoNo, 0)
    current_time = time.time()
    return current_time - last_time >= VERIFICATION_COOLDOWN_PERIOD

def generate_frames(detector, predictor):
    global result_status
    global verified_frame
    global blink_count
    global ppoNo
    global verification_complete

    with app.app_context():
        required_blinks = 3
        consecutive_frames = 0
        blink_count = 0
        EYE_AR_CONSEC_FRAMES = 3
        EYE_AR_THRESH = 0.2
        BLINK_DELAY = 1
        last_blink_time = time.time()

        ear = 0.0
        verified_frame = None
        cap = get_video_capture(ppoNo, ip_address, port=8000)

        match_found = False

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
                        consecutive_frames = 0
                        last_blink_time = time.time()
                        blink_count += 1

                        # Print blink count information before verification
                        print(f"Third blink detected: {blink_count} blinks")

                else:
                    consecutive_frames = 0

            cv2.putText(frame, f"Blink count: {blink_count}/{required_blinks}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

            if blink_count >= required_blinks:
                verified_frame = frame
                result_status = "Verification In Progress"

                if match_found:
                    cap.release()
                    break  # Exit the loop if verification is successful
                else:
                    cap.release()
                    break
            
        cap.release()

@app.route('/access/<employeeId>')
def index(employeeId):
    global ppoNo
    if employeeId is None:
        # If employeeId is None, halt the process
        print("Employee code is None. Unable to proceed.")
        return "Employee code is None. Unable to proceed."
    ppoNo = employeeId
    print("Employee Code:", ppoNo)
    return render_template('index.html')

@app.route('/home/<employeeId>')
def home_page(employeeId):
    try:
        global ppoNo
        ppoNo = employeeId
        return render_template('home.html', ppoNo=ppoNo)
    except Exception as e:
        logger.error(f"Error rendering home.html: {str(e)}")
        return "An error occurred while rendering the home page.", 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(detector, predictor), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result():
    global verification_complete
    global verification_result
    global ppoNo
    
    if verification_complete:
        verification_complete = False  # Reset the flag for the next verification
        return jsonify(verification_result)
    
    # Retrieve latitude and longitude from the request
    latitude = 123.456  # Replace with actual request data
    longitude = 123.456  # Replace with actual request data
    print("longitude and latitude ", latitude, longitude)

    if should_attempt_verification(ppoNo):
        # Perform the verification process
        matches, face_dist, emp_name = perform_verification(ppoNo, verified_frame, mysql_config)

        if matches and matches[0] and face_dist and face_dist[0] < 0.55:
            verification_result = handle_successful_verification(ppoNo, verified_frame, longitude, latitude, mysql_config)
            last_verification_time[ppoNo] = time.time()  # Update last verification time
            verification_complete = True  # Set verification complete
            return jsonify(verification_result)  # Return success immediately

        else:
            verification_result = handle_failed_verification(ppoNo, verified_frame, mysql_config)

    verification_complete = True
        
    # Check if verification is complete
    if verification_complete:
        return jsonify(verification_result)

# Dictionary to store the last successful verification time for each employee
last_verification_time = {}

# Cooldown period in seconds
VERIFICATION_COOLDOWN_PERIOD = 60  # Adjust this as needed

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


from app_rest import VerificationHistoryResource,StoreEmployeeResource
api.add_resource(StoreEmployeeResource, '/store_employee')
api.add_resource(VerificationHistoryResource, '/get_verification_history')

    
@app.route('/store_employee/<employeeId>')
def store_employee(employeeId):
    
    return render_template('store_employee.html',ppoNo=employeeId)

# @app.route('/get_verification_history/', defaults={'parameter': None})
@app.route('/get_verification_history/<parameter>',methods=["POST"])
def verification_history(parameter):
    print("Parameter         ",parameter)
    # You can use EmpCode in the function if needed
    return render_template('verification_history.html',ppoNo=parameter)

@app.route('/verification_history/<parameter>')
def verification_history_html(parameter):
    ppoNo=parameter
    # You can use EmpCode in the function if needed
    return render_template('verification_history.html',ppoNo=ppoNo)


@app.route('/checkEmployeeId/<employeeId>')
def check_employee(employeeId):
    try:
        ppoNo=employeeId
        # Your code for storing employee data in MySQL
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()

        query = f"SELECT COUNT(*) as count_empcode FROM pensioners WHERE ppoNo = '{ppoNo}'"
        cursor.execute(query)
        result = cursor.fetchone()

        count_empcode = result[0]
        cursor.close()
        connection.close()
        
        if count_empcode > 0:
            return jsonify({'message': f'Employee  {ppoNo} exists', 'type': 'registered'})
        else:
            return jsonify({'message': f'Employee  {ppoNo} does not exist', 'type': 'not_registered'})

        
        
            
    except Exception as e:
        return jsonify({'message': str(e), 'type': 'error'})

if __name__ == '__main__':
    verification_complete = False  # Initialize verification flag
    app.run(debug=True, host='localhost', port=8000)
