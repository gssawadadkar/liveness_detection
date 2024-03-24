from flask import Flask, render_template, g,request,jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import face_recognition
import cv2
import dlib
from src.face_recog import  get_video_capture, generate_frames, release_camera #, recognize_employee
from src.mysql_operations import fetch_employee_image, mysql_config, get_employee_name
from src.eye_blink_detection import eye_aspect_ratio, EYE_AR_THRESH, generate_random_blink_count
import mysql.connector
from imutils import face_utils
from logger_config import logger
import time
import os
import warnings
import numpy as np
import base64


# from pyngrok import ngrok
# import ngrok

# To ignore all warnings
warnings.filterwarnings("ignore")

app = Flask(__name__,static_url_path='', static_folder='static')
CORS(app)
api = Api(app)



parser = reqparse.RequestParser()
parser.add_argument('EmpCode', type=str, help='Employee Code is required', required=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

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

def is_webcam_open():
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (webcam)

    if not cap.isOpened():
        logger.error("Webcam not opened!")
        return False

    # Release the capture object
    cap.release()
    logger.debug("Webcam successfully opened!")
    return True


def recognize_employee(verified_frame, known_face, EmpCode, mysql_config):
    known_face = fetch_employee_image(EmpCode, mysql_config)
    logger.info("Known face shape: %s", known_face.shape)
    face_known_frame = face_recognition.face_locations(known_face)
    encode_known_face = face_recognition.face_encodings(known_face, face_known_frame)
    if not encode_known_face:
        # No faces found in the known face image
        # release_camera()
        return [], []
    while True:
        img_s = cv2.resize(verified_frame, (0, 0), None, 0.25, 0.25)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
        face_cur_frame = face_recognition.face_locations(img_s)
        encode_cur_frame = face_recognition.face_encodings(img_s, face_cur_frame)
        
        if not encode_cur_frame:
            # No faces detected in the current frame
            continue

        for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
            matches = face_recognition.compare_faces(encode_known_face, encode_face, tolerance=0.5)
            face_dist = face_recognition.face_distance(encode_known_face, encode_face)

            logger.info("matches:", matches)
            logger.info("data type of matches sssssssssssssssss:", type(matches[0]))
            logger.info("face_dist:  %s", face_dist)
            if matches and matches[0]:
                # Face match found
                # Perform further actions or return relevant information
                logger.info("Face match found for employee: %s", EmpCode)
                break
        # Release the camera here if needed
        # release_camera()

        return matches, face_dist

def verify_liveness(cap, detector, predictor):
    if not is_webcam_open():
        return False, None

    required_blinks = generate_random_blink_count()  # Adjust as needed
    blink_count = 0
    consecutive_frames = 0
    EYE_AR_CONSEC_FRAMES = 5  # Adjust as needed
    BLINK_DELAY = 1  # Adjust as needed (time in seconds)

    logger.info("Please perform {} blinks for liveness verification.".format(required_blinks))

    last_blink_time = time.time()
    verification_frame = None

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[36:42]
            rightEye = shape[42:48]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            # Draw bounding box around the face
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw contours around the eyes
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

            cv2.putText(frame, "Blink count: {}/{}".format(blink_count, required_blinks), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Liveness Verification", frame)

        # Save the frame for recognition
        verification_frame = frame

        if blink_count >= required_blinks:
            logger.info("Liveness verified!")
            break

        if cv2.waitKey(1) & 0xFF == 27:  # Break the loop if 'Esc' key is pressed
            break

    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    if blink_count >= required_blinks:
        return True, verification_frame
    else:
        logger.info("Liveness verification failed. Insufficient blinks.")
        return False, verification_frame


camera = None  # Define the camera variable globally

def open_webcam():
    global camera
    camera = cv2.VideoCapture(0)  # Adjust the index (0) based on your camera configuration
    if not camera.isOpened():
        return False
    return True

def is_webcam_open():
    return camera is not None and camera.isOpened()

def release_camera():
    global camera
    if camera is not None:
        camera.release()

def verify_liveness_and_recognition(detector, predictor, emp_code):
    global camera
    if not is_webcam_open():
        if not open_webcam():
            return "error", "Webcam not opened.", None

    liveness_verified, verified_frame = verify_liveness(camera, detector, predictor)

    if liveness_verified:
        consecutive_matches_threshold = 1
        consecutive_matches_count = 0
        match_found = False

        for _ in range(5):
            known_face = fetch_employee_image(emp_code, mysql_config)

            # Check if the image is found in the database
            if known_face is None:
                release_camera()
                return "error", "Employee image not found in the database.", None

            matches, face_dist = recognize_employee(verified_frame, known_face, emp_code, mysql_config)

            # Check if at least one face is detected
            if len(matches) > 0 and face_dist[0] < 0.55:
                consecutive_matches_count += 1

                if consecutive_matches_count == consecutive_matches_threshold:
                    employee_name = get_employee_name(emp_code, mysql_config)
                    match_found = True

                    break
            else:
                consecutive_matches_count = 0

        if match_found:
            if len(matches) > 1:
                # Multiple faces detected
                return "error", "Multiple faces detected. Please ensure only one face is visible for verification.", None
            else:
                release_camera()
                return "success", employee_name, verified_frame
        else:
            release_camera()
            return "error", "Employee verification failed. Please try again.", None

    else:
        release_camera()
        return "error", "Liveness verification failed.", None






def get_camera():
    if 'camera' not in g:
        g.camera = get_video_capture()
    return g.camera

@app.teardown_appcontext
def teardown_appcontext(exception=None):
    camera = g.pop('camera', None)
    if camera is not None:
        camera.release()

def release_camera():
    camera = getattr(g, 'camera', None)
    if camera is not None:
        camera.release()

# class IndexPageResource(Resource):
#     def get(self):
#         release_camera()
#         return render_template('index.html')


class LivenessVerificationResource(Resource):

    def get(self):
        return render_template('index.html')

    def post(self):
        data = request.get_json()

        if data is None:
            return {'status': 'error', 'message': 'Invalid JSON data received'}, 400

        emp_code = data.get('empCode')
        app.logger.info(f"Received POST request with EmpCode: {emp_code}")

        try:
            cap = get_camera()
            if cap is None:
                return {'status': 'error', 'message': 'Camera not initialized'}, 500

            status, employee_name, verified_frame = verify_liveness_and_recognition(detector, predictor, emp_code)

            if status == "success":
                folder_path = r"C:\Users\gssaw\inten\finalnew\static\images"
                image_filename = f"{emp_code}_{int(time.time())}.jpg"
                image_path = os.path.join(folder_path, image_filename)
                cv2.imwrite(image_path, verified_frame)

                insert_query = """
                INSERT INTO blink_result (EmpCode, FirstName, LastName, ver_img, ver_time)
                VALUES (%s, %s, %s, %s, NOW())
                """
                cursor.execute(insert_query, (emp_code, employee_name.split(" ")[0], employee_name.split(" ")[1], image_path))
                conn.commit()

                response_data = {
                    'status': 'success',
                    'employee_name': employee_name,
                    'image_path': image_path
                }

                return response_data, 200
            else:
                return {'status': 'error', 'message': employee_name}, 400

        except Exception as e:
            # Log the exception
            app.logger.error(f"Exception during liveness verification: {str(e)}")
            return {'status': 'error', 'message': str(e)}, 500

        finally:
            release_camera()


# @app.route('/')
# def index():
#     return render_template('form.html')

# Route for handling form submission and storing employee data
# @app.route('/store_employee', methods=['POST'])
# def store_employee():
#     # Extract data from the POST request
#     emp_data = request.json

#     # Insert employee data into the database
#     cursor = conn.cursor()
#     cursor.execute("""
#         INSERT INTO employees (EmpCode, FirstName, LastName, PAN, Aadhar, Designation, Department, ImageData)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#     """, (
#         emp_data['EmpCode'], 
#         emp_data['FirstName'], 
#         emp_data['LastName'], 
#         emp_data['PAN'], 
#         emp_data['Aadhar'], 
#         emp_data['Designation'], 
#         emp_data['Department'], 
#         emp_data['img_data']
#     ))
#     conn.commit()

#     cursor.close()

#     # Return success response
#     return jsonify({'message': 'Employee data stored successfully'})


# import cv2
# import numpy as np
# import base64
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# Load pre-trained cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    frame_data = request.json['frame']
    
    # Decode base64 image data and convert to OpenCV format
    frame_bytes = frame_data.split(',')[1].encode()
    nparr = np.frombuffer(base64.b64decode(frame_bytes), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # For each detected face, detect eyes
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Draw rectangles around the detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # Convert the processed frame back to base64 for sending to frontend
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Prepare data to send back to frontend
    processed_data = {
        'frame': frame_base64  # Base64 encoded image with detected faces and eyes
        # Add other relevant data if needed
    }
    
    return jsonify(processed_data)





from app_rest import VerificationHistoryResource,StoreEmployeeResource
api.add_resource(StoreEmployeeResource, '/store_employee')
api.add_resource(VerificationHistoryResource, '/get_verification_history')
# api.add_resource(IndexPageResource, '/')
api.add_resource(LivenessVerificationResource, '/liveness_verification')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/liveness_verification')
# def liveness_verification():
#     app.logger.info("Inside liveness_verification function.")
#     return render_template('access.html')

# @app.route('/access.html')
# def access_page():
#     # Render the HTML template
#     return render_template('access.html'), 200, {'Content-Type': 'text/html'}
# @app.route('/verify_liveness', methods=['POST'])
# def verify_liveness():
#     emp_code = request.form.get('EmpCode')
    
#     # Perform liveness verification logic here
#     status, result_message = verify_liveness_and_recognition(detector, predictor, emp_code)
    
#     return render_template('result.html', status=status, message=result_message)

@app.route('/store_employee')
def store_employee():
    return render_template('store_employee.html')

@app.route('/get_verification_history')  
def verification_history(EmpCode):
    # You can use EmpCode in the function if needed
    return render_template('verification_history.html', EmpCode=EmpCode)

# listener = ngrok.forward("localhost:8080", authtoken_from_env=True,
#     domain="example.ngrok.app")

# print(f"Ingress established at: {listener.url()}")

if __name__ == '__main__':
    app.run(debug=True, port=9000)