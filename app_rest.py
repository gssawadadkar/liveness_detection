from flask import Flask, request, render_template, jsonify
from flask_restful import Api, Resource, reqparse, fields, marshal_with
from src.facial_recog_working import recognize_employee, get_video_capture,release_camera
from flask_cors import CORS
import cv2
import dlib
from datetime import datetime
import mysql.connector
import os
import base64
import time
import importlib



app = Flask(__name__)
CORS(app)
api = Api(app)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
EYE_AR_THRESH = 1.4  # 0.23

mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'jay@gajanan123',
    'database': 'liveness_detection'
}

# Function to capture image from webcam
def capture_webcam_image():
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (you can change it if you have multiple cameras)
    ret, frame = cap.read()
    cap.release()
    return frame


class StoreEmployeeResource(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('pensionId', type=str, required=True, help='Pension Id is required')
        parser.add_argument('ppoNo', type=int, required=True, help='PPO Number is required')
        parser.add_argument('firstName', type=str, required=True, help='First Name is required')
        parser.add_argument('lastName', type=str, required=True, help='Last Name is required')
        parser.add_argument('middlestName', type=str, required=True, help='Middlest Name is required')
        # parser.add_argument('img_path', type=str, required=True, help='Image path is required')
        parser.add_argument('img_data', type=str, required=True, help='Image data is required')
        parser.add_argument('longitude', type=str, required=True, help='Longitude is required')
        parser.add_argument('latitude', type=str, required=True, help='Latitude is required')
        
        
        args = parser.parse_args()
        pensionId = args['pensionId']
        ppoNo = args['ppoNo']
        firstName = args['firstName']
        lastName = args['lastName']
        middlestName = args['middlestName']
        longitude = args['longitude']
        latitude = args['latitude']
        img_data = args["img_data"]
        print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPNNNNNNNNNNNNNN",ppoNo)
        try:
           
            ver_time=str(int(time.time()))
            
        
            # Pad the base64-encoded string if needed
            img_data_padded = img_data[21:] + '=' * ((4 - len(img_data) % 4) % 4)
            # with open("base64_data.txt", "w") as file:
            #     file.write(img_data_padded)

            # Save the image file with employee code as the filename
            image_filename = f"{ppoNo}.jpg"
            img_path = os.path.join(r"C:\Users\gssaw\inten\finalnew\static\images", image_filename)
            with open(img_path, "wb") as fh:
                fh.write(base64.decodebytes(img_data_padded.encode()))

            # Your code for storing employee data in MySQL
            connection = mysql.connector.connect(**mysql_config)
            cursor = connection.cursor()
            
            cursor.execute("""
                INSERT INTO pensioners (pensionId, ppoNo, firstName, lastName, `middlestName`, img_path, ver_time, longitude, latitude)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s)
            """, (pensionId, ppoNo, firstName, lastName, middlestName, img_path, longitude, latitude))
            
            cursor.close()
            connection.commit()
            connection.close()

            return {'status': 'success', 'message': 'Employee data stored successfully!'}

        except Exception as e:
            print(f"Error: {e}")
            return {'status': 'error', 'message': str(e)}, 500
        
        
from flask import request

# class VerificationHistoryResource(Resource):
#     def get(self):
#         try:
#             data = request.get_json()
#             if 'ppoNo' not in data:
#                 return {'error': 'ppoNo not provided in JSON'}, 400
            
#             ppoNo = data['ppoNo']
            
#             connection = mysql.connector.connect(**mysql_config)
#             cursor = connection.cursor()
#             IMAGE_FOLDER=r"C:\Users\gssaw\inten\finalnew\static\images"
#             cursor.execute("""
#                 SELECT pensionId,ppoNo, firstName, lastName, ver_time, ver_img_path
#                 FROM verification_result
#                 WHERE ppoNo = %s
#                 ORDER BY ver_time DESC
#                 LIMIT 5
#             """, (ppoNo,))

#             history = cursor.fetchall()

#             if not history:
#                 return {'message': f'No data found for EmpCode {ppoNo}'}, 404

#             # Process the history to add image paths
#             formatted_history = []
#             for record in history:
#                 ppoNo = record[1]
#                 timestamp = datetime.strptime(str(record[3]), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
#                 image_path = os.path.join(IMAGE_FOLDER, f"{ppoNo}_{timestamp}.jpg")
#                 formatted_record = {
#                     'pensionId': record[0],
#                     'ppoNo': record[1],
#                     'firstName': record[2],
#                     'lastName': timestamp,
#                     'ver_time': timestamp
#                     # "ver_img_path":ver_img_path
#                 }
#                 formatted_history.append(formatted_record)

#             return {'message': 'Last 5 history of employee', 'history': formatted_history}

#         except Exception as e:
#             return {'error': str(e)}, 500

# Add resources to the API



from datetime import datetime

class VerificationHistoryResource(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('ppoNo', type=str, required=True, help='ppoNo is required')

    def get(self):
        
        return render_template('index.html')

    def post(self):
        try:
            data = self.parser.parse_args()
            ppoNo = data['ppoNo']
            print("ppooooooooooooooooooooo",ppoNo)
            connection = mysql.connector.connect(**mysql_config)
            cursor = connection.cursor()
            IMAGE_FOLDER = r"C:\Users\gssaw\inten\finalnew\static\images"
            cursor.execute("""
                SELECT pensionId, ppoNo, firstName, lastName, ver_time, ver_img_path,ver_status
                FROM verification_result
                WHERE ppoNo = %s
                ORDER BY ver_time DESC
                LIMIT 5
            """, (ppoNo,))

            history = cursor.fetchall()
            print("histporyyyyyyyyyyyyy",history)
            if not history:
                return {'message': f'No data found for ppoNo {ppoNo}'}, 404

            # formatted_history = []
            # for record in history:
            formatted_records = []

            for record in history:
                formatted_record = {
                    'ppoNo': record[1],
                    'firstName': record[2],
                    'lastName': record[3],
                    'ver_time': record[4],
                    'ver_img_path': record[5],
                    "ver_status":record[-1]
                }
                formatted_records.append(formatted_record)

            # print(formatted_history)
            # return render_template('verification_history.html', history=formatted_history)
            return jsonify(formatted_records)

        except Exception as e:
            return {'error': str(e)}, 500






# Add resources to the API
# api.add_resource(StoreEmployeeResource, '/store_employee')
api.add_resource(VerificationHistoryResource, '/get_verification_history')
  
if __name__ == '__main__':
    app.run(debug=True, port=8000)
