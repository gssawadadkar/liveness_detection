
import cv2
import dlib
import os
from src.mysql_operations import fetch_employee_image, mysql_config, get_employee_name
import numpy as np
import face_recognition



EmpCode=input("Enter enployee code : ")
# Load face recognition model from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Function to calculate Euclidean distance between two embeddings
def calculate_distance(embedding1, embedding2):
    return sum((x - y) ** 2 for x, y in zip(embedding1, embedding2)) ** 0.5

# Function to recognize faces in a video
def recognize_faces():
    # Load known face embeddings
    known_embeddings = {}
    # for filename in os.listdir(known_folder):
    name = get_employee_name(EmpCode,mysql_config)
    known_video_path = fetch_employee_image(EmpCode,mysql_config)
    print("array of known image : ",known_video_path)
    # loaded_image = cv2.imread(known_video_path)
    print(f"Image dimensions: {known_video_path.shape}")
    # known_embedding = compute_embedding(known_video_path)   
    # known_embedding = compute_embedding(cv2.imread(known_video_path))  # Load image using cv2.imread
    # known_embeddings[name] = known_embedding

    # Open video capture
    cap = open_camera(video_path)

    while True:
        frame = cap.read()

        # Find faces in the frame
        faces = detect_faces(frame)

        # Recognize faces and display video file name
        for frame in faces:
            face_embedding = compute_embedding(frame)
            match_name = compare_embeddings(face_embedding, known_embeddings)
            print(f"Match: {match_name}")

        # Display the frame with rectangles around detected faces
        # display_frame(frame, faces)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Load the Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def open_camera(video_path):
    cap=cv2.VideoCapture(video_path)
    # cv2.imshow("Camera Feed", cap)
    return cap

# Function to detect faces in a frame using Haarcascade classifier
def detect_faces(frame):
    frame=open_camera(video_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # gray = 0.299 * frame[:, :, 2] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 0]
    gray = gray.astype(np.uint8)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces




# Function to compute face embedding
def compute_embedding(image, face=None):
    # If face is provided, use it; otherwise, use the entire image
    if face is not None:
        shape = predictor(image, face)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Assume the entire image is a face
        shape = predictor(image, dlib.rectangle(left=0, top=0, right=image.shape[1], bottom=image.shape[0]))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_embedding = recognizer.compute_face_descriptor(rgb_image, shape)
    return face_embedding


# Function to compare face embeddings with known embeddings
def compare_embeddings(face_embedding, known_embeddings, threshold=0.6):
    match_name = "Unknown"

    if face_embedding is not None:
        for name, known_embedding in known_embeddings.items():
            distance = calculate_distance(face_embedding, known_embedding)
            if distance < threshold:
                match_name = name
                break

    return match_name

# Function to display the frame with rectangles around detected faces
# def display_frame(frame, faces):
#     for face in faces:
#         (top, right, bottom, left) = (face.top(), face.right(), face.bottom(), face.left())
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         frame=cv2.imread(frame)
#         cv2.imshow('Face Recognition', frame)

# Main function
if __name__ == "__main__":
   
    video_path = 0  # Use 0 for live webcam stream, or provide the path to a video file
    open_camera(video_path)
    frame=open_camera(video_path)
    recognize_faces()
    detect_faces(frame)
