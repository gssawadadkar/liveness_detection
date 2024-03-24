# mysql_operations.py
import mysql.connector
import numpy as np
import cv2
import os
from logger_config import logger
# import imageio as iio
import imageio.v2

mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'jay@gajanan123',
    'database': 'liveness_detection'
}




import os
import imageio
import mysql.connector

def fetch_employee_image(ppoNo, mysql_config):
    try:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()

        query = "SELECT img_path FROM pensioners WHERE ppoNo = %s"
        cursor.execute(query, (ppoNo,))

        result = cursor.fetchone()
        print("Image path from database:", result)
        

        if result:
            # Extract image path
            image_path = result[0]

            # Ensure correct path format
            image_path = os.path.abspath(image_path.replace("\\", "/"))

            print("Absolute image path:", image_path)

            # Check if the file exists
            if os.path.exists(image_path):
                # Read the image using OpenCV
                image = cv2.imread(image_path)

                # Check if the image was successfully loaded
                if image is not None:
                    
                    logger.info("Image loaded successfully.")
                    return image
                else:
                    print(f"Error reading image from path: {image_path}")
                    return None
            else:
                print(f"File does not exist: {image_path}")
                return None
        else:
            print(f"No image path found for ppoNo: {ppoNo}")
            return None

    except Exception as e:
        print(f"Error fetching image: {str(e)}")
        return None
    finally:
        cursor.close()
        connection.close()




def get_employee_name(ppoNo, mysql_config):
    try:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()

        query = "SELECT pensionId, ppoNo, firstName, lastName, middlestName FROM pensioners WHERE ppoNo = %s"
        cursor.execute(query, (ppoNo,))
        result = cursor.fetchone()
        print(result)
        cursor.close()
        connection.close()

        # if result:
        #     # Check if result is not None before joining
        #     result_str = " ".join(result)
        return result

        return None
    except Exception as e:
        print("An error occurred:", e)
        return None


if __name__=="__main__":
    EmpCode="PN16"
    mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'jay@gajanan123',
    'database': 'liveness_detection'
} 
    get_employee_name(EmpCode, mysql_config)
  
    employee_image = fetch_employee_image(EmpCode, mysql_config)

    if employee_image is not None:
        # Now, 'employee_image' contains the image data as a NumPy array
        # Display the image using OpenCV
        cv2.imshow("Employee Image", employee_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image not fetched or an error occurred.")
