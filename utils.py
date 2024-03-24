import cv2
import numpy as np
# def create_white_frame(height, width):
#     white_frame = np.ones((height, width, 3), np.uint8) * 255  # Create a white frame
#     return white_frame

# def put_text_on_frame(frame, message):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.7
#     font_thickness = 2
#     text_size = cv2.getTextSize(message, font, font_scale, font_thickness)[0]
#     text_position = ((frame.shape[1] - text_size[0]) // 2, (frame.shape[0] + text_size[1]) // 2)

#     cv2.putText(frame, message, text_position, font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
#     return frame

# def overlay_frames(original_frame, overlay_frame):
#     return cv2.addWeighted(original_frame, 1, overlay_frame, 0.5, 0)


import numpy as np

def create_white_frame(height, width):
    return np.ones((height, width, 3), np.uint8) * 255

def put_text_on_frame(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = ((frame.shape[1] - text_size[0]) // 2, (frame.shape[0] + text_size[1]) // 2)
    cv2.putText(frame, text, text_position, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    return frame

def overlay_frames(original_frame, overlay_frame, position=(0, 0)):
    h, w, _ = original_frame.shape

    # Ensure that the overlay frame dimensions match the region to be replaced
    overlay_h, overlay_w, _ = overlay_frame.shape
    if position[0] + overlay_h > h or position[1] + overlay_w > w:
        raise ValueError("Overlay dimensions exceed original frame dimensions.")

    # Create a copy of the original frame to avoid modifying it directly
    result_frame = original_frame.copy()

    # Define the region in the original frame to be replaced with the overlay
    roi = result_frame[position[0]:position[0] + overlay_h, position[1]:position[1] + overlay_w]

    # Apply the overlay on the region
    result_frame[position[0]:position[0] + overlay_h, position[1]:position[1] + overlay_w] = cv2.addWeighted(roi, 1, overlay_frame, 0.5, 0)

    return result_frame

    
