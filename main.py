import cv2
from face_recognition.testing_engine import face_recognition_knn
from gun_detection.keras_frcnn import tf_fit_img, class_to_color

# Flag to enable gun detection
detection_gun = True

# Initialize gun color if gun detection is enabled
if detection_gun:
    gun_color = (
        int(class_to_color['Gun'][0]),
        int(class_to_color['Gun'][1]),
        int(class_to_color['Gun'][2])
    )

# Open the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame of video
    ret, frame = video_capture.read()

    # Break the loop if there's no frame captured
    if not ret:
        break

    # Perform gun detection if enabled
    if detection_gun:
        all_dets = tf_fit_img(frame)

    # Resize the frame for face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    boxes, names = face_recognition_knn(small_frame, 0.4)

    # Initialize a message list to store detections
    Message = []

    # Loop through the detected faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Draw rectangles around the detected faces
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Display the name above the rectangles
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        # Add recognized names to the Message list
        if name != 'Unknown':
            Message.append(name)

    # Perform gun detection if enabled
    if detection_gun:
        for (real_x1, real_y1, real_x2, real_y2) in all_dets:
            # Draw rectangles around detected guns
            cv2.rectangle(frame, (real_x1, real_y1), (real_x2, real_y2), gun_color, 2)
            
            # Display 'Gun' label above the rectangles
            top = min(real_y1, real_y2)
            left = min(real_x1, real_x2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, 'Gun', (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, gun_color, 2)
        
        # Add 'Gun' or 'Guns' to the Message list based on detections
        if len(all_dets) > 1:
            Message.append('Guns')
        elif len(all_dets) > 0:
            Message.append('Gun')

    # Display the detection message
    if len(Message) > 0:
        Message = 'Detected ' + ', '.join(Message)
        print(Message)

    # Show the video frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
