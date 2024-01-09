from face_recognition.testing_engine import face_recognition_knn
import gun_detection.keras_frcnn as keras_frcnn
import cv2

detection_gun = True
if detection_gun:
    from gun_detection.test_frcnn_engine import tf_fit_img, class_to_color
    gun_color = (int(class_to_color['Gun'][0]), int(class_to_color['Gun'][1]), int(class_to_color['Gun'][2]))

video_capture = cv2.VideoCapture(0)
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        break

    if detection_gun:
        all_dets = tf_fit_img(frame)
    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    boxes, names = face_recognition_knn(small_frame, 0.4)

    Message = []
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        if name != 'Unknown':
            Message.append(name)

    if detection_gun:
        for (real_x1, real_y1, real_x2, real_y2) in all_dets:
            cv2.rectangle(frame, (real_x1, real_y1), (real_x2, real_y2), gun_color, 2)
            top = min(real_y1, real_y2)
            left = min(real_x1, real_x2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, 'Gun', (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, gun_color, 2)
        if len(all_dets) > 1:
            Message.append('Guns')
        elif len(all_dets) > 0:
            Message.append('Gun')

    if len(Message) > 0:
        Message = 'Detected ' + ', '.join(Message)
        print(Message)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
