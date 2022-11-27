import mediapipe as mp
from util import *
import threading
import pyttsx3


face_mesh = mp.solutions.face_mesh
draw_utils = mp.solutions.drawing_utils
landmark_style = draw_utils.DrawingSpec((0, 255, 0), thickness=1, circle_radius=1)
connection_style = draw_utils.DrawingSpec((0, 0, 255), thickness=1, circle_radius=1)

STATIC_IMAGE = False
MAX_NO_FACES = 1
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)

LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]

RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]

FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                max_num_faces=MAX_NO_FACES,
                                min_detection_confidence=DETECTION_CONFIDENCE,
                                min_tracking_confidence=TRACKING_CONFIDENCE)

capture = cv.VideoCapture(0)

frame_count_sleep = 0
min_frame_sleep = 6
ratio_eyes_min = 5.0

frame_count_yawn = 0
min_frame_yawn = 24
ratio_lips_max = 1.8


speech = pyttsx3.init()

while True:
    result, image = capture.read()

    if result:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        outputs = face_model.process(image_rgb)

        if outputs.multi_face_landmarks:

            draw_landmarks(image, outputs, FACE, COLOR_GREEN)

            draw_landmarks(image, outputs, LEFT_EYE_TOP_BOTTOM, COLOR_RED)
            draw_landmarks(image, outputs, LEFT_EYE_LEFT_RIGHT, COLOR_RED)

            draw_landmarks(image, outputs, RIGHT_EYE_TOP_BOTTOM, COLOR_RED)
            draw_landmarks(image, outputs, RIGHT_EYE_LEFT_RIGHT, COLOR_RED)

            draw_landmarks(image, outputs, UPPER_LOWER_LIPS, COLOR_BLUE)
            draw_landmarks(image, outputs, LEFT_RIGHT_LIPS, COLOR_BLUE)

            ratio_left = get_aspect_ratio(image, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
            ratio_right = get_aspect_ratio(image, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
            ratio_eyes = (ratio_left + ratio_right) / 2.0
            ratio_lips = get_aspect_ratio(image, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)

            if ratio_eyes > ratio_eyes_min:
                frame_count_sleep += 1
            else:
                frame_count_sleep = 0

            if ratio_lips < ratio_lips_max:
                frame_count_yawn += 1
            else:
                frame_count_yawn = 0

            if frame_count_sleep > min_frame_sleep:
                # Closing the eyes
                message = 'Drowsy Alert: It appears you are sleeping. Kindly wake up'
                t = threading.Thread(target=run_speech, args=(speech, message))  # create new instance if thread is dead
                t.start()   
            elif frame_count_yawn > min_frame_yawn:
                # Open his mouth
                message = 'Drowsy Warning: You look tired. Kindly take a rest.'
                p = threading.Thread(target=run_speech, args=(speech, message))  # create new instance if thread is dead
                p.start()

        cv.imshow("FACE MESH", image)
        if cv.waitKey(1) & 255 == 27:
            break

capture.release()
cv.destroyAllWindows()