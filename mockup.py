import cv2
import mediapipe as mp
import numpy as np

#Program Variables
image_scale_test = 0.5
test_video = 'D:/Kauel/KMT/videos/probanding_cut.mp4'

#Initialazing Objects and Methods of OpenCV and MediaPipe
#Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def kscale_image(image, scale):

    new_width = image.shape[1]*scale
    new_height = image.shape[0]*scale

    new_width = int(new_width)
    new_height = int(new_height)

    dsize = (new_width, new_height)

    return cv2.resize(image, dsize)

def kextract_frames(video_file, save_folder):

    """
    :param video_file: Path to video
    :param image_scale: factor for resizing image. Default = 1
    """


    print('Starting to extract frames from ' + video_file)

    cap = cv2.VideoCapture(video_file)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_number = 0

    while frame_number < video_length and cap.isOpened():
        success, image = cap.read()
            
        if not success:
            break

        frame_name = save_folder +'/frame{:05d}.jpg'.format(frame_number)
        cv2.imwrite(frame_name, image)

        frame_number += 1
    
    cap.release()
    print('Freames extracted from ' + video_file)
    return True


def kbase_analysis(video_file: str, image_scale: float = 1):
    """
    :param video: Path to video
    :param image_scale: factor for resizing image. Default = 1
    """

    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            break
        
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = holistic.process(image)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        if image_scale <= 1:
            cv2.imshow('Body Tracking', kscale_image(image, image_scale))
        else:
            cv2.imshow('Body Tracking', image)
        
        if cv2.waitKey(int(5)) & 0xFF == ord('q'):
            cv2.destroyWindow('Body Tracking')
            break


    holistic.close()
    cap.release()


def kbase_frame_analysis(frames_folder: str, frame_initial: int, frame_final: int, image_scale: float = 1):
    """
    :param frames_folder: Path to frame folder
    :param frame_initial: First frame of secuence
    :param frame_final: Last frame of secuence
    :param image_scale: factor for resizing image. Default = 1
    """
    frames_id = [i for i in range(frame_initial, frame_final + 1)]

    for frame_id in frames_id:

        image = cv2.imread(frames_folder + '/frame{:05d}.jpg'.format(frame_id))

        image.flags.writeable = False
        results = holistic.process(image)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


        if image_scale <= 1:
            cv2.imshow('Body Tracking', kscale_image(image, image_scale))
        else:
            cv2.imshow('Body Tracking', image)
    
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyWindow('Body Tracking')
            break

