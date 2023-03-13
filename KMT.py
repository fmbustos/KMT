import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3

#Program Variables
image_scale_test = 0.5
test_video = 'D:/Kauel/KMT/videos/probanding_cut.mp4'

#Initialazing Objects and Methods of MediaPipe
#Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

pose_landmark = {
    'NOSE' : 0,
    'LEFT_EYE_INNER' : 1,
    'LEFT_EYE' : 2,
    'LEFT_EYE_OUTER' : 3,
    'RIGHT_EYE_INNER' : 4,
    'RIGHT_EYE' : 5,
    'RIGHT_EYE_OUTER' : 6,
    'LEFT_EAR' : 7,
    'RIGHT_EAR' : 8,
    'MOUTH_LEFT' : 9,
    'MOUTH_RIGHT' : 10,
    'LEFT_SHOULDER' : 11,
    'RIGHT_SHOULDER' : 12,
    'LEFT_ELBOW' : 13,
    'RIGHT_ELBOW' : 14,
    'LEFT_WRIST' : 15,
    'RIGHT_WRIST' : 16,
    'LEFT_PINKY' : 17,
    'RIGHT_PINKY' : 18,
    'LEFT_INDEX' : 19,
    'RIGHT_INDEX' : 20,
    'LEFT_THUMB' : 21,
    'RIGHT_THUMB' : 22,
    'LEFT_HIP' : 23,
    'RIGHT_HIP' : 24,
    'LEFT_KNEE' : 25,
    'RIGHT_KNEE' : 26,
    'LEFT_ANKLE' : 27,
    'RIGHT_ANKLE' : 28,
    'LEFT_HEEL' : 29,
    'RIGHT_HEEL' : 30,
    'LEFT_FOOT_INDEX' : 31,
    'RIGHT_FOOT_INDEX' : 32
}

pose_connections = mp_holistic.POSE_CONNECTIONS

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


def kbase_analysis(video_file: str, image_scale: float = 1, show: bool = True  , save: bool = False):
    """
    :param video: Path to video
    :param image_scale: Factor for resizing image. Default = 1
    :param save: True or False, if set to True the video will be showed in a window while analyzing.
    :param save: False or True, if set to True the video will be saved in the same folder of the original video.
    """

    cap = cv2.VideoCapture(video_file)

    if save is True:
        framesize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap_writer = cv2.VideoWriter('analysis.avi', 
                                     cv2.VideoWriter_fourcc(*'MJPG'), 
                                     30, 
                                     framesize)

    video_landmarks_word = []
    video_landmarks_image = []

    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            break
        
        image.flags.writeable = False
        results = holistic.process(image)

        try:
            frame_landmarks_world = results.pose_world_landmarks.landmark
            frame_landmarks_image = results.pose_landmarks.landmark
        except AttributeError:
            print('AttributeError')
            continue

        x_world = []
        y_world = []
        z_world = []
        x_image = []
        y_image = []
        z_image = []

        for landmark_world, landmark_image in zip(frame_landmarks_world,frame_landmarks_image):
            x_world.append(landmark_world.x)
            y_world.append(-landmark_world.y)
            z_world.append(landmark_world.z)
            x_image.append(landmark_image.x)
            y_image.append(landmark_image.y)
            z_image.append(landmark_image.z)

        video_landmarks_word.append([x_world,y_world,z_world])
        video_landmarks_image.append([x_image, y_image, z_image])

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        if save is True:
            cap_writer.write(image)
        
        if show is not True:
            continue

        if image_scale <= 1:
            cv2.imshow('Body Tracking', kscale_image(image, image_scale))
        else:
            cv2.imshow('Body Tracking', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow('Body Tracking')
            show = False
                

    holistic.close()
    cap.release()
    if save is True:
        cap_writer.release()

    return video_landmarks_word, video_landmarks_image


def kbase_frame_analysis(frames_folder: str, frame_initial: int, frame_final: int, image_scale: float = 1):
    """
    :param frames_folder: Path to frame folder
    :param frame_initial: First frame of secuence
    :param frame_final: Last frame of secuence
    :param image_scale: factor for resizing image. Default = 1
    """
    frames_id = [i for i in range(frame_initial, frame_final + 1)]

    video_landmarks_word = []
    video_landmarks_image = []

    for frame_id in frames_id:

        image = cv2.imread(frames_folder + '/frame{:05d}.jpg'.format(frame_id))

        image.flags.writeable = False
        results = holistic.process(image)
        frame_landmarks_world = results.pose_world_landmarks.landmark
        frame_landmarks_image = results.pose_landmarks.landmark

        x_world = []
        y_world = []
        z_world = []
        x_image = []
        y_image = []
        z_image = []

        for landmark_world, landmark_image in zip(frame_landmarks_world,frame_landmarks_image):
            x_world.append(landmark_world.x)
            y_world.append(-landmark_world.y)
            z_world.append(landmark_world.z)
            x_image.append(landmark_image.x)
            y_image.append(landmark_image.y)
            z_image.append(landmark_image.z)

        video_landmarks_word.append([x_world,y_world,z_world])
        video_landmarks_image.append([x_image, y_image, z_image])

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
            
    return video_landmarks_word, video_landmarks_image

def kshow_landmarks(landmarks):
        
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1,projection="3d")
    scatter_plot = ax.scatter3D([],[],[], color='m')
    ax.set_xticks(np.arange(-1, 1, 0.25))
    ax.set_yticks(np.arange(-1, 1, 0.25))
    ax.set_zticks(np.arange(-1, 1, 0.25))

    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1, top=1) 

    ax.set_box_aspect(aspect = (1,1,1))

    ax.set(xlabel='X')
    ax.set(ylabel='Z, profundidad')
    ax.set(zlabel='Y, altura')

    lines = [ax.plot([], [], [])[0] for _ in pose_connections]

    def update_plot(num, lines):

        x = landmarks[num][0]
        y = landmarks[num][1]
        z = landmarks[num][2]

        scatter_plot._offsets3d = (x, z, y)

        for line, connection in zip(lines, pose_connections):
            line.set_data([x[connection[0]],x[connection[1]]],[z[connection[0]],z[connection[1]]])
            line.set_3d_properties([y[connection[0]],y[connection[1]]])

        return lines



    ani = animation.FuncAnimation(
        fig, update_plot, len(landmarks) ,fargs=([lines]), interval=10)

    plt.show()

wl, il = kbase_analysis('D:/Kauel/KMT/videos/probanding_cut.mp4',1)

kshow_landmarks(wl)