import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import json
import inspect
import re
import time
import os
from copy import deepcopy
from copy import copy

try:
    from alive_progress import alive_bar
    show_alive_bar = True
except:
    show_alive_bar = False
try:
    from alive_progress import alive_it
    show_alive_it = True
except:
    show_alive_it = False

#Testing Variables
image_scale_test = 0.5
test_video = 'D:/Kauel/KMT/videos/probanding_cut.mp4'
test_frame_folder = "D:/Kauel/KMT/videos/probanding_cut_frames"
test_video_folder = "D:/Kauel/KMT/videos"
test_data_folder = "D:/Kauel/KMT/data"
test_frame = test_frame_folder + '/frame01558.jpg'

#Initialazing Objects and Methods of MediaPipe
#Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
static_image_mode=False
model_complexity=2
smooth_landmarks=True
enable_segmentation=False
smooth_segmentation=True
refine_face_landmarks=False
min_detection_confidence=0.5
min_tracking_confidence=0.5
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

holistic = mp_holistic.Holistic(
static_image_mode = static_image_mode,
model_complexity = model_complexity,
smooth_landmarks = smooth_landmarks,
enable_segmentation = enable_segmentation,
smooth_segmentation = smooth_segmentation,
refine_face_landmarks = refine_face_landmarks,
min_detection_confidence = min_detection_confidence,
min_tracking_confidence = min_tracking_confidence)



#Global Variables
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
    'RIGHT_FOOT_INDEX' : 32,
    'nose' : 0,
    'left_eye_inner' : 1,
    'left_eye' : 2,
    'left_eye_outer' : 3,
    'right_eye_inner' : 4,
    'right_eye' : 5,
    'right_eye_outer' : 6,
    'left_ear' : 7,
    'right_ear' : 8,
    'mouth_left' : 9,
    'mouth_right' : 10,
    'left_shoulder' : 11,
    'right_shoulder' : 12,
    'left_elbow' : 13,
    'right_elbow' : 14,
    'left_wrist' : 15,
    'right_wrist' : 16,
    'left_pinky' : 17,
    'right_pinky' : 18,
    'left_index' : 19,
    'right_index' : 20,
    'left_thumb' : 21,
    'right_thumn' : 22,
    'left_hip' : 23,
    'right_hip' : 24,
    'left_knee' : 25,
    'right_knee' : 26,
    'left_ankle' : 27,
    'right_ankle' : 28,
    'left_heel' : 29,
    'right_heel' : 30,
    'left_foot_index' : 31,
    'right_foot_index' : 32,
    'nose' : 0,
    'left eye inner' : 1,
    'left eye' : 2,
    'left eye outer' : 3,
    'right eye inner' : 4,
    'right eye' : 5,
    'right eye outer' : 6,
    'left ear' : 7,
    'right ear' : 8,
    'mouth left' : 9,
    'mouth right' : 10,
    'left shoulder' : 11,
    'right shoulder' : 12,
    'left elbow' : 13,
    'right elbow' : 14,
    'left wrist' : 15,
    'right wrist' : 16,
    'left pinky' : 17,
    'right pinky' : 18,
    'left index' : 19,
    'right ndex' : 20,
    'left thumb' : 21,
    'right thumn' : 22,
    'left hip' : 23,
    'right hip' : 24,
    'left knee' : 25,
    'right knee' : 26,
    'left ankle' : 27,
    'right ankle' : 28,
    'left heel' : 29,
    'right heel' : 30,
    'left foot index' : 31,
    'right foot index' : 32
}
axis_landmarks = ['left shoulder', 'right shoulder', 'left hip', 'right hip']

important_pose_landmarks = {
    'NOSE' : 0,
    'LEFT_SHOULDER' : 11,
    'RIGHT_SHOULDER' : 12,
    'LEFT_ELBOW' : 13,
    'RIGHT_ELBOW' : 14,
    'LEFT_WRIST' : 15,
    'RIGHT_WRIST' : 16,
    'LEFT_HIP' : 23,
    'RIGHT_HIP' : 24,
    'LEFT_KNEE' : 25,
    'RIGHT_KNEE' : 26,
    'LEFT_ANKLE' : 27,
    'RIGHT_ANKLE' : 28
}

pose_connections = mp_holistic.POSE_CONNECTIONS

k_pose_connection = {
    "HEAD": (0,11,12),
    "RIGHT_ARM": (12,14),
    "RIGHT_FOREARM": (14,16),
    "RIGHT_HAND": (16, 18, 20, 22),
    "LEFT_ARM": (11,13),
    "LEFT_FOREARM": (13,15),
    "LEFT_HAND": (15, 17, 19, 21),
    "BODY": (11, 12, 24, 23),
    "RIGHT_TIGHT": (24, 26),
    "RIGHT_SHIN": (26, 28),
    "RIGHT_FOOT": (28, 30, 32),
    "LEFT_TIGHT": (23, 25),
    "LEFT_SHIN": (25, 27),
    "LEFT_FOOT": (27, 29, 31)
}

plt_colors = {'b' : 'blue', 'g':'green', 'r':'red', 'c':'cyan', 'm':'magenta', 'y':'yellos', 'k':'black', 'w':'white'}
rot_mat = np.array([[1,0,0],
                   [0,0,1],
                   [0,-1,0]])

arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)

body_parts_1_angle = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']
body_parts_2_angles = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
body_axis_places = ['center', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
angles_body_parts = {
    'left_elbow':'left_elbow', 'right_elbow' : 'right_elbow', 'left_knee' : 'left_knee', 'right_knee' : 'right_knee',
    'left_shoulder': ('left_shoulder_theta','left_shoulder_phi'), 
    'right_shoulder': ('right_shoulder_theta', 'right_shoulder_phi'), 
    'left_hip': ('right_shoulder_theta','right_shoulder_phi'), 
    'right_hip': ('right_hip_theta', 'right_hip_phi'), 
    'left_shoulder': ('left_shoulder_theta','left_shoulder_phi'), 
    'right_shoulder' : ('right_shoulder_theta','right_shoulder_phi'), 
    'left_hip' : ('left_hip_theta','left_hip_phi'), 
    'right_hip': ('right_hip_theta','right_hip_phi')
    }

paused = False
theta = 'Θ'
omega = 'ω'


#Debugging functions and classes
def debug(*args):

    message = 'Debug: '

    frame = inspect.currentframe().f_back
    frame_info = inspect.getframeinfo(frame).code_context[0]
    variables_names = re.search(r"\((.*)\)", frame_info).group(1)
    variables_names = variables_names.split(',')

    i = 0

    if (variables_names[0][0] == '\'' and variables_names[0][-1] == '\'') or (variables_names[0][0] == '\"' and variables_names[0][-1] == '\"'):
        message += '\"' + args[0] + '\"'
        i+= 1

    if len(variables_names) > i:
        title = '\n' + text_bold_underline('.{:^20s}   {:^60s}.'.format('','') + '\n|{:^20s} | {:^60s}|'.format('Variable Name','Variable Value'))
        message += title
    while i < len(variables_names):
        message += '\n' + text_underline("|{:^20s} | {:^60s}|".format(variables_names[i],str(args[i])))
        i+=1
    
    print(message)

def text_bold(text):
    return '\033[1m' + text + '\033[0m'

def text_underline(text):
    return '\033[4m' + text + '\033[0m'

def text_bold_underline(text):
    return text_underline(text_bold(text))

#Process Bar functions and classes
def progress_bar(loop_function, *total):
    if len(total) == 1:
        total = int(total[0])
    else:
        total = False

    if not show_alive_bar:
        result = loop_function()
        return result
    
    with alive_bar(total) as bar:
        result = loop_function(bar)
    
    return result

#Utility functions and classes
def is_landmark(landmark):
    if type(landmark) == type(None):
        return False
    elif type(landmark) == type([]):
        return True
    elif type(landmark) == type(np.array([])):
        return True
    else:
        return False

#Maths and Matrix functions and classes
def landmarks_average(*video_landmarks):

    average_video_landmarks = []

    for landmarks_in_frames in zip(*video_landmarks):
        average_landmarks_in_frame = []
        for landmarks in zip(*landmarks_in_frames):
            average_landmarks = np.nanmean(landmarks, axis=0)
            average_landmarks_in_frame.append(average_landmarks)
        average_video_landmarks.append(average_landmarks_in_frame)

    return average_video_landmarks

def rotate_landmarks(video_landmarks):
    rot_mat = np.array([[1,0,0],
                   [0,0,1],
                   [0,-1,0]])

    rotated_video_landmarks = []
    for landmarks_in_frame in video_landmarks:
        if type(landmarks_in_frame) != type(None):
            rotated_frame_landmarks = []
            for landmark in landmarks_in_frame:
                rotated_landmark = rot_mat.dot(landmark)
                rotated_frame_landmarks.append(rotated_landmark)
                # print(landmark, rotated_landmark)
        else:
            rotated_frame_landmarks = None
        rotated_video_landmarks.append(rotated_frame_landmarks)
    return rotated_video_landmarks

def landmarks_error(landmarks1, landmarks2):

    delta_landmarks = []

    for landmark1, landmark2 in zip(landmarks1, landmarks2):
        if is_landmark(landmark1) and is_landmark(landmark2):
            landmark1 = np.array(landmark1)
            landmark2 = np.array(landmark2)
            delta_landmark = (landmark1 - landmark2)
            
            landmark_error = []
            for dx,dy,dz, x, y, z in zip(*delta_landmark, *landmark1):
                error = np.sqrt(dx**2 + dy**2 + dz**2)/np.sqrt(x**2 + y**2 + z**2)
                landmark_error.append(error)
            
            delta_landmarks.append(landmark_error)

        else:

            delta_landmark = None
    
    return delta_landmarks

def landmarks_errors(*landmarks_set):
    
    errors = np.array(landmarks_error(landmarks_set[0], landmarks_set[0]))

    for landmark in landmarks_set[1:]:

        results = np.array(landmarks_error(landmarks_set[0], landmark))

        errors = errors + results

    errors = list(errors)

    return errors

def principal_lines(landmarks):
    
    points = {}

    for axis_landmark in axis_landmarks:
        x, y, z = landmarks[pose_landmark[axis_landmark]][0], landmarks[pose_landmark[axis_landmark]][1], landmarks[pose_landmark[axis_landmark]][2]
        point = np.array([x,y,z])
        points.update({axis_landmark : point})
    
    upper_point = points['left shoulder']/2 + points['right shoulder']/2
    # lower_point = points['left hip']/2 + points['right hip']/2
    lower_point = np.array([0,0,0])
    central_line = np.array([lower_point, upper_point])
    upper_line = np.array([points['left shoulder'], points['right shoulder']])
    lower_line = np.array([points['left hip'], points['right hip']])

    return lower_line, central_line, upper_line

def body_axis(landmark, body_part = 'center'):

    lower_line, central_line, upper_line = principal_lines(landmark)

    if body_part == 'left_shoulder':
        origin = upper_line[0]
        b2 = upper_line[0] - central_line[1]
    elif body_part == 'right_shoulder':
        origin = upper_line[1]
        b2 = central_line[1] - upper_line[1]
    elif body_part == 'left_hip':
        origin = lower_line[0]
        b2 = lower_line[0] - central_line[0]
    elif body_part == 'right_hip':
        origin = lower_line[1]
        b2 = central_line[0] - lower_line[1]
    elif body_part == 'left_hip':
        origin = central_line[1]
        b2 = upper_line[0] - central_line[1]
    else:
        origin = [0,0,0]
        b2 = lower_line[0]

    b2 = b2/np.linalg.norm(b2)
    
    spine = central_line[1]

    b1 = np.cross(b2, spine)

    b1 = b1/np.linalg.norm(b1)

    b3 = np.cross(b1,b2)

    return np.array([origin, b1]), np.array([origin, b2]), np.array([origin, b3])

def rotation_matrix(axis, angle):
    if axis == 0:
        return np.array([[1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]])
    if axis == 1:
        return np.array([[np.cos(angle), 0,  -np.sin(angle)],
                        [0, 1,0],
                        [np.sin(angle), 0, np.cos(angle)]])
    if axis == 2:
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]])

def vector_angle(v1,v2):

    if len(v1) == 3 and len(v2) == 3:
        v1_magnitude = np.linalg.norm(v1)
        v2_magnitude = np.linalg.norm(v2)
    if len(v1) == 2 and len(v2) == 2:
        v1 = v1[1]
        v2 = v2[1]
        v1_magnitude = np.linalg.norm(v1)
        v2_magnitude = np.linalg.norm(v2)

    return np.arccos(np.dot(v1,v2)/v1_magnitude/v2_magnitude)

def arc_points(v1,v2, radious = 0.1):

    v1n = v1[1]/np.linalg.norm(v1[1])*radious
    v2n = v2[1]/np.linalg.norm(v2[1])*radious
    parameter = np.linspace(0, 1, 15)
    angle = vector_angle(v1n,v2n)
    points = []
    for t in parameter:
        #https://en.wikipedia.org/wiki/Slerp
        v_arc = np.sin((1-t) * angle)/np.sin(angle) * v1n + np.sin(t*angle)/np.sin(angle)*v2n
        v_arc = v_arc + v1[0]
        points.append(v_arc)
    
    return [point[0] for point in points], [point[1] for point in points], [point[2] for point in points]

def line_points(v1, v2):
    v1n = v1[1]
    v2n = v2[1]
    lineal_combinations = np.linspace(0, 1, 20)
    points = []
    for a in lineal_combinations:
        v0 = v1n * a + v2n * (1-a)
        v0 = v0 + v1[0]
        points.append(v0)
    
    return [point[0] for point in points], [point[1] for point in points], [point[2] for point in points]

def scale_vector(vector, factor):
    vector = np.array(vector)
    if len(vector) == 2:
        vector[1] = vector[1]*factor
        return vector
    elif len(vector) == 3:
        return vector*factor
    else:
        debug('In scale_vector: input is not a vector like, it must be a list with shape (1,3) or (2,3)', vector, factor)

def points_to_vector(initial_point, end_point):
    origin = np.array(initial_point)
    vector = np.array(end_point) - np.array(initial_point)
    return np.array([origin, vector])

def vector_to_points(vector):
    initial_point = vector[0]
    end_point = vector[1] + vector[0]
    return np.array([initial_point, end_point])

def vector_xyz(vector):

    x = [vector[0][0], vector[0][0] + vector[1][0]]
    y = [vector[0][1], vector[0][1] + vector[1][1]]
    z = [vector[0][2], vector[0][2] + vector[1][2]]

    return x, y, z

def points_xyz(point_list):

    x = [v[0] for v in point_list]
    y = [v[1] for v in point_list]
    z = [v[2] for v in point_list]

    return x, y, z

def calculate_angles(landmarks):

    angles = {}

    for body_part in body_parts_1_angle:
        index = pose_landmark[body_part]
        origin = landmarks[index]
        v1 = landmarks[index + 2] - origin
        v2 = landmarks[index - 2] - origin
        angle = vector_angle(v1,v2)
        angles.update({body_part:angle})

    for body_part in body_parts_2_angles:
        b1, b2, b3 = body_axis(landmarks, body_part)
        index = pose_landmark[body_part]
        origin = landmarks[index]
        v = landmarks[index + 2] - origin
        v = change_basis(v, [1,0,0],[0,1,0],[0,0,1],b1[1],b2[1],b3[1])
        angle1 = np.arctan2(v[1],v[0])
        angle2 = np.arctan2(np.sqrt(v[0]**2 + v[1]**2),v[2])
        # if angle1 < 0:
        #     angle1 += 2*np.pi
        angles.update({body_part + '_theta':angle1})
        angles.update({body_part + '_phi':angle2})

    return angles

def calculate_video_angles(video_landmarks):
    video_angles = []
    for landmarks in video_landmarks:
        video_angles.append(calculate_angles(landmarks))
    return video_angles

def angles_absolute_to_relative(angles):

    relative_angles = deepcopy(angles)

    relative_angles['left_shoulder_phi'] = np.pi - relative_angles['left_shoulder_phi']
    relative_angles['left_shoulder_theta'] = np.pi/2 - relative_angles['left_shoulder_theta']
    
    relative_angles['right_shoulder_phi'] = np.pi - relative_angles['right_shoulder_phi']
    relative_angles['right_shoulder_theta'] = relative_angles['right_shoulder_theta'] + np.pi/2 

    relative_angles['left_hip_phi'] = np.pi - relative_angles['left_hip_phi']
    # relative_angles['left_hip_theta'] = relative_angles['left_hip_theta']

    relative_angles['right_hip_phi'] = np.pi - relative_angles['right_hip_phi']
    relative_angles['right_hip_theta'] = - relative_angles['right_hip_theta']

    return relative_angles

def body_reference_lines(landmarks, body_part = 'left_shoulder'):

    e1,e2,e3 = body_axis(landmarks, body_part)


    if body_part != 'center':
        e3[1] = - e3[1]

    if body_part == 'left_shoulder':
        v_holder = deepcopy(e1)
        e1 = e2
        e2 = v_holder
    elif body_part == 'right_shoulder':
        v_holder = deepcopy(e1)
        e1[1] = - e2[-1]
        e2 = v_holder
    elif body_part == 'left_hip':
        pass
    elif body_part == 'right_hip':
        e2[1] = -e2[1]

    return e1, e2, e3

def change_basis(v, e1, e2, e3, b1, b2, b3):
    
    transformation_matrix = np.linalg.solve([e1,e2,e3],[b1,b2,b3])

    return transformation_matrix.dot(v)

def calculate_angular_velocities(video_landmarks, fps = 30):
    
    video_velocities = []

    for landmarks0, landmarks1 in zip(video_landmarks[:-1], video_landmarks[1:]):
        
        velocities = {}

        angles0 = calculate_angles(landmarks0)
        angles1 = calculate_angles(landmarks1)

        for angle_key in angles0.keys():
            angle0 = angles0[angle_key]
            angle1 = angles1[angle_key]
            
            velocity = (angle1 - angle0)/fps

            velocities.update({angle_key:velocity})

        video_velocities.append(velocities)
    
    video_velocities.append(video_velocities[-1])

    return video_velocities

#Core functions and classes
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

def kanalyze_image(image):
    """
    Analyzes an image with mediapipe and returns the landmarks.
    :param image: a cv2 image to be analyzed
    :return: list of world landmarks, list of image landmarks, image with the draw of the solution
    """

    image.flags.writeable = False
    results = holistic.process(image)

    try:
        frame_landmarks_world = results.pose_world_landmarks.landmark
        frame_landmarks_image = results.pose_landmarks.landmark
    except AttributeError:
        debug('No solutions found in a frame')
        landmarks_world_vectors = np.array([[np.nan,np.nan,np.nan]]*33)
        landmarks_image_vectors = np.array([[np.nan,np.nan,np.nan]]*33)
        return landmarks_world_vectors, landmarks_image_vectors, image

    landmarks_world_vectors = []
    landmarks_image_vectors = []


    for landmark_world, landmark_image in zip(frame_landmarks_world,frame_landmarks_image):
        landmark_world_vector = np.array([landmark_world.x,landmark_world.y,landmark_world.z])
        landmark_image_vector = np.array([landmark_image.x,landmark_image.y,landmark_image.z])
        
        rotated_landmark_world_vector = rot_mat.dot(landmark_world_vector)

        landmarks_world_vectors.append(rotated_landmark_world_vector)
        landmarks_image_vectors.append(landmark_image_vector)


    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    return landmarks_world_vectors, landmarks_image_vectors, image

def kanalyze_image_xyz(image):
    """
    Analyzes an image with mediapipe and returns the landmarks.
    :param image: a cv2 image to be analyzed
    :return: list of world landmarks, list of image landmarks, image with the draw of the solution. 
    \nThe lists have the shape: [list of x values, list of y values, list of z values], wich each list having 33 values corresponding with the landmark.
    """


    image.flags.writeable = False
    results = holistic.process(image)

    try:
        frame_landmarks_world = results.pose_world_landmarks.landmark
        frame_landmarks_image = results.pose_landmarks.landmark
    except AttributeError:
        debug('No solutions found in a frame')
        return None, None, image

    x_world = []
    y_world = []
    z_world = []
    x_image = []
    y_image = []
    z_image = []
    

    for landmark_world, landmark_image in zip(frame_landmarks_world,frame_landmarks_image):
        x_world.append(landmark_world.x)
        y_world.append(landmark_world.y)
        z_world.append(landmark_world.z)
        x_image.append(landmark_image.x)
        y_image.append(landmark_image.y)
        z_image.append(landmark_image.z)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    return [x_world,y_world,z_world], [x_image, y_image, z_image], image

def kvideo_analysis(video_file: str):
    """
    Makes an motion tracking analysis of a given video file and returns the data.
    :param video_file: Path to video
    :return: video_landmarks_world, video_landmarks_image, image_list
    """

    if not os.path.isfile(video_file):
        debug('kvideo_analysis: could not find video file in ' + video_file + '\'')
        return None, None, None

    holistic = mp_holistic.Holistic(
    static_image_mode = static_image_mode,
    model_complexity = model_complexity,
    smooth_landmarks = smooth_landmarks,
    enable_segmentation = enable_segmentation,
    smooth_segmentation = smooth_segmentation,
    refine_face_landmarks = refine_face_landmarks,
    min_detection_confidence = min_detection_confidence,
    min_tracking_confidence = min_tracking_confidence)

    cap = cv2.VideoCapture(video_file)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    video_landmarks_world = []
    video_landmarks_image = []
    image_list = []

    def loop(*additional_functions):

        while cap.isOpened():

            success, image = cap.read()
            

            if not success:
                break
            

            for func in additional_functions:
                func()
            
            frame_landmarks_world, frame_landmarks_image, image = kanalyze_image(image)
            video_landmarks_world.append(frame_landmarks_world)
            video_landmarks_image.append(frame_landmarks_image)
            image_list.append(image)

    try:
        progress_bar(loop, total_frames)
    except:
        debug('Error with progress_bar in kvideo_analysis')
        video_landmarks_world = []
        video_landmarks_image = []
        image_list = []
        loop()

    holistic.close()
    cap.release()

    debug("kvideo_analysis finished")
    return video_landmarks_world, video_landmarks_image, image_list

def kframe_analysis(frames_folder: str, frame_initial: int, frame_final: int):
    """
     Makes an motion tracking analysis of a series of images and returns the data.
    :param frames_folder: Path to frame folder
    :param frame_initial: First frame of secuence
    :param frame_final: Last frame of secuence
    :return: video_landmarks_world, video_landmarks_image, image_list
    """
    holistic = mp_holistic.Holistic(
    static_image_mode = static_image_mode,
    model_complexity = model_complexity,
    smooth_landmarks = smooth_landmarks,
    enable_segmentation = enable_segmentation,
    smooth_segmentation = smooth_segmentation,
    refine_face_landmarks = refine_face_landmarks,
    min_detection_confidence = min_detection_confidence,
    min_tracking_confidence = min_tracking_confidence)

    frames_id = [i for i in range(frame_initial, frame_final + 1)]
    total_frames = frame_final - frame_initial + 1
    video_landmarks_world = []
    video_landmarks_image = []
    image_list = []

    def loop(*additional_functions):
        for frame_id in frames_id:

            image = cv2.imread(frames_folder + '/frame{:05d}.jpg'.format(frame_id))

            if type(image) == type(None):
                continue

            frame_landmarks_world, frame_landmarks_image, image = kanalyze_image(image)
            video_landmarks_world.append(frame_landmarks_world)
            video_landmarks_image.append(frame_landmarks_image)
            image_list.append(image)

            for func in additional_functions:
                func()

    try:
        progress_bar(loop, total_frames)
    except:
        debug('Error with progress_bar in kframe_anaysis')
        loop()
    
    holistic.close()
    
    debug("kframe_analysis finished")
    return video_landmarks_world, video_landmarks_image, image_list

#Plot functions and classes
def kplot_landmarks_world(video_landmarks_world, video_images = [], plot_body_axis = ['central'], plot_angles = [], save = False, filename = 'animation.gif', fps = 30):
    """
    Plots the landmarks in a 3d grid, aditionally plots the image used to make the data beside for comparation.
    :param video_landmarks_world: List of landmarks to animate.
    :param image_list: list of cv2 images from witch the landmarks where calculated to put beside the animation.
    :save: True or False. If true the video will be recorded as a .gif
    :param filename: If save is True, the video will be saved with this name.
    :param fps: If save is True, this will be the framerate of the saved video
    :return: True
    """
    frames_images = len(video_images)
    total_frames = len(video_landmarks_world)

    #Make the matplotlib figure to create graphs and plots
    fig = plt.figure()

    #Makes the axes for ploting the landmarks and image if required.
    if frames_images == total_frames:
        plot_image = True
        ax = fig.add_subplot(1,2,1,projection="3d")
        ax2 = fig.add_subplot(1,2,2)

    elif frames_images != 0:
        plot_image = False
        print('Size Error: The list of landmarks doesn\'t coincide with the list of images.')
        ax = fig.add_subplot(projection="3d")
    else:
        plot_image = False
        ax = fig.add_subplot(projection="3d")
    
    #Settings of the plot
    set_axes_for_landmarks_world(ax)

    handlers = [Landmark_Plotter_Handler(video_landmarks_world, ax, plot_body_axis=plot_body_axis, plot_angles = plot_angles)]

    if plot_image:
        handlers.append(Image_Plotter_Handler(video_images, ax2))

    return plot_animated_figure(fig, total_frames, handlers, save, filename, fps)

def kplot_landmarks(*landmarks_set, plot_body_axis = ['central'], plot_angles = [], save = False, filename = 'compare_plot.gif', fps = 30):
    """
    Plots landmarks.
    :param *landmarks_set: Any amount of landmarks to animate.
    :param plot_body_axis: List of body parts. The function will plot axis x,y,z that follows that part.
    :plot_angles: List of body parts. The function will plot the important angles made by that body part with another, or with certain axis.
    :save: True or False. If true the video will be recorded as a .gif
    :param filename: If save is True, the video will be saved with this name.
    :param fps: If save is True, this will be the framerate of the saved video
    :return: True
    """
    
    for landmarks, index in zip(landmarks_set, range(len(landmarks_set))):
        if not is_landmark(landmarks):
            debug('error in kplot_landmarks: list ' + str(index) + ' is empty.')
            return False

    ammount_of_plots = len(landmarks_set)
    total_frames = len(max(landmarks_set, key=len))
    rows, columns = subplot_grid(ammount_of_plots)

    fig = plt.figure()

    handlers = []

    index = 1
    for video_landmarks in landmarks_set:
        
        ax = fig.add_subplot(rows, columns, index, projection="3d")

        landmark_handler = Landmark_Plotter_Handler(
            video_landmarks, 
            ax, 
            plot_body_axis = plot_body_axis, 
            plot_angles = plot_angles,
            name = 'Handler of video landmarks {}'.format(index))
        
        handlers.append(landmark_handler)
        
        index += 1


    return plot_animated_figure(fig, total_frames, handlers, save, filename, fps)

def kplot_angles(video_landmarks_world: list, 
                video_images: list, 
                angles = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee','right_knee'],
                plot_body_axis = [], 
                plot_angles = [],
                save = False, 
                filename = 'angle_plot.gif', 
                fps = 10
                ):
    """
    Plots the angles of the landmarks. Also plots the landmarks in a 3d grid and the image used to make the data for comparation.
    :param video_landmarks_world: List of landmarks to animate.
    :param video_images: list of cv2 images from witch the landmarks where calculated to put beside the animation.
    :param angles: List of body angles to plot. The posible options are 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee' and'right_knee'
    :param save: True or False. If true the video will be recorded as a .gif
    :param filename: If save is True, the video will be saved with this name.
    :param fps: If save is True, this will be the framerate of the saved video
    :return: True
    """
    fig = plt.figure(figsize=(16, 9), dpi=1920/16)
    total_frames = len(video_landmarks_world)

    axs = []
    handlers = []
    
    columns = 3

    axs_ammount = 0
    axs_left = 0
    axs_right = 0
    for angle in angles:
        axs_ammount += 1
        if angle[0] == 'l':
            axs_left += 1
        elif angle[0] == 'r':
            axs_right += 1   

    rows = max(axs_left, axs_right)

    video_angles = calculate_video_angles(video_landmarks_world)

    row_left = 0
    row_right = 0
    
    for angle in angles:

        if angle in body_parts_1_angle:
            if angle[0] == 'l':
                index = row_left*3 + 1
                row_left += 1
            else:
                index = row_right*3 + 3
                row_right += 1
            ax = fig.add_subplot(rows, columns, index)
            handler = Angle_Plotter_Handler(video_angles, ax, angle_name = angle)
            axs.append(ax)
            handlers.append(handler)

        elif angle in body_parts_2_angles:
            if angle[0] == 'l':
                index = row_left*6 + 1
                row_left += 1
            else:
                index = row_right*6 + 5
                row_right += 1
            ax1 = fig.add_subplot(rows, columns+3, index)
            ax2 = fig.add_subplot(rows, columns+3, index+1)
            handler = Angle_Plotter_Handler(video_angles, [ax1,ax2], angle_name = angle)
            axs.append(ax1)
            axs.append(ax2)
            handlers.append(handler)


    axs.append(fig.add_subplot(rows,columns,2))
    handler = Image_Plotter_Handler(video_images, axs[-1])
    handlers.append(handler)

    landmarks_row = max(int(np.floor(rows/2)),2)
    axs.append(fig.add_subplot(landmarks_row,1,2, projection = '3d'))
    handler = Landmark_Plotter_Handler(video_landmarks_world,axs[-1], plot_body_axis=plot_body_axis, plot_angles=plot_angles)
    handlers.append(handler)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.tight_layout()

    return plot_animated_figure(fig, total_frames, handlers, save, filename, fps)

def update_plot(frame, handlers, save):

        artists = []

        for handler in handlers:
            handler_artists = handler.update_plot(frame)
            artists = np.concatenate((artists, handler_artists))

        if show_alive_bar and save:
            try:
                bar()
            except Exception as e:
                debug('in update plot {}'.format(e))


        return artists

def plot_animated_figure(fig, total_frames, handlers, save, filename, fps):

    ani = animation.FuncAnimation(fig, update_plot, total_frames, fargs= (handlers,save), interval=30, blit = True)

    if save:
        if show_alive_bar:
            with alive_bar(total_frames) as bar:
                ani.save(filename, writer='imagemagick', fps=fps)
        else:
            ani.save(filename, writer='imagemagick', fps=fps)
    
    def pause(*args, **kwargs):
        ani.pause()
    
    def unpause(*args, **kwargs):
        ani.resume()

    fig.canvas.mpl_connect('button_press_event', pause)
    fig.canvas.mpl_connect('button_release_event', unpause)

    plt.show()
    
    return True

def subplot_grid(num):
    row = int(round(np.sqrt(num)))
    col = int(np.ceil(np.sqrt(num)))
    return row, col

def set_axes_for_landmarks_world(ax, index = 1):
    ax.set_title('Landmarks Positions {}'.format(index))    
    ax.set_xticks(np.arange(-1, 1, 0.25))
    ax.set_yticks(np.arange(-1, 1, 0.25))
    ax.set_zticks(np.arange(-1, 1, 0.25))

    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1, top=1) 

    # ax.set_box_aspect(aspect = (1,1,1))

    ax.set(xlabel='X')
    ax.set(ylabel='Y')
    ax.set(zlabel='Z')

def set_axes_for_angles(ax):
    ax.set_xticks([])
    ax.set_yticks([])

    ax.axes.set_xlim(left=-2, right=2)
    ax.axes.set_ylim(bottom=-2, top=2) 
    ax.set_box_aspect(1)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)
    
    def set_data(self, xs, ys, zs):
        self._verts3d = xs, ys, zs

class Landmark_Plotter_Handler:
    def __init__(self, video_landmarks, ax, name = 'Video Landmark Plotter', plot_body_axis = ['central'], plot_angles = []) -> None:
        self.video_landmarks = video_landmarks
        self.ax = self.set_ax(ax)
        self.name = name

        self.angular_velcities = calculate_angular_velocities(video_landmarks)
        self.angles = calculate_video_angles(video_landmarks)

        self.lines = [ax.plot([], [], [], color = 'black', linewidth = 2)[0] for _ in pose_connections]
        self.scatter = ax.scatter3D([],[],[], color='m')
        self.plot_body_axis = self.set_plot_body_axis(plot_body_axis)
        self.plot_angles = self.set_plot_angles(plot_angles)
        self.arrows = self.set_arrows()
        self.arcs = self.set_arcs()
        self.video_landmarks_xyz, self.video_lines_xyz, self.video_axis_xyz, self.video_arcs_xyz = self.set_data_xyz()
        self.factor = 0.3
        
    def set_ax(self, ax):

        set_axes_for_landmarks_world(ax)
        ax.set_title('')
        return ax
        
    def set_arrows(self, mutation_scale=10, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0):
        
        arrows = []
        for _ in self.plot_body_axis:
            for _ in range(3):
                arrow = Arrow3D([0,0],[0,0],[0,0], mutation_scale=mutation_scale, arrowstyle=arrowstyle, color=color, shrinkA=shrinkA, shrinkB=shrinkB)
                self.ax.add_artist(arrow)
                arrows.append(arrow)
        
        return arrows
    
    def set_arcs(self):

        ax = self.ax
        arcs = []
        landmarks = self.video_landmarks[0]

        for angle in self.plot_angles:

            landmark_index = pose_landmark[angle]
            landmark_angle = landmarks[landmark_index]
            landmark_next = landmarks[landmark_index + 2]
            landmark_prev = landmarks[landmark_index - 2]
    
            if angle in body_parts_1_angle:

                v1 = points_to_vector(landmark_angle, landmark_next)
                v2 = points_to_vector(landmark_angle, landmark_prev)
                x,y,z = arc_points(v1,v2)
                arc = ax.plot(x, y, z, color = 'red')[0]
                arcs.append(arc)

            elif angle in body_parts_2_angles:

                v1 = points_to_vector(landmark_angle, landmark_next)
                l1, l2, l3 = body_reference_lines(landmarks, body_part=angle)
                v_projection = np.array([v1[0], v1[1] - np.dot(v1[1], l3[1]) * l3[1]])

                px1,py1,pz1 = vector_xyz(v_projection)
                px2,py2,pz2 = line_points(v1, v_projection)
                x,y,z = np.concatenate((px1,px2)), np.concatenate((py1,py2)), np.concatenate((pz1,pz2))
                projection_line = ax.plot(x, y, z, color = 'blue', linewidth = 1, linestyle = 'dotted')[0]
                arcs.append(projection_line)

                x,y,z = arc_points(v_projection, l1)
                arc = ax.plot(x, y, z, color = 'red')[0]

                arcs.append(arc)
                x,y,z = arc_points(v1,l3)

                arc = ax.plot(x, y, z, color = 'red')[0]
                arcs.append(arc)

        return arcs
    
    def set_plot_body_axis(self, plot_body_axis):
        
        body_parts = []
        for body_part in plot_body_axis:
            if body_part in body_axis_places:
                body_parts.append(body_part)
        
        return body_parts
    
    def set_plot_angles(self, plot_angles):
        
        angles = []
        for angle in plot_angles:
            if angle in body_parts_1_angle or angle in body_parts_2_angles:
                angles.append(angle)
        
        return angles

    def set_data_xyz(self):

        video_lines_xyz = []
        video_landmarks_xyz = []
        video_axis_xyz = []
        video_arcs_xyz = []

        for landmarks in self.video_landmarks:
            frame_lines_xyz = []
            x, y, z = points_xyz(landmarks)
            video_landmarks_xyz. append([x,y,z])

            for connection in pose_connections:
                connection0 = connection[0]
                connection1 = connection[1]
                landmark0 = landmarks[connection0]
                landmark1 = landmarks[connection1]
                xc, yc, zc = points_xyz([landmark0, landmark1])
                frame_lines_xyz.append([xc, yc, zc])
            
            video_lines_xyz.append(frame_lines_xyz)
            
            frame_axis_xyz = []

            for body_part in self.plot_body_axis:
                arrows_vectors = body_reference_lines(landmarks, body_part=body_part)

                for vector in arrows_vectors:
                    vector[1] = 0.15 * vector[1]
                    x,y,z = vector_xyz(vector)
                    frame_axis_xyz.append([x,y,z])
            
            video_axis_xyz.append(frame_axis_xyz)

            frame_arcs_xyz = []
            
            for angle in self.plot_angles:
                
                landmark_index = pose_landmark[angle]
                landmark_angle = landmarks[landmark_index]
                landmark_next = landmarks[landmark_index + 2]
                landmark_prev = landmarks[landmark_index - 2]

                if angle in body_parts_1_angle:

                    v1 = points_to_vector(landmark_angle, landmark_next)
                    v2 = points_to_vector(landmark_angle, landmark_prev)
                    x,y,z = arc_points(v1,v2)
                    frame_arcs_xyz.append([x,y,z])

                elif angle in body_parts_2_angles:
                    v1 = points_to_vector(landmark_angle, landmark_next)
                    l1, l2, l3 = body_reference_lines(landmarks, body_part=angle)
                    v_projection = np.array([v1[0], v1[1] - np.dot(v1[1], l3[1]) * l3[1]])

                    px1,py1,pz1 = vector_xyz(v_projection)
                    px2,py2,pz2 = line_points(v1, v_projection)
                    x,y,z = np.concatenate((px1,px2)), np.concatenate((py1,py2)), np.concatenate((pz1,pz2))
                    frame_arcs_xyz.append([x,y,z])

                    x,y,z = arc_points(v_projection,l1)
                    frame_arcs_xyz.append([x,y,z])

                    x,y,z = arc_points(v1,l3)
                    frame_arcs_xyz.append([x,y,z])
                
            video_arcs_xyz.append(frame_arcs_xyz)
        
        return video_landmarks_xyz, video_lines_xyz, video_axis_xyz, video_arcs_xyz

    def update_plot(self, frame):
        
        landmarks_xyz = self.video_landmarks_xyz[frame]
        lines_xyz = self.video_lines_xyz[frame]
        axis_xyz = self.video_axis_xyz[frame]
        arcs_xyz = self.video_arcs_xyz[frame]
        scatter = self.scatter
        lines = self.lines
        arrows = self.arrows
        arcs = self.arcs
        
        try:
            x, y, z = landmarks_xyz
            scatter._offsets3d = (x, y, z)
            scatter.do_3d_projection()
        except Exception as e:
            debug('In ' + self.name +  ' in update_plot while plotting landmarks: ', str(e) + '\'')
        try:
            for line, xyz in zip(lines, lines_xyz):
                x,y,z = xyz
                line.set_data(x,y)
                line.set_3d_properties(z)
        except Exception as e:
            debug('In ' + self.name +  ' in update_plot while plotting lines: ', str(e) + '\'')

        try:
            for arrow, xyz in zip(arrows, axis_xyz):
                x,y,z = xyz
                arrow.set_data(x, y, z)
                arrow.do_3d_projection()
        except Exception as e:
            debug('In ' + self.name +  ' in update_plot while plotting body axis: ' + str(e) + '\'')
        try:
            for arc, xyz in zip(arcs, arcs_xyz):
                x,y,z = xyz
                arc.set_data(x,y)
                arc.set_3d_properties(z)
        except Exception as e:
            debug('In ' + self.name +  ' in update_plot while plotting arcs: ' + str(e) + '\'')

        return [self.scatter, *self.lines, *self.arrows, *self.arcs]

class Image_Plotter_Handler:
    def __init__(self, images, ax, name = 'Video Image Plotter') -> None:
        self.images = images
        self.name = name
        self.ax = ax
        self.im = ax.imshow(self.images[0])

    def update_plot(self, frame):
        try:
            image = self.images[frame]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.im.set_array(self.images[frame])
        except Exception as e:
            debug('In ' + self.name +  ' in update_plot: ' + str(e) + '\'')
        
        return [self.im]

class Angle_Plotter_Handler:
    def __init__(self, video_angles, axs, angle_name = 'left_shoulder', name = 'Video Angles Plotter', show_velocities = True, fps = 30) -> None:
        
        self.angle_name = angle_name
        self.name = name
        self.fps = fps
        self.axs = self.set_axs(axs)
        self.video_theta, self.video_phi = self.set_angles(video_angles, angle_name)
        self.video_velocities_theta, self.video_velocities_phi = self.set_velocities(show_velocities)
        self.body_points1 = self.set_body_points1(angle_name, axs)
        self.body_points2 = self.set_body_points2(angle_name, axs)
        self.show_velocities = show_velocities

        self.lines = self.set_lines()
        self.arcs = self.set_arcs()
        self.arrows = self.set_arrows(show_velocities)
        self.legends = self.set_legends()
        self.video_lines_xy, self.video_arcs_xy, self.video_arrows_xydxdy = self.set_data(show_velocities)
        

    def set_angles(self, video_angles, angle_name):
        
        video_theta = []
        video_phi = []

        if self.angle_name in body_parts_2_angles:

            for frame_angles in video_angles:
                frame_angles = angles_absolute_to_relative(frame_angles)
                theta = frame_angles[angle_name + '_theta']
                phi = frame_angles[angle_name + '_phi']
                video_theta.append(theta)
                video_phi.append(phi)
            
            return video_theta, video_phi

        for frame_angles in video_angles:
            frame_angles = angles_absolute_to_relative(frame_angles)
            phi = frame_angles[angle_name]
            video_theta.append(np.nan)
            video_phi.append(phi)

        return video_theta, video_phi
        
    def set_axs(self,axs):
        
        angle = self.angle_name.replace('_', ' ')
        if angle[0] == 'l':
            angle = 'L' + angle[1:]
        elif angle[0] == 'r':
            angle = 'R' + angle[1:]
                
        if type(axs) != type([]):
            axs = [axs]
        
        for ax in axs:
            set_axes_for_angles(ax)
            ax.set_title(angle)

        return axs
    
    def set_body_points1(self, angle_name, axs):

        torso_height = np.sqrt(3)
        plot_reference = False

        if type(axs) != type([]):
            ax = axs
        else:
            ax = axs[0]

        if angle_name == 'left_knee' or angle_name == 'right_knee':
            if angle_name[:4] == 'left':
                knee_hip_distance_x = np.sqrt(2)/2
                knee_hip_distance_y = np.sqrt(2)/2
            else:
                knee_hip_distance_x = - np.sqrt(2)/2
                knee_hip_distance_y = np.sqrt(2)/2

            hip_point = [knee_hip_distance_x, knee_hip_distance_y,0]
            shoulder_point = [knee_hip_distance_x, knee_hip_distance_y + torso_height,0]
            body_points = [hip_point,shoulder_point]

            t = np.linspace(0, np.pi*2, 20)
            head_radious = 0.2
            xhead = head_radious*np.cos(t)
            yhead = head_radious*np.sin(t) + head_radious + torso_height + knee_hip_distance_y

            x,y,z = points_xyz(body_points)

            body_line = ax.plot(x, y, color = 'green', linewidth = 2)[0]
            head_line = ax.plot(xhead, yhead, color = 'green', linewidth = 2)[0]

            return body_points
        
        elif angle_name == 'left_hip' or angle_name == 'right_hip':
            t = np.linspace(0, np.pi*2, 20)
            origin = np.array([0,0,0])
            head_radious = 0.2
            xhead = head_radious*np.cos(t)
            yhead = head_radious*np.sin(t)
            shoulder = origin + np.array([0,np.sqrt(3),0])
            yhead += np.sqrt(3) + head_radious
            xr = [0,0]
            yr = [0,-2]

            body_points = [origin, shoulder]

            x,y,z = points_xyz(body_points)

            body_line = axs[0].plot(x, y, color = 'green', linewidth = 2)[0]
            head_line = axs[0].plot(xhead, yhead, color = 'green', linewidth = 2)[0]
            reference_line = axs[0].plot(xr, yr, color = 'blue', linestyle = 'dotted', linewidth = 1)[0]

            return body_points
            

        if angle_name == 'left_shoulder':
            left_shoulder = np.array([0,0,0])
            plot_reference = True
        elif angle_name == 'right_shoulder':
            left_shoulder = np.array([-1,0,0])
            plot_reference = True
        elif angle_name == 'left_elbow':
            elbow_shoulder_distance_x = np.sqrt(2)/2
            elbow_shoulder_distance_y = np.sqrt(2)/2
            left_shoulder = np.array([elbow_shoulder_distance_x,elbow_shoulder_distance_y,0])
        elif angle_name == 'right_elbow':
            elbow_shoulder_distance_x = -np.sqrt(2)/2 - 1
            elbow_shoulder_distance_y = np.sqrt(2)/2
            left_shoulder = np.array([elbow_shoulder_distance_x,elbow_shoulder_distance_y,0])
        elif angle_name == 'left_hip':
            left_shoulder = np.array([-0.1,torso_height,0])
            plot_reference = True
        elif angle_name == 'right_hip':
            left_shoulder = np.array([-0.9,torso_height,0])
            plot_reference = True
        else:
            left_shoulder = np.array([0,0,0])
    
        right_shoulder = left_shoulder + np.array([1,0,0])
        left_hip = left_shoulder + np.array([0.1,-torso_height,0])
        right_hip = left_shoulder + np.array([0.9, -torso_height,0])

        body_points = [left_shoulder, right_shoulder, right_hip, left_hip, left_shoulder]

        x,y,z = points_xyz(body_points)
        
        body_line = ax.plot(x, y, color = 'green', linewidth = 2)[0]

        if plot_reference:
            xr = [0,0]
            yr = [0,-2]
            reference_line = ax.plot(xr, yr, color = 'blue', linestyle = 'dotted', linewidth = 1)[0]

        return body_points[:-1]
    
    def set_body_points2(self, angle_name, axs):
      
        if angle_name in body_parts_1_angle:
            return None
        
        t = np.linspace(0, np.pi*2, 20)
        origin = np.array([0,0,0])
        head_radious = 0.2
        xhead = head_radious*np.cos(t)
        yhead = head_radious*np.sin(t)

        if angle_name == 'left_shoulder':
            shoulder = origin + np.array([1,0,0])
            xhead += 0.5
            xr = [0,-2]
            yr = [0,0]
        elif angle_name == 'right_shoulder':
            shoulder = origin + np.array([-1,0,0])
            xhead += -0.5
            xr = [0,2]
            yr = [0,0]
        elif angle_name == 'left_hip' or angle_name == 'right_hip':
            ax = axs[1]
            if angle_name[:4] == 'left':
                translate_x = 0.4
            else:
                translate_x = -0.4

            t = np.linspace(0, np.pi*2, 20)
            head_radious = 0.2
            xhead = head_radious*np.cos(t) + translate_x
            yhead = head_radious*np.sin(t)

            left_shoulder_point = [-0.5 + translate_x, 0, 0]
            right_shoulder_point = [0.5 + translate_x, 0, 0]
            body_points = [left_shoulder_point,right_shoulder_point]

            x,y,z = points_xyz(body_points)
            
            body_line = ax.plot(x, y, color = 'green', linewidth = 2)[0]
            head_line = ax.plot(xhead, yhead, color = 'green', linewidth = 2)[0]

            xr = [0,0]
            yr = [0,1]
            reference_line = ax.plot(xr, yr, color = 'blue', linestyle = 'dotted', linewidth = 1)[0]

            return body_points

        body_points = [origin, shoulder]

        x,y,z = points_xyz(body_points)

        body_line = axs[1].plot(x, y, color = 'green', linewidth = 2)[0]
        head_line = axs[1].plot(xhead, yhead, color = 'green', linewidth = 2)[0]
        reference_line = axs[1].plot(xr, yr, color = 'blue', linestyle = 'dotted', linewidth = 1)[0]

        return body_points

    def set_lines(self):

        lines = []

        for ax in self.axs:
            for _ in range(2):
                lines.append(ax.plot([], [], [], color = 'black', linewidth = 2)[0])

        return lines

    def set_arcs(self):

        arcs = []

        for ax in self.axs:
            arcs.append(ax.plot([], [], [], color = 'red', linewidth = 1, label = 'Angle: 000.0')[0])

        return arcs

    def set_legends(self):
        legends = []
        arcs = self.arcs
        arrows = self.arrows

        if self.angle_name.split('_')[1] == 'knee':
            locs = ['upper center']
        elif self.angle_name.split('_')[1] == 'hip':
            locs = ['upper center','lower center']
        elif self.angle_name in body_parts_1_angle:
            locs = ['lower center']
        else:
            locs = ['lower center','lower center']
        
        for arc, arrow, ax, loc in zip(arcs, arrows, self.axs, locs):
            if self.show_velocities:
                artists = [arc,arrow]
            else:
                artists = [arc]
            legends.append(ax.legend(handles = artists,  loc = loc))
    
        return legends
    
    def set_arrows(self, show_velocities):

        if not show_velocities:
            return []

        arrows = []

        for ax in self.axs:
            arrows.append(ax.arrow([],[],[],[] , color = 'blue', width = 0.01, label = 'velocity: 000.0'))

        return arrows
    
    def set_velocities(self, show_velocities):

        if not show_velocities:
            return None
        
        video_velocities_theta = []
        video_velocities_phi = []
        thetas = self.video_theta
        phis = self.video_phi
        fps = self.fps

        for theta0, theta1, phi0, phi1 in zip(thetas[:-1], thetas[1:], phis[:-1], phis[1:]):
            delta_theta = (theta1 - theta0)/fps
            delta_phi = (phi1 - phi0)/fps
            video_velocities_theta.append(delta_theta)
            video_velocities_phi.append(delta_phi)

        video_velocities_theta.append(video_velocities_theta[-1])
        video_velocities_phi.append(video_velocities_phi[-1])
        
        return video_velocities_theta, video_velocities_phi

    def smooth_velocities(self):

        video_velocities_theta = []
        video_velocities_phi = []
        thetas = self.video_velocities_theta
        phis = self.video_velocities_phi
        fps = self.fps

        for theta0, theta1, theta2, phi0, phi1, phi2 in zip(thetas[:-2], thetas[1:-1], thetas[2:], phis[:-2], phis[1:-1],phis[:-2]):
            smooth_theta = (theta0 + theta1 + theta2)/3
            smooth_phi = (phi0 + phi1 + phi2)/3
            video_velocities_theta.append(smooth_theta)
            video_velocities_phi.append(smooth_phi)

        
        video_velocities_theta.append(video_velocities_theta[-1])
        video_velocities_theta.insert(0,thetas[0])
        video_velocities_phi.append(video_velocities_phi[-1])
        video_velocities_phi.insert(0,phis[0])
        
        self.video_velocities_theta = video_velocities_theta
        self.video_velocities_phi = video_velocities_phi
        
        return video_velocities_theta, video_velocities_phi

    def set_data(self, show_velocities):
        
        video_lines_xy = []
        video_arcs_xy = []
        video_arrows_xydxdy = []
        t = np.linspace(0, 2*np.pi, 20)

        if self.angle_name[0] == 'l':
            reverse_x = 1
        else:
            reverse_x = -1

        for theta, phi, velocity_theta, velocity_phi in zip(self.video_theta, self.video_phi, self.video_velocities_theta, self.video_velocities_phi):

            frame_lines_xy = []
            frame_arcs_xy = []
            frame_arrows_xydxdy = []
            if self.angle_name == 'left_shoulder' or self.angle_name == 'right_shoulder':
                
                delta_phi = phi - np.pi/4
                shoulder_point = [0,0,0]
                phi_reference = [0,-1,0]
                theta_reference = [-1*reverse_x,0,0]

                elbow_x = -np.sin(phi) * reverse_x
                elbow_y = -np.cos(phi)
                elbow_point = [elbow_x, elbow_y, 0]

                hand_x =  elbow_x - np.cos(delta_phi)*reverse_x
                hand_y = elbow_y + np.sin(delta_phi)
                hand_point = [hand_x, hand_y, 0]

                arm = [shoulder_point, elbow_point, hand_point]

                x_part1,y_part1,z_part1 = points_xyz(arm)

                x_appendage1 = (0.05*np.cos(t) - np.cos(delta_phi) - np.sin(phi))*reverse_x
                y_appendage1 = 0.05*np.sin(t) + np.sin(delta_phi) - np.cos(phi)

                angle_vector1 = points_to_vector(shoulder_point, elbow_point)
                reference_vector1 = points_to_vector(shoulder_point, phi_reference)

                x_arc1,y_arc1,z_arc1 = arc_points(angle_vector1,reference_vector1, radious=0.5)

                elbow_x = -np.cos(theta)*reverse_x
                elbow_y = np.sin(theta)
                elbow_point = [elbow_x, elbow_y, 0]

                hand_x = elbow_x + np.sqrt(2)/2*elbow_x
                hand_y = elbow_y + np.sqrt(2)/2*elbow_y
                hand_point = [hand_x, hand_y, 0]

                arm = [shoulder_point, elbow_point,hand_point]

                x_part2, y_part2, z_part2 = points_xyz(arm)

                x_appendage2 = (0.05*np.cos(t) + -(np.sqrt(2)/2+1)*np.cos(theta))*reverse_x
                y_appendage2 = 0.05*np.sin(t) + (np.sqrt(2)/2+1)*np.sin(theta)

                angle_vector2 = points_to_vector(shoulder_point, elbow_point)
                reference_vector2 = points_to_vector(shoulder_point, theta_reference)

                x_arc2,y_arc2,z_arc2 = arc_points(angle_vector2,reference_vector2, radious=0.5)

            elif self.angle_name == 'left_elbow' or self.angle_name == 'right_elbow':

                shoulder_x = np.sqrt(2)/2*reverse_x
                shoulder_y = np.sqrt(2)/2

                shoulder_point = [shoulder_x, shoulder_y, 0]
                elbow_point = [0,0,0]

                hand_x = - np.cos(3*np.pi/4 - phi)*reverse_x
                hand_y =  np.sin(3*np.pi/4 - phi)
                hand_point = [hand_x, hand_y, 0]

                arm = [shoulder_point, elbow_point, hand_point]
                
                x_part1,y_part1,z_part1 = points_xyz(arm)

                x_appendage1 = (0.05*np.cos(t) - np.cos(3*np.pi/4 - phi))*reverse_x
                y_appendage1 = 0.05*np.sin(t) + np.sin(3*np.pi/4 - phi)

                angle_vector1 = points_to_vector(elbow_point, hand_point)
                reference_vector1 = points_to_vector(elbow_point, shoulder_point)

                x_arc1,y_arc1,z_arc1 = arc_points(angle_vector1, reference_vector1, radious=0.5)
            

            elif self.angle_name == 'left_hip' or self.angle_name == 'right_hip':
                
                delta_phi = phi - np.pi/4
                phi_reference = [0,1,0]
                theta_reference = [0,-1,0]
                hip_point = [0,0,0]

                knee_x = -np.sin(phi)*reverse_x
                knee_y = -np.cos(phi)
                knee_point = [knee_x, knee_y, 0]

                foot_x = knee_x
                foot_y = knee_y - 1
                foot_point = [foot_x, foot_y, 0]

                leg = [hip_point, knee_point, foot_point]

                x_part1, y_part1, z_part1 = points_xyz(leg)

                x_appendage1 = 0.05*np.cos(t)*reverse_x + foot_x
                y_appendage1 = 0.05*np.sin(t) - 0.05 + foot_y

                angle_vector1 = points_to_vector(hip_point, knee_point)
                reference_vector1 = points_to_vector(hip_point, theta_reference)

                x_arc1,y_arc1,z_arc1 = arc_points(angle_vector1,reference_vector1, radious=0.5)

                knee_x = - np.sin(theta) * reverse_x
                knee_y = np.cos(theta)
                knee_point = [knee_x, knee_y, 0]

                foot_x = knee_x
                foot_y = knee_y
                foot_point = [foot_x, foot_y, 0]

                leg = [hip_point, knee_point, foot_point]
                
                x_part2,y_part2,z_part2 = points_xyz(leg)

                x_appendage2 = 0.05*np.cos(t)*2*reverse_x + foot_x 
                y_appendage2 = 0.05*np.sin(t) + 0.05 + foot_y

                angle_vector2 = points_to_vector(hip_point, knee_point)
                reference_vector2 = points_to_vector(hip_point, phi_reference)

                x_arc2,y_arc2,z_arc2 = arc_points(angle_vector2,reference_vector2, radious=0.5)

            
            elif self.angle_name == 'left_knee' or self.angle_name == 'right_knee':
                hip_x = np.sqrt(2)/2*reverse_x
                hip_y = np.sqrt(2)/2

                hip_point = [hip_x, hip_y, 0]
                knee_point = [0,0,0]

                foot_x = np.cos(np.pi/4-phi)*reverse_x
                foot_y = np.sin(np.pi/4-phi)
                foot_point = [foot_x, foot_y, 0]

                leg = [hip_point, knee_point, foot_point]
                
                x_part1,y_part1,z_part1 = points_xyz(leg)

                x_appendage1 = (0.05*np.cos(t)-0.05)*2*reverse_x + foot_x 
                y_appendage1 = 0.05*np.sin(t) + foot_y

                angle_vector1 = points_to_vector(knee_point, foot_point)
                reference_vector1 = points_to_vector(knee_point, hip_point)

                x_arc1,y_arc1,z_arc1 = arc_points(angle_vector1, reference_vector1, radious=0.5)

            frame_lines_xy.append([x_part1,y_part1])
            frame_lines_xy.append([x_appendage1,y_appendage1])
            frame_arcs_xy.append([x_arc1,y_arc1])
            
            if show_velocities:
                velocity_vector = points_to_vector([x_arc1[1],y_arc1[1],0], [x_arc1[0],y_arc1[0],0])
                velocity_vector[1] = velocity_vector[1]/np.linalg.norm(velocity_vector[1])*velocity_phi*60
                x,y,dx,dy = x_arc1[0],y_arc1[0],velocity_vector[1][0],velocity_vector[1][1]
                frame_arrows_xydxdy.append([x,y,dx,dy])
            else:
                frame_arrows_xydxdy.append(np.nan)

            if len(self.axs) == 2:
                frame_lines_xy.append([x_part2,y_part2])
                frame_lines_xy.append([x_appendage2,y_appendage2])
                frame_arcs_xy.append([x_arc2,y_arc2])

                if show_velocities:
                    velocity_vector = points_to_vector([x_arc2[1],y_arc2[1],0], [x_arc2[0],y_arc2[0],0])
                    velocity_vector[1] = velocity_vector[1]/np.linalg.norm(velocity_vector[1])*velocity_theta*60
                    x,y,dx,dy = x_arc2[0],y_arc2[0],velocity_vector[1][0],velocity_vector[1][1]
                    frame_arrows_xydxdy.append([x,y,dx,dy])
                else:
                    frame_arrows_xydxdy.append(np.nan)
        
            video_lines_xy.append(frame_lines_xy)
            video_arcs_xy.append(frame_arcs_xy)
            video_arrows_xydxdy.append(frame_arrows_xydxdy)
            
        return video_lines_xy, video_arcs_xy, video_arrows_xydxdy
    
    def update_plot(self, frame):
        lines = self.lines
        arcs = self.arcs
        lines_xy = self.video_lines_xy[frame]
        arcs_xy = self.video_arcs_xy[frame]
        phi = self.video_phi[frame]
        theta = self.video_theta[frame]
        legends = self.legends
        velocity_theta = self.video_velocities_theta[frame]
        velocity_phi = self.video_velocities_phi[frame]
        arrows = self.arrows
        arrows_xydxdy = self.video_arrows_xydxdy[frame]

        if theta != np.nan:
            angles = (phi, theta)
        
        if self.show_velocities:
            if velocity_theta != np.nan:
                velocities = (velocity_phi, velocity_theta)
        
        try:
            for line, xy in zip(lines, lines_xy):
                x,y = xy
                line.set_data(x,y)
        except Exception as e:
            debug('In ' + self.name +  ' in update_plot while plotting lines: ', str(e) + '\'')
        try:
            for arc, xy, angle, legend in zip(arcs, arcs_xy, angles, legends):
                x,y = xy
                arc.set_data(x,y)
                arc.set_label('Angle: {:^.1f}°'.format(angle*180/np.pi))
                legend_text = legend.get_texts()[0]
                legend_text.set_text(arc.get_label())
        except Exception as e:
            debug('In ' + self.name +  ' in update_plot while plotting arcs: ' + str(e) + '\'')

        if not self.show_velocities:
            return [*self.lines, *self.arcs, *self.legends]

        try:
            for arrow, xydxdy, velocity, legend in zip(arrows, arrows_xydxdy, velocities, legends):
                x,y,dx,dy = xydxdy
                arrow.set_data(x=x,y=y,dx=dx,dy=dy)
                arrow.set_label('Velocity: {:^.1f}\'/s'.format(velocity*180/np.pi*60))
                legend_text = legend.get_texts()[1]
                legend_text.set_text(arrow.get_label())
        except Exception as e:
            debug('In ' + self.name +  ' in update_plot while plotting body velocities: ' + str(e) + '\'')


        return [*self.lines, *self.arcs, *self.arrows, *self.legends]

#Save functions and classes
def save_list(filename, landmarks):

    with open(filename + '.json', 'w') as file:
        json.dump(landmarks, file, indent=2, cls=NumpyArrayEncoder)

def load_list(filename):

    with open(filename, 'r') as file:
        landmarks = json.load(file)
    
    return landmarks

def save_video(filename, images):
    
    framesize = (images[0].shape[1], images[0].shape[0])
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap_writer = cv2.VideoWriter(filename + '.avi', fourcc, fps, framesize)
    total_frames = len(images)

    def loop(*additional_functions):
        for image in images:
    
            cap_writer.write(image)
            for func in additional_functions:
                func()
    
    try:
        progress_bar(loop, total_frames)
        return True
    except:
        debug('Error in function list_to_video')
        return False

def load_video(video_file):

    cap = cv2.VideoCapture(video_file)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    image_list = []

    def loop(*additional_functions):
        while cap.isOpened():

            success, image = cap.read()
            
            if not success:
                break
            
            image_list.append(image)

            for func in additional_functions:
                func()

    try:
        progress_bar(loop, total_frames)
        return image_list
    except Exception as e:
        debug('Error in function video_to_list', e)
        loop()
        return image_list

def ksave_analysis(video_data, folder_name = 'analysis', data_name = '' , landmarks_world_name = '', landmarks_image_name = '', video_name = ''):
    """
    Saves the data of the video analysis to a analysis folder.
    :param video_data: A tuple in the form of (landmarks_world, landmarks_image, video), as the kvideo_analysis and kframe_analysis functions returns.
    :param folder_name: Name of the folder in wich the data will be saved. If it does not exist it will be created.
    :param data_name: Optional, for saving diferent but related data in the same folder, all files will have this name plus an indicator of wich type of data is.
    :param landmarks_world_name: Optional, for saving diferent but related data in the same folder, this file will be have this name ignoring \"data_name\".
    :param landmarks_image_name: Optional, for saving diferent but related data in the same folder, this file will be have this name ignoring \"data_name\".
    :param video_name: Optional, for saving diferent but related data in the same folder, this file will be have this name ignoring \"data_name\".
    """
    
    video_landmarks_world, video_landmarks_image, video_images = video_data[0], video_data[1], video_data[2]


    if data_name == '':
        if landmarks_world_name == '':
            landmarks_world_name = 'lw'
        if landmarks_image_name == '':
            landmarks_image_name = 'li'
        if video_name == '':
            video_name = 'video'
    else:
        if landmarks_world_name == '':
            landmarks_world_name = data_name + '_lw'
        if landmarks_image_name == '':
            landmarks_image_name = data_name + '_li'
        if video_name == '':
            video_name = data_name + '_video'  

    path = os.getcwd()
    new_folder = os.path.join(path, folder_name)

    try:
        os.makedirs(new_folder)
    except Exception as e:
        debug(str(e))
    
    save_list(os.path.join(new_folder, landmarks_world_name),video_landmarks_world)
    save_list(os.path.join(new_folder,landmarks_image_name), video_landmarks_image)
    save_video(os.path.join(new_folder, video_name), video_images)

    debug('Analysis saved in ' + folder_name + '\'')

    return video_landmarks_world, video_landmarks_image, video_images

def kload_analysis(folder_name, data_name = '' , video_landmarks_world_name = '', video_landmarks_image_name = '', video_name = ''):
    """
    Load the data of the video analysis to a analysis folder.
    :param folder_name: Name of the folder in wich the data is saved.
    :param data_name: Optional, when related data was saved in the same folder to load the specific files, all files with this name will be loaded.
    :param landmarks_world_name: Optional, when related data was saved in the same folder, this will load that specific file ignoring \"data_name\".
    :param landmarks_image_name: Optional, when related data was saved in the same folder, this will load that specific file ignoring \"data_name\".
    :param video_name: Optional, when related data was saved in the same folder, this will load that specific file ignoring \"data_name\".
    """

    if data_name == '':
        if video_landmarks_world_name == '':
            video_landmarks_world_name = 'lw.json'
        if video_landmarks_image_name == '':
            video_landmarks_image_name = 'li.json'
        if video_name == '':
            video_name = 'video.avi'
    else:
        if video_landmarks_world_name == '':
            video_landmarks_world_name = data_name + '_lw.json'
        elif video_landmarks_world_name[-5:] != '.json':
            video_landmarks_world_name += 'json'
        if video_landmarks_image_name == '':
            video_landmarks_image_name = data_name + '_li.json'
        elif video_landmarks_image_name[-5:] != '.json':
            video_landmarks_image_name += 'json'
        if video_name == '':
            video_name = data_name + '_video.avi'
        elif video_name[-4:] != '.avi':
            video_name += '.avi'

    if not os.path.exists(folder_name):
        debug('kload_analysis: folder name is not the full path to folder, searching in local folder.', folder_name)
        path = os.getcwd()
        folder_name = os.path.join(path, folder_name)
        if not os.path.exists(folder_name):
            debug('kload_analysis: Could not find a folder ' + folder_name + '\'')
            return None, None, None

    video_landmarks_world_name = os.path.join(folder_name, video_landmarks_world_name)
    video_landmarks_image_name = os.path.join(folder_name, video_landmarks_image_name)
    video_name = os.path.join(folder_name, video_name)

    if not os.path.isfile(video_landmarks_world_name):
        debug('kload_analysis: Could not find file ' + video_landmarks_world_name + '\'')
        video_landmarks_world = None
    else:
        video_landmarks_world = load_list(video_landmarks_world_name)

    if not os.path.isfile(video_landmarks_image_name):
        debug('kload_analysis: Could not find file ' + video_landmarks_image_name + '\'')
        video_landmarks_image = None
    else:
        video_landmarks_image = load_list(video_landmarks_image_name)

    if not os.path.isfile(video_name):
        debug('kload_analysis: Could not find file ' + video_name + '\'')
        video_images = None
    else:
        video_images = load_video(video_name)

    debug('Analysis loaded from ' + folder_name + '\'')

    return np.array(video_landmarks_world), np.array(video_landmarks_image), video_images

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#Old functions and clases
def kvideo_analysis0(video_file: str, return_image_list = False, show: bool = False , save: bool = False, image_scale: float = 1):
    """
    Makes an motion tracking analysis of a given video file and returns the data.
    :param video_file: Path to video
    :param return_image_list: True or False. If True will save every frame in a list of cv2 images. If false, the list will be empty.
    :param show: True or False, if set to True the video will be showed in a window while analyzing.
    :param save: True or False, if set to True the video will be saved in the same folder of the original video.
    :param image_scale: Factor for resizing image. Default = 1
    :return: video_landmarks_world, video_landmarks_image, image_list
    """

    holistic = mp_holistic.Holistic(
    static_image_mode = static_image_mode,
    model_complexity = model_complexity,
    smooth_landmarks = smooth_landmarks,
    enable_segmentation = enable_segmentation,
    smooth_segmentation = smooth_segmentation,
    refine_face_landmarks = refine_face_landmarks,
    min_detection_confidence = min_detection_confidence,
    min_tracking_confidence = min_tracking_confidence)

    cap = cv2.VideoCapture(video_file)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if save:
        framesize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap_writer = cv2.VideoWriter(video_file[:-4]+'_analysis.avi',fourcc , fps, framesize)

    video_landmarks_world = []
    video_landmarks_image = []
    image_list = []

    def loop(*additional_functions):
        loop_show = show
        while cap.isOpened():

            success, image = cap.read()

            if not success:
                break
            
            frame_landmarks_world, frame_landmarks_image, image = kanalyze_image(image)
            video_landmarks_world.append(frame_landmarks_world)
            video_landmarks_image.append(frame_landmarks_image)

            if return_image_list:
                image_list.append(image)

            if save:
                cap_writer.write(image)

            for func in additional_functions:
                func()

            if not show or not loop_show:
                continue

            if image_scale <= 1:
                cv2.imshow('Body Tracking', kscale_image(image, image_scale))
            else:
                cv2.imshow('Body Tracking', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow('Body Tracking')
                loop_show = not loop_show
  
    progress_bar(loop, total_frames)

    holistic.close()
    cap.release()
    if save:
        cap_writer.release()

    debug("kvideo_analysis finished")
    return video_landmarks_world, video_landmarks_image, image_list

def kframe_analysis0(frames_folder: str, frame_initial: int, frame_final: int,  return_image_list = False, show: bool = False, save: bool = False, image_scale: float = 1):
    """
     Makes an motion tracking analysis of a series of images and returns the data.
    :param frames_folder: Path to frame folder
    :param frame_initial: First frame of secuence
    :param frame_final: Last frame of secuence
    :param image_scale: factor for resizing image. Default = 1
    :param show: True or False, if set to True the video will be showed in a window while analyzing.
    :param save: True or False, if set to True the video will be saved in the same folder of the original video.
    :return: video_landmarks_world, video_landmarks_image, image_list
    """

    holistic = mp_holistic.Holistic(
    static_image_mode = static_image_mode,
    model_complexity = model_complexity,
    smooth_landmarks = smooth_landmarks,
    enable_segmentation = enable_segmentation,
    smooth_segmentation = smooth_segmentation,
    refine_face_landmarks = refine_face_landmarks,
    min_detection_confidence = min_detection_confidence,
    min_tracking_confidence = min_tracking_confidence)
    
    frames_id = [i for i in range(frame_initial, frame_final + 1)]
    total_frames = frame_final - frame_initial + 1

    if save:
        first_image = cv2.imread(frames_folder + '/frame{:05d}.jpg'.format(frames_id[0]))
        framesize = (first_image.shape[1], first_image.shape[0])
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap_writer = cv2.VideoWriter(frames_folder+'/analysis.avi',fourcc ,fps, framesize)


    video_landmarks_world = []
    video_landmarks_image = []
    image_list = []

    def loop(*additional_functions):
        loop_show = show
        for frame_id in frames_id:

            image = cv2.imread(frames_folder + '/frame{:05d}.jpg'.format(frame_id))

            if type(image) == type(None):
                continue

            frame_landmarks_world, frame_landmarks_image, image = kanalyze_image(image)
            video_landmarks_world.append(frame_landmarks_world)
            video_landmarks_image.append(frame_landmarks_image)

            if return_image_list:
                image_list.append(image)

            if save:
                cap_writer.write(image)

            for func in additional_functions:
                func()

            if not show or not loop_show:
                continue
            
            if image_scale <= 1:
                cv2.imshow('Body Tracking', kscale_image(image, image_scale))
            else:
                cv2.imshow('Body Tracking', image)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow('Body Tracking')
                loop_show = not loop_show

    progress_bar(loop, total_frames)
    
    holistic.close()
    if save:
        cap_writer.release()
    
    debug("kframe_analysis finished")
    return video_landmarks_world, video_landmarks_image, image_list

class Landmark_Plotter_Handler0:
    def __init__(self, video_landmarks, ax, name = 'Video Landmark Plotter', plot_body_axis = ['central'], plot_angles = []) -> None:
        self.video_landmarks = video_landmarks
        self.ax = ax
        self.name = name

        self.angular_velcities = calculate_angular_velocities(video_landmarks)
        self.angles = calculate_video_angles(video_landmarks)

        self.lines = [ax.plot([], [], [], color = 'black', linewidth = 2)[0] for _ in pose_connections]
        self.scatter = ax.scatter3D([],[],[], color='k')
        self.plot_body_axis = self.set_plot_body_axis(plot_body_axis)
        self.plot_angles = self.set_plot_angles(plot_angles)
        self.arrows = self.set_arrows()
        self.arcs = self.set_arcs()
        self.line_data = None
        self.factor = 0.3
        
    def set_arrows(self, mutation_scale=10, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0):
        
        arrows = []
        for _ in self.plot_body_axis:
            for _ in range(3):
                arrow = Arrow3D([0,0],[0,0],[0,0], mutation_scale=mutation_scale, arrowstyle=arrowstyle, color=color, shrinkA=shrinkA, shrinkB=shrinkB)
                self.ax.add_artist(arrow)
                arrows.append(arrow)
        
        return arrows
    
    def set_arcs(self):

        ax = self.ax
        arcs = []
        landmarks = self.video_landmarks[0]

        for angle in self.plot_angles:

            landmark_index = pose_landmark[angle]
            landmark_angle = landmarks[landmark_index]
            landmark_next = landmarks[landmark_index + 2]
            landmark_prev = landmarks[landmark_index - 2]
    
            if angle in body_parts_1_angle:

                v1 = points_to_vector(landmark_angle, landmark_next)
                v2 = points_to_vector(landmark_angle, landmark_prev)
                x,y,z = arc_points(v1,v2)
                arc = ax.plot(x, y, z, color = 'red')[0]
                arcs.append(arc)

            elif angle in body_parts_2_angles:

                v1 = points_to_vector(landmark_angle, landmark_next)
                l1, l2, l3 = body_reference_lines(landmarks, body_part=angle)
                v_projection = np.array([v1[0], v1[1] - np.dot(v1[1], l3[1]) * l3[1]])

                px1,py1,pz1 = vector_xyz(v_projection)
                px2,py2,pz2 = line_points(v1, v_projection)
                x,y,z = np.concatenate((px1,px2)), np.concatenate((py1,py2)), np.concatenate((pz1,pz2))
                projection_line = ax.plot(x, y, z, color = 'blue', linewidth = 1, linestyle = 'dotted')[0]
                arcs.append(projection_line)

                x,y,z = arc_points(v_projection, l1)
                arc = ax.plot(x, y, z, color = 'red')[0]

                arcs.append(arc)
                x,y,z = arc_points(v1,l3)

                arc = ax.plot(x, y, z, color = 'red')[0]
                arcs.append(arc)

        return arcs
    
    def set_plot_body_axis(self, plot_body_axis):
        
        body_parts = []
        for body_part in plot_body_axis:
            if body_part in body_axis_places:
                body_parts.append(body_part)
        
        return body_parts
    
    def set_plot_angles(self, plot_angles):
        
        angles = []
        for angle in plot_angles:
            if angle in body_parts_1_angle or angle in body_parts_2_angles:
                angles.append(angle)
        
        return angles


    def update_plot(self, frame):
            
            landmarks = self.video_landmarks[frame]

            try:
                x, y, z = points_xyz(landmarks)
                self.scatter._offsets3d = (x, y, z)
                self.scatter.do_3d_projection()
            except Exception as e:
                debug('In ' + self.name +  ' in update_plot while plotting landmarks: ', str(e) + '\'')

            try:
                for line, connection in zip(self.lines, pose_connections):
                    connection0 = connection[0]
                    connection1 = connection[1]
                    landmark0 = landmarks[connection0]
                    landmark1 = landmarks[connection1]
                    xc, yc, zc = points_xyz([landmark0, landmark1])
                    line.set_data(xc,yc)
                    line.set_3d_properties(zc)
            except Exception as e:
                debug('In ' + self.name +  ' in update_plot while plotting lines: ', str(e) + '\'')

            try:
                axis_to_plot = []
                for body_part in self.plot_body_axis:
                    # body_axis_vectors = body_axis(landmarks, body_part=body_part)
                    body_axis_vectors = body_reference_lines(landmarks, body_part=body_part)
                    for body_axis_vector in body_axis_vectors:
                        axis_to_plot.append(body_axis_vector)

                for arrow, e in zip(self.arrows, axis_to_plot):
                    start = e[0]
                    end = e[1] * self.factor + e[0]
                    arrow.set_data([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
                    arrow.do_3d_projection()
            except Exception as e:
                debug('In ' + self.name +  ' in update_plot while plotting body axis: ' + str(e) + '\'')

            try:
                
                i = 0
                for angle in self.plot_angles:
                    
                    landmark_index = pose_landmark[angle]
                    landmark_angle = landmarks[landmark_index]
                    landmark_next = landmarks[landmark_index + 2]
                    landmark_prev = landmarks[landmark_index - 2]
                    arcs = self.arcs

                    if angle in body_parts_1_angle:

                        v1 = points_to_vector(landmark_angle, landmark_next)
                        v2 = points_to_vector(landmark_angle, landmark_prev)
                        x,y,z = arc_points(v1,v2)
                        self.arcs[i].set_data(x,y)
                        self.arcs[i].set_3d_properties(z)
                        i += 1

                    elif angle in body_parts_2_angles:
                        v1 = points_to_vector(landmark_angle, landmark_next)
                        l1, l2, l3 = body_reference_lines(landmarks, body_part=angle)
                        v_projection = np.array([v1[0], v1[1] - np.dot(v1[1], l3[1]) * l3[1]])

                        px1,py1,pz1 = vector_xyz(v_projection)
                        px2,py2,pz2 = line_points(v1, v_projection)
                        x,y,z = np.concatenate((px1,px2)), np.concatenate((py1,py2)), np.concatenate((pz1,pz2))
                        arcs[i].set_data(x,y)
                        arcs[i].set_3d_properties(z)
                        i += 1

                        x,y,z = arc_points(v_projection,l1)
                        arcs[i].set_data(x,y)
                        arcs[i].set_3d_properties(z)
                        i += 1

                        x,y,z = arc_points(v1,l3)
                        arcs[i].set_data(x,y)
                        arcs[i].set_3d_properties(z)
                        i += 1

            except Exception as e:
                debug('In ' + self.name +  ' in update_plot while plotting arcs: ' + str(e) + '\'')

            return [self.scatter, *self.lines, *self.arrows, *self.arcs]