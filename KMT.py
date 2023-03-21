import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3
import json
import inspect
import re
import time
import os

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
    'RIGHT_FOOT_INDEX' : 32
}

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
#Debugging functions

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


#Process Bar functions
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

#Utility functions
def subplot_grid(num):
    row = int(round(np.sqrt(num)))
    col = int(np.ceil(np.sqrt(num)))
    return row, col

def is_landmark(landmark):
    if type(landmark) == type(None):
        return False
    elif type(landmark) == type([]):
        return True
    elif type(landmark) == type(np.array([])):
        return True
    else:
        return False

#Maths and Matrix functions
def landmarks_average(*landmarks):

    total_landmarks = len(landmarks)
    average_landmarks = []

    for frame in zip(*landmarks):
        average_landmark = np.array([[0] * 33, [0] * 33, [0] * 33])
        invalid_landmarks = 0
        for landmark in frame:
            if type(landmark) != type(None):
                average_landmark = average_landmark + np.array(landmark)
            else:
                invalid_landmarks += 1
                
        
        if type(average_landmark) != type(None) and total_landmarks > invalid_landmarks:
            average_landmarks.append(average_landmark/(total_landmarks - invalid_landmarks))
        else:
            average_landmarks.append(None)
    
    return average_landmarks

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

#Core functions
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
        return None, None, image

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
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

#Plot functions
def kplot_landmarks_world(landmarks_world, image_list = [], save = False, filename = 'D:/Kauel/KMT/videos/animation.gif'):
    """
    Plots the landmarks in a 3d grid, aditionally plots the image used to make the data beside for comparation.
    :param landmarks_world: List of landmarks to animate.
    :param image_list: list of cv2 images from witch the landmarks where calculated to put beside the animation.
    :save: True or False. If true the video will be recorded as a .gif
    :return: True
    """
    image_list_size = len(image_list)
    landmarks_world_size = len(landmarks_world)

    #Make the matplotlib figure to create graphs and plots
    fig = plt.figure()

    #Makes the axes for ploting the landmarks and image if required.
    if image_list_size == landmarks_world_size:
        plot_image = True
        ax = fig.add_subplot(1,2,1,projection="3d")
        ax2 = fig.add_subplot(1,2,2)
        im = ax2.imshow(image_list[0])
    elif image_list_size != 0:
        plot_image = False
        print('Size Error: The list of landmarks doesn\'t coincide with the list of images.')
        ax = fig.add_subplot(projection="3d")
    else:
        plot_image = False
        ax = fig.add_subplot(projection="3d")
    
    #Settings of the plot
    ax.set_title('Landmarks Positions')
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

    def update_plot(frame):
        
        if type(landmarks_world[frame]) != type(None):
            x = landmarks_world[frame][0]
            y = landmarks_world[frame][1]
            z = landmarks_world[frame][2]
            scatter_plot._offsets3d = (x, z, y)

            for line, connection in zip(lines, pose_connections):
                line.set_data([x[connection[0]],x[connection[1]]],[z[connection[0]],z[connection[1]]])
                line.set_3d_properties([y[connection[0]],y[connection[1]]])

        if show_alive_bar and save:
            bar()

        if plot_image:
            image = image_list[frame]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im.set_array(image_list[frame])
            return [im]

    ani = animation.FuncAnimation(fig, update_plot, landmarks_world_size, interval=10)

    if save:
        if show_alive_bar:
            with alive_bar(landmarks_world_size) as bar:
                ani.save(filename, writer='imagemagick', fps=30)
        else:
            ani.save(filename, writer='imagemagick', fps=30)

    plt.show()
    return True

def kplot_landmarks(*landmarks_set, save = False, filename = 'compare_plot.gif'):
    """
    Plots landmarks.
    :param *landmarks_set: Any amount of landmarks to animate.
    :save: True or False. If true the video will be recorded as a .gif
    :return: True
    """
    
    for landmarks, index in zip(landmarks_set, range(len(landmarks_set))):
        if landmarks == None:
            debug('error in kplot_landmarks: list ' + str(index) + ' is empty.')
            return False

    ammount_of_plots = len(landmarks_set)
    total_frames = len(max(landmarks_set, key=len))
    rows, columns = subplot_grid(ammount_of_plots)

    fig = plt.figure()

    lines_list = []
    scatter_list = []

    index = 1
    for _ in landmarks_set:
        ax = fig.add_subplot(rows, columns, index, projection="3d")
        ax.set_title('Landmarks Positions {}'.format(index))
        scatter_list.append(ax.scatter3D([],[],[], color='m'))
        ax.set_xticks(np.arange(-1, 1, 0.25))
        ax.set_yticks(np.arange(-1, 1, 0.25))
        ax.set_zticks(np.arange(-1, 1, 0.25))

        ax.axes.set_xlim3d(left=-1, right=1) 
        ax.axes.set_ylim3d(bottom=-1, top=1) 
        ax.axes.set_zlim3d(bottom=-1, top=1) 

        ax.set_box_aspect(aspect = (1,1,1))

        ax.set(xlabel='X')
        ax.set(ylabel='Z')
        ax.set(zlabel='Y')
        index += 1

        lines_list.append([ax.plot([], [], [])[0] for _ in pose_connections])

    def update_plot(frame):
        
        for landmarks, scatter_plot, lines in zip(landmarks_set, scatter_list, lines_list):
            try:
                x = landmarks[frame][0]
                y = landmarks[frame][1]
                z = landmarks[frame][2]
                scatter_plot._offsets3d = (x, z, y)

                for line, connection in zip(lines, pose_connections):
                    line.set_data([x[connection[0]],x[connection[1]]],[z[connection[0]],z[connection[1]]])
                    line.set_3d_properties([y[connection[0]],y[connection[1]]])

            except Exception as e:

                debug('In kplot_landmarks in update_plot: ', e)
        if show_alive_bar and save:
            bar()

    ani = animation.FuncAnimation(fig, update_plot, total_frames, interval=10)

    if save:
        if show_alive_bar:
            with alive_bar(total_frames) as bar:
                ani.save(filename, writer='imagemagick', fps=30)
        else:
            ani.save(filename, writer='imagemagick', fps=30)
    
    plt.show()
    return True

def kplot_compare(*landmarks_set, save = False, filename = 'compare.gif'):
    """
    Plots landmarks.
    :param *landmarks_set: Any amount of landmarks to animate.
    :save: True or False. If true the video will be recorded as a .gif
    :return: True
    """
    
    for landmarks, index in zip(landmarks_set, range(len(landmarks_set))):
        if landmarks == None:
            debug('error in kplot_landmarks: list ' + str(index) + ' is empty.')
            return False

    total_frames = len(max(landmarks_set, key=len))

    fig = plt.figure()

    lines_list = []
    scatter_list = []
    ax = fig.add_subplot(1,2,1, projection="3d")
    ax.set_title('Landmarks Positions')
    ax.set_xticks(np.arange(-1, 1, 0.25))
    ax.set_yticks(np.arange(-1, 1, 0.25))
    ax.set_zticks(np.arange(-1, 1, 0.25))

    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1, top=1) 

    ax.set_box_aspect(aspect = (1,1,1))

    ax.set(xlabel='X')
    ax.set(ylabel='Z')
    ax.set(zlabel='Y')

    color_index = 0
    for landmarks in landmarks_set:
        scatter_list.append(ax.scatter3D([],[],[], color=list(plt_colors.keys())[color_index]))
        lines_list.append([ax.plot([], [], [])[0] for _ in pose_connections])
        color_index += 1

        if color_index >= len(plt_colors):
            color_index = 0
    

    average = landmarks_average(*landmarks_set)
    # errors = landmarks_errors(average, *landmarks_set)

    ax2 = fig.add_subplot(1,2,2, projection="3d")
    ax2.set_title('Average Landmarks Positions')
    ax2.set_xticks(np.arange(-1, 1, 0.25))
    ax2.set_yticks(np.arange(-1, 1, 0.25))
    ax2.set_zticks(np.arange(-1, 1, 0.25))

    ax2.axes.set_xlim3d(left=-1, right=1) 
    ax2.axes.set_ylim3d(bottom=-1, top=1) 
    ax2.axes.set_zlim3d(bottom=-1, top=1) 

    ax2.set_box_aspect(aspect = (1,1,1))

    ax2.set(xlabel='X')
    ax2.set(ylabel='Z')
    ax2.set(zlabel='Y')

    average_scatter_plot = ax2.scatter3D([],[],[], color='m', )
    
    average_lines = [ax2.plot([], [], [])[0] for _ in pose_connections]
    # text = ax2.text2D(0.5,-0.1, '', color='r')
    # legend = ax2.legend(pose_landmark.keys())

    def update_plot(frame):
        
        for landmarks, scatter_plot, lines in zip(landmarks_set, scatter_list, lines_list):
            try:
                x = landmarks[frame][0]
                y = landmarks[frame][1]
                z = landmarks[frame][2]
                scatter_plot._offsets3d = (x, z, y)

                for line, connection in zip(lines, pose_connections):
                    line.set_data([x[connection[0]],x[connection[1]]],[z[connection[0]],z[connection[1]]])
                    line.set_3d_properties([y[connection[0]],y[connection[1]]])

            except Exception as e:

                debug('In kplot_landmarks in update_plot: ', e)
        
        try:
            x = average[frame][0]
            y = average[frame][1]
            z = average[frame][2]
            average_scatter_plot._offsets3d = (x, z, y)

            for line, connection in zip(average_lines, pose_connections):
                line.set_data([x[connection[0]],x[connection[1]]],[z[connection[0]],z[connection[1]]])
                line.set_3d_properties([y[connection[0]],y[connection[1]]])

            # new_text = 'Position Errors:'

            # for error, pose_name in zip(errors[frame], pose_landmark.keys()):
            #     new_text += '\n{}: {:.0%}'.format(pose_name, error)

            # text.set_text(new_text)

        except Exception as e:

            debug('In kplot_landmarks in update_plot: ', e)

        if show_alive_bar and save:
            bar()

    ani = animation.FuncAnimation(fig, update_plot, total_frames, interval=10)

    if save:
        if show_alive_bar:
            with alive_bar(total_frames) as bar:
                ani.save(filename, writer='imagemagick', fps=30)
        else:
            ani.save(filename, writer='imagemagick', fps=30)
    
    plt.show()
    return True

#Core functions old
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

#Save functions
def save_list(filename, landmarks):

    with open(filename + '.json', 'w') as file:
        json.dump(landmarks, file, indent=2)

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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if not success:
                break
        
            image_list.append(image)

            for func in additional_functions:
                func()

    try:
        progress_bar(loop, total_frames)
        return image_list
    except:
        debug('Error in function video_to_list')
        return

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
    
    landmarks_world, landmarks_image, video = video_data[0], video_data[1], video_data[2]


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
    
    save_list(os.path.join(new_folder, landmarks_world_name), landmarks_world)
    save_list(os.path.join(new_folder,landmarks_image_name), landmarks_image)
    save_video(os.path.join(new_folder, video_name), video)

    debug('Analysis saved in ' + folder_name + '\'')

    return landmarks_world, landmarks_image, video

def kload_analysis(folder_name, data_name = '' , landmarks_world_name = '', landmarks_image_name = '', video_name = ''):
    """
    Load the data of the video analysis to a analysis folder.
    :param folder_name: Name of the folder in wich the data is saved.
    :param data_name: Optional, when related data was saved in the same folder to load the specific files, all files with this name will be loaded.
    :param landmarks_world_name: Optional, when related data was saved in the same folder, this will load that specific file ignoring \"data_name\".
    :param landmarks_image_name: Optional, when related data was saved in the same folder, this will load that specific file ignoring \"data_name\".
    :param video_name: Optional, when related data was saved in the same folder, this will load that specific file ignoring \"data_name\".
    """

    if data_name == '':
        if landmarks_world_name == '':
            landmarks_world_name = 'lw.json'
        if landmarks_image_name == '':
            landmarks_image_name = 'li.json'
        if video_name == '':
            video_name = 'video.avi'
    else:
        if landmarks_world_name == '':
            landmarks_world_name = data_name + '_lw.json'
        elif landmarks_world_name[-5:] != '.json':
            landmarks_world_name += 'json'
        if landmarks_image_name == '':
            landmarks_image_name = data_name + '_li.json'
        elif landmarks_image_name[-5:] != '.json':
            landmarks_image_name += 'json'
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

    landmarks_world_name = os.path.join(folder_name, landmarks_world_name)
    landmarks_image_name = os.path.join(folder_name, landmarks_image_name)
    video_name = os.path.join(folder_name, video_name)

    if not os.path.isfile(landmarks_world_name):
        debug('kload_analysis: Could not find file ' + landmarks_world_name + '\'')
        video_landmarks_world = None
    else:
        video_landmarks_world = load_list(landmarks_world_name)

    if not os.path.isfile(landmarks_image_name):
        debug('kload_analysis: Could not find file ' + landmarks_image_name + '\'')
        landmarks_image = None
    else:
        landmarks_image = load_list(landmarks_image_name)

    if not os.path.isfile(video_name):
        debug('kload_analysis: Could not find file ' + video_name + '\'')
        video = None
    else:
        video = load_video(video_name)

    debug('Analysis loaded from ' + folder_name + '\'')

    return video_landmarks_world, landmarks_image, video
