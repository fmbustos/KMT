import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3
from KMT import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture('D:/Kauel/KMT/videos/probanding_cut.mp4')
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("CAP_PROP_FPS : '{}'".format(cap.get(cv2.CAP_PROP_FPS)))

frames_id = [i for i in range(471, 553)]

#################################################################################################################################################################

# for frame_id in frames_id:

#     image = cv2.imread('D:/Kauel/KMT/videos/probanding_cut_frames/frame{:05d}.jpg'.format(frame_id))

#     image.flags.writeable = False
#     results = holistic.process(image)

#     landmarks_list = results.pose_world_landmarks.landmark


#     font = cv2.FONT_HERSHEY_SIMPLEX
#     x = []
#     y = []
    # for landmark in landmarks_list:
    #     x.append(landmark.x)
    #     y.append(landmark.y)
    #     # cv2.circle(image, (int(image.shape[1]*landmark.x),int(image.shape[0]*landmark.y)), 10, (0,0,0), 3)
    #     print(landmark)

    # for connection in mp_holistic.POSE_CONNECTIONS:
    #     p1 = (int(image.shape[1] * x[connection[0]]), int(image.shape[0]*y[connection[0]]))
    #     p2 = (int(image.shape[1]*x[connection[1]]), int(image.shape[0]*y[connection[1]]))
    #     # cv2.line(image, p1,p2,(0,0,0), 10)


#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_holistic.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    

#     cv2.imshow('probanding1',image)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         cv2.destroyWindow('probanding1')
#         break

#################################################################################################################################################################

# image = cv2.imread('D:/Kauel/KMT/videos/probanding_cut_frames/frame00450.jpg')

# image.flags.writeable = False
# results = holistic.process(image)

# landmarks_list = results.pose_world_landmarks.landmark


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.grid()


# x = []
# y = []
# z = []
# for landmark in landmarks_list:
#     x.append(landmark.x)
#     y.append(-landmark.y)
#     z.append(landmark.z)

# for connection in mp_holistic.POSE_CONNECTIONS:
#     p1 = (x[connection[0]], y[connection[0]], z[connection[0]])
#     p2 = (x[connection[1]], y[connection[1]], z[connection[1]])
#     ax.plot([x[connection[0]],x[connection[1]]], [z[connection[0]],z[connection[1]]],[y[connection[0]],y[connection[1]]])


# ax.scatter(x, z, y, c = 'r', s = 50)
# ax.scatter(0,0,0, c = 'b', s=100)
# ax.set(xlabel='X, frame')
# ax.set(ylabel='Y, profundidad')
# ax.set(zlabel='Z, altura')

# plt.show()

##################################################################################################################################################################33
# data = []
# data_image = []
# right_shoulder = []
# for frame_id in frames_id:

#     image = cv2.imread('D:/Kauel/KMT/videos/probanding_cut_frames/frame{:05d}.jpg'.format(frame_id))

#     image.flags.writeable = False
#     results = holistic.process(image)

#     landmarks_list = results.pose_world_landmarks.landmark
#     landmarks_image_list = results.pose_landmarks.landmark


#     font = cv2.FONT_HERSHEY_SIMPLEX
#     x = []
#     y = []
#     z = []
#     x_i = []
#     y_i = []
#     z_i = []
#     for landmark, landmark_i in zip(landmarks_list,landmarks_image_list):
#         x.append(landmark.x)
#         y.append(-landmark.y)
#         z.append(landmark.z)
#         x_i.append(landmark_i.x)
#         y_i.append(landmark_i.y)
#         z_i.append(landmark_i.z)



#     # for connection in mp_holistic.POSE_CONNECTIONS:
#     #     p1 = (x[connection[0]], y[connection[0]], z[connection[0]])
#     #     p2 = (x[connection[1]], y[connection[1]], z[connection[1]])

    
#     data.append([x,y,z])
#     data_image.append([x_i,y_i,z_i])
#     right_shoulder.append([x[pose_landmark['RIGHT_SHOULDER']],y[pose_landmark['RIGHT_SHOULDER']],z[pose_landmark['RIGHT_SHOULDER']]])

# fig = plt.figure()
# ax = fig.add_subplot(1,2,1,projection="3d")
# ax.set_xticks(np.arange(-0.5, 0.5, 0.1))
# ax.set_yticks(np.arange(-0.5, 0.5, 0.1))
# ax.set_zticks(np.arange(-1, 1, 0.1))

# ax.axes.set_xlim3d(left=-0.5, right=0.5) 
# ax.axes.set_ylim3d(bottom=-0.5, top=0.5) 
# ax.axes.set_zlim3d(bottom=-1, top=1) 
# ax2 = fig.add_subplot(1,2,2)
# ax.set(xlabel='X')
# ax.set(ylabel='Z, profundidad')
# ax.set(zlabel='Y, altura')

# lines = [ax.plot([], [], [])[0] for _ in data]

# def update_lines(num):
#     ax.clear()
#     ax.set(xlabel='X, frame{}'.format(num))
#     ax.set(ylabel='Z, profundidad')

#     ax.set(zlabel='Y, altura')
#     ax.set_xticks(np.arange(-0.5, 0.5, 0.1))
#     ax.set_yticks(np.arange(-0.5, 0.5, 0.1))
#     ax.set_zticks(np.arange(-1, 1, 0.1))

#     ax.axes.set_xlim3d(left=-0.5, right=0.5) 
#     ax.axes.set_ylim3d(bottom=-0.5, top=0.5) 
#     ax.axes.set_zlim3d(bottom=-1, top=1) 
#     x = data[num][0]
#     y = data[num][1]
#     z = data[num][2]
#     x_i = data_image[num][0]
#     y_i = data_image[num][1]
#     z_i = data_image[num][2]
#     # ax.scatter(data[num][0], data[num][1], data[num][2], c = 'r', s = 50)
#     ax.scatter(0,0,0, c = 'b', s=100)

#     for connection in mp_holistic.POSE_CONNECTIONS:
#         ax.plot([x[connection[0]],x[connection[1]]],[z[connection[0]],z[connection[1]]],[y[connection[0]],y[connection[1]]])
    
#     # if num > 0:
#     #     for i in range(num):
#     #         ax.plot([right_shoulder[i-1][0],right_shoulder[i][0]],[right_shoulder[i-1][2],right_shoulder[i][2]],[right_shoulder[i-1][1],right_shoulder[i][1]], c='r')

    # image = cv2.imread('D:/Kauel/KMT/videos/probanding_cut_frames/frame{:05d}.jpg'.format(frames_id[num]))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # for x,y in zip(x_i, y_i):
    #     cv2.circle(image, (int(image.shape[1]*x),int(image.shape[0]*y)), 10, (0,0,0), 3)

    # for connection in mp_holistic.POSE_CONNECTIONS:
    #     p1 = (int(image.shape[1] * x_i[connection[0]]), int(image.shape[0]*y_i[connection[0]]))
    #     p2 = (int(image.shape[1]*x_i[connection[1]]), int(image.shape[0]*y_i[connection[1]]))
    #     cv2.line(image, p1,p2,(0,0,0), 10)

#     ax2.clear()
#     ax2.imshow(image)

# pause = False

# def onClick(event):
#     global pause
#     pause = True
# fig.canvas.mpl_connect('button_press_event', onClick)
# ani = animation.FuncAnimation(
#     fig, update_lines, len(frames_id), interval=10)

# plt.show()

##################################################################################################################################################################

data = []
data_image = []
images = []
for frame_id in frames_id:

    image = cv2.imread('D:/Kauel/KMT/videos/probanding_cut_frames/frame{:05d}.jpg'.format(frame_id))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = holistic.process(image)

    landmarks_list = results.pose_world_landmarks.landmark
    landmarks_image_list = results.pose_landmarks.landmark

    x = []
    y = []
    z = []
    x_i = []
    y_i = []
    z_i = []
    for landmark, landmark_i in zip(landmarks_list,landmarks_image_list):
        x.append(landmark.x)
        y.append(-landmark.y)
        z.append(landmark.z)
        x_i.append(landmark_i.x)
        y_i.append(landmark_i.y)
        z_i.append(landmark_i.z)
        cv2.circle(image, (int(image.shape[1]*landmark_i.x),int(image.shape[0]*landmark_i.y)), 10, (0,0,0), 3)

    for connection in mp_holistic.POSE_CONNECTIONS:
        p1 = (int(image.shape[1] * x_i[connection[0]]), int(image.shape[0]*y_i[connection[0]]))
        p2 = (int(image.shape[1]*x_i[connection[1]]), int(image.shape[0]*y_i[connection[1]]))
        cv2.line(image, p1,p2,(0,0,0), 10)



    # for connection in mp_holistic.POSE_CONNECTIONS:
    #     p1 = (x[connection[0]], y[connection[0]], z[connection[0]])
    #     p2 = (x[connection[1]], y[connection[1]], z[connection[1]])

    
    data.append([x,y,z])
    data_image.append([x_i,y_i,z_i])
    images.append(image)

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

ax2 = fig.add_subplot(1,2,2)
ax.set(xlabel='X')
ax.set(ylabel='Z, profundidad')
ax.set(zlabel='Y, altura')

lines = [ax.plot([], [], [])[0] for _ in pose_connections]

def update_plot(num, lines):

    image = images[num]
    x = data[num][0]
    y = data[num][1]
    z = data[num][2]
    x_i = data_image[num][0]
    y_i = data_image[num][1]

    scatter_plot._offsets3d = (x, z, y)

    for line, connection in zip(lines, pose_connections):
        line.set_data([x[connection[0]],x[connection[1]]],[z[connection[0]],z[connection[1]]])
        line.set_3d_properties([y[connection[0]],y[connection[1]]])


    ax2.clear()
    ax2.imshow(image)

    return lines



ani = animation.FuncAnimation(
    fig, update_plot, len(frames_id) ,fargs=([lines]), interval=10)

# ani.save('D:/Kauel/KMT/videos/animation.gif', writer='imagemagick', fps=30)

plt.show()


