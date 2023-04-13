from KMT import *
# from ultralytics import YOLO
# import matplotlib
# matplotlib.use('tkagg')

# model = YOLO("yolov8n.pt")

futbol_folder = "D:/Kauel/KMT/videos/futbol_frames"
golf_folder = "D:/Kauel/KMT/videos/golf_frames"

lws, lis, ims = kload_analysis('futbol_analysis')

kplot_angles(lws,ims)

