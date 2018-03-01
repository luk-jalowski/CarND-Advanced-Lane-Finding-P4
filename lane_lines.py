import calibration as cal
import pipeline as pipe
import numpy as np
import glob
import cv2
from moviepy.editor import VideoFileClip

#Import calibration images from camera_calibration folder
cal_images = []
list_images = glob.glob('./camera_calibration/calibration*.jpg')

for image_path in list_images:
    image = cv2.imread(image_path)
    cal_images.append(image)

print(list_images)
#Calculate camera distortion coefficients
ret, mtx, dist, rvecs, tvecs = cal.calibrate_camera(cal_images, 9, 6)

img_height, img_width = image.shape[:2]

source = np.float32([(575,460),
                  (710,460), 
                  (260,680), 
                  (1060,680)])
offset = 350
destination = np.float32([(offset, 0),
                  (img_width-offset, 0),
                  (offset, img_height),
                  (img_width-offset, img_height)])
    


pipe.setup(mtx, dist, source, destination)

video_output1 = 'output_images/project_video_output.mp4'
video_input1 = VideoFileClip('project_video.mp4')
processed_video = video_input1.fl_image(pipe.process_frame)
processed_video.write_videofile(video_output1, audio=False)



pipe.setup(mtx, dist, source, destination)

video_output2 = 'output_images/challenge_video_output.mp4'
video_input2 = VideoFileClip('challenge_video.mp4')
processed_video = video_input1.fl_image(pipe.process_frame)
processed_video.write_videofile(video_output2, audio=False)



pipe.setup(mtx, dist, source, destination)

video_output2 = 'output_images/harder_challenge_video_output.mp4'
video_input2 = VideoFileClip('harder_challenge_video.mp4')
processed_video = video_input1.fl_image(pipe.process_frame)
processed_video.write_videofile(video_output2, audio=False)
