import calibration as cal
import pipeline as pipe
import numpy as np
import matplotlib.pyplot as plt
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


#cv2.imwrite("example_images/cal15_undistorted.jpg", cal.undistortImage(cal_images[15], mtx, dist))


exampleImg = cv2.imread('./test_images/test5.jpg')
exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)

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
    
processed_image, Minv = pipe.get_aerial_view(exampleImg, mtx, dist, source, destination)

binary_image, Minv = pipe.process_image(exampleImg)

fig, (ax1,ax2, ax3) = plt.subplots(3,1, figsize=(16, 11)) 
ax1.imshow(processed_image)
ax2.imshow(binary_image, cmap='gray')

left_fit, right_fit, left_lane_inds, right_lane_inds = pipe.sliding_window_polyfit(binary_image)
#left_fit, right_fit, left_lane_inds, right_lane_inds = pipe.polyfit_with_previous_frame(
#                                                              binary_image3,
#                                                              left_fit, right_fit,
#                                                              left_lane_inds,
#                                                              right_lane_inds)



left_curvature, right_curvature, offset_center = pipe.calculate_curvature_and_offset(
                                            binary_image, left_fit, right_fit, left_lane_inds, right_lane_inds)


#result = pipe.fill_lane(exampleImg2, binary_image2, left_fit, right_fit, Minv)
#result = pipe.display_info(result, (left_curvature+right_curvature)/2, offset_center)
#plt.imshow(result)
pipe.setup(mtx, dist, source, destination)
result = pipe.process_frame(exampleImg)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
cv2.imwrite("example_images/test5_result.jpg", result)

plt.imshow(result)


pipe.setup(mtx, dist, source, destination)

#my_clip.write_gif('test.gif', fps=12)
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
