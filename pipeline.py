import calibration as cal
import cv2
import numpy as np
from lane import Line
from camera import Camera

def setup( mtx, dist, source, destination):
    #initialize line variables
    global camera
    global left_line
    global right_line

    camera = Camera()
    left_line = Line()
    right_line = Line()
    
    camera.mtx = mtx
    camera.dist = dist
    camera.source = source
    camera.destination = destination
    return

def process_image(image):
    #return image of road lanes from top view
    processed_image, Minv = get_aerial_view(image, camera.mtx, camera.dist, camera.source, camera.destination)
    yuv = cv2.cvtColor(processed_image, cv2.COLOR_RGB2YUV)
    
    Y = yuv[:,:,0]
    V = yuv[:,:,2]
    
    Y_binary = np.zeros_like(Y)
    Y_binary[(Y >=200) & (Y<=255)] = 1
    
    V_binary = np.zeros_like(V)
    V_binary[(V >=0) & (V<=90)] = 1
    
    combined_image = np.zeros_like(Y)
    combined_image[(Y_binary == 1) | (V_binary == 1)] = 1
    return combined_image, Minv
    
def get_aerial_view(image, mtx, dist, source, destination):
    #returns image viewed from top
    undistorted = cal.undistortImage(image, mtx, dist)
    warped, M, Minv = cal.unwrap(undistorted, source, destination)
    return warped, Minv

def sliding_window_polyfit(binary_image):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_image[binary_image.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_image, binary_image, binary_image))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(binary_image.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_image.shape[0] - (window+1)*window_height
        win_y_high = binary_image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_lane_inds, right_lane_inds

def polyfit_with_previous_frame(binary_warped,
                                prev_left_fit, prev_right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + 
    prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + 
    prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin))) 
    
    right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + 
    prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + 
    prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds


def calculate_curvature_and_offset(binary_image, left_fit, right_fit, left_lane_inds, right_lane_inds):
    #calculates lane curvature and vehicle offset from the center of the road
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_curvature = 0
    right_curvature = 0
    offset_center = 0
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = binary_image.shape[0]
    max_y = h-1
  
    
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    
    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new curvature radius in meters
        left_curvature = ((1 + (2*left_fit_cr[0]*max_y*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curvature = ((1 + (2*right_fit_cr[0]*max_y*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Distance from center is image x midpoint
    # Assuming camera is mounted in the center of the car
    if right_fit is not None and left_fit is not None:
        car_position = binary_image.shape[1]/2
        l_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        r_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        offset_center = (car_position - lane_center_position) * xm_per_pix
    
    left_line.add_curvature(left_curvature)
    right_line.add_curvature(right_curvature)
    
    return left_curvature, right_curvature, offset_center
    

def fill_lane(original_image, binary_image, left_fit, right_fit, Minv):
    #Marks lanes and road between them in the image
    new_image = np.copy(original_image)
    
    if left_fit is None or right_fit is None:
        return original_image
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_image.shape
    vertical_points = np.linspace(0, h-1, num=h)# to cover same y-range as image
    left_fitx = left_fit[0]*vertical_points**2 + left_fit[1]*vertical_points + left_fit[2]
    right_fitx = right_fit[0]*vertical_points**2 + right_fit[1]*vertical_points + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, vertical_points]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, vertical_points])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0, 255, 0), thickness=35)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 255, 0), thickness=35)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    result = cv2.addWeighted(new_image, 1, newwarp, 0.5, 0)
    return result


def display_info(original_image, road_curvature, center_offset, color = (255,255,0)):
    #writes info about curvature and offset on image
    new_image = np.copy(original_image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Curvature: ' + '{:.2f}'.format(road_curvature) + ' m'
    
    cv2.putText(new_image, text, (200,50), font, 1.5, color, 2, cv2.LINE_AA)
    direction = ''
    
    if center_offset > 0:
        direction = 'right'
    elif center_offset < 0:
        direction = 'left'
    abs_center_dist = abs(center_offset)
    text = '{:.2f}'.format(abs_center_dist) + ' m ' + direction + ' of center'
    cv2.putText(new_image, text, (200,100), font, 1.5, color, 2, cv2.LINE_AA)
    return new_image

def process_frame(image):
    image_copy = np.copy(image)
    image_binary, Minv = process_image(image_copy)
    
    # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
    if not left_line.detected or not right_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds = sliding_window_polyfit(image_binary)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds =  polyfit_with_previous_frame(image_binary, left_line.best_fit, right_line.best_fit)

            
    left_line.add_fit(left_fit, left_lane_inds)
    right_line.add_fit(right_fit, right_lane_inds)
    
    # draw the current best fit if it exists
    if left_line.best_fit is not None and right_line.best_fit is not None:
        image_ouput1 = fill_lane(image_copy, image_binary, left_line.best_fit, right_line.best_fit, Minv)
        rad_l, rad_r, d_center = calculate_curvature_and_offset(image_binary, left_line.best_fit, right_line.best_fit, 
                                                               left_lane_inds, right_lane_inds)
        curvature_left = left_line.get_line_curvature()
        curvature_right = right_line.get_line_curvature()
        image_ouput = display_info(image_ouput1, (curvature_left+curvature_right)/2, d_center)
    else:
        image_ouput = image_copy

    return image_ouput
    

