import cv2
import numpy as np

def calibrate_camera(images, nx, ny):
    #Takes chessboard images taken with a camera and returns camera coeffs
    imgpoints, objpoints = process_calibration_images(images, nx, ny)
    img_size = (images[0].shape[1], images[0].shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)
    return ret, mtx, dist, rvecs, tvecs


def process_calibration_images(images, nx, ny):
    #Returns chessboard corners for each of input images
    imgpoints = []
    objpoints= []
        
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0: nx, 0: ny].T.reshape(-1, 2)

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        
        #If corners found
        if ret == True:
            #img = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            #plt.imshow(img)
            objpoints.append(objp)
            imgpoints.append(corners)
            
    return imgpoints, objpoints
    
def undistortImage(image, mtx, dist):
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
    return undistorted_image

def unwrap(image, source, destination):
    h,w = image.shape[:2]
    M = cv2.getPerspectiveTransform(source, destination)
    Minv = cv2.getPerspectiveTransform(destination, source)
    warped = cv2.warpPerspective(image, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv # Minv is inverse matrix of M
    