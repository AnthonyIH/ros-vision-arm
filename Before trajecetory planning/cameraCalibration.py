import numpy as np
import cv2 as cv
import glob

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (10, 7)  # Ensure this matches the actual number of internal corners
frameSize = (640, 480)

# Termination criteria for refining detected corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points 
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 25  # Set the real-world size of chessboard squares (millimeters)
objp *= size_of_chessboard_squares_mm

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load all images in the directory
images = glob.glob('*.jpg')

if not images:
    print("No images found! Ensure images are in the correct directory.")
    exit()

found_chessboard = False  # Flag to track if any chessboard is found

for image in images:
    img = cv.imread(image)
    if img is None:
        print(f"Failed to load {image}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret:
        print(f"Chessboard found in {image}")
        found_chessboard = True
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)  # Store the refined corners

        # Draw and display the detected corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('Detected Chessboard', img)
        cv.waitKey(500)

    else:
        print(f"No valid chessboard detected in {image}. Skipping.")

cv.destroyAllWindows()

# If no chessboard was found, print a message and exit safely
if not found_chessboard:
    print("No valid chessboard pattern detected in any images. Exiting.")
    exit()

############## CAMERA CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

print("Camera Calibrated:", ret)
print("\nCamera Matrix:\n", cameraMatrix)
print("\nDistortion Coefficients:\n", dist)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)

############## SAVE CALIBRATION PARAMETERS TO FILE #####################

np.savez('calibration.npz', 
         camMatrix=cameraMatrix, 
         distCoef=dist, 
         rVector=rvecs, 
         tVector=tvecs)

print("Calibration data saved to 'calibration.npz'.")

############## UNDISTORTION #####################################################

img = cv.imread('cali5.jpg')
if img is None:
    print("Error: cali5.jpg not found!")
    exit()

h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# Undistort the image
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# Crop the image if necessary
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# Crop and save the final image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)

############## REPROJECTION ERROR COMPUTATION #################################

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Total reprojection error:", mean_error / len(objpoints))
