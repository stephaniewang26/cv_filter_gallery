"""
Unit 1 Project: Computer Vision Filter Gallery
Requirements:
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import Tk
from tkinter import filedialog

# ----------------------------
# Filter Functions (TODOs)
# ----------------------------

def filter1(img, threshold1, threshold2):
    canny_img = img.copy()
    gray_img = cv2.cvtColor(canny_img, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray_img, threshold1, threshold2)
    canny_img = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
    return canny_img
    #Return the resulting filtered image 
   
def filter2(img, blockSize, ksize):
    harris_img = img.copy()
    gray_img = cv2.cvtColor(harris_img, cv2.COLOR_BGR2GRAY)
    #create corner map
    harris_corners = cv2.cornerHarris(gray_img, blockSize=blockSize, ksize=ksize, k=0.04)
    #dilate to mark corners better
    harris_corners = cv2.dilate(harris_corners, None)
    #mark corners in red on original image
    harris_img[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
    return harris_img
    #Return the resulting filtered image 

def filter3(img, max_corners, quality_level, min_distance, block_size):
    shi_tomasi_img = img.copy()
    gray_img = cv2.cvtColor(shi_tomasi_img, cv2.COLOR_BGR2GRAY)
    shi_tomasi_corners = cv2.goodFeaturesToTrack(gray_img, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance, blockSize=block_size)
    shi_tomasi_corners = np.int32(shi_tomasi_corners)
    for corner in shi_tomasi_corners:
        x, y = corner.ravel()
        cv2.circle(shi_tomasi_img, (x, y), 20, 255, -1)
    return shi_tomasi_img
    #Return the resulting filtered image 

def filter4(img, contrast_threshold, edge_threshold):
    sift_img = img.copy()
    gray_img = cv2.cvtColor(sift_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=1.6
    )
    keypoints = sift.detect(gray_img, None)
    sift_img = cv2.drawKeypoints(gray_img, keypoints, sift_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return sift_img
    #Return the resulting filtered image 

def filter5(img, threshold):
    fast_img = img.copy()
    gray_img = cv2.cvtColor(fast_img, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create(
        threshold=threshold,
        nonmaxSuppression=True,
        type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
    )
    keypoints = fast.detect(gray_img, None)
    fast_img = cv2.drawKeypoints(gray_img, keypoints, None, color=(255,0,0))
    return fast_img
    #Return the resulting filtered image

def filter6(img, scale_factor, fast_threshold):
    orb_img = img.copy()
    gray_img = cv2.cvtColor(orb_img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(
        nfeatures=500,
        scaleFactor=scale_factor,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=fast_threshold
    )
    keypoints = orb.detect(gray_img, None)
    orb_img = cv2.drawKeypoints(gray_img, keypoints, None, color=(0,255,0))
    return orb_img
    #Return the resulting filtered image

def create_gallery(og, f1, f2, f3, f4, f5, f6):
    """
    Stack all images into a gallery layout using np.hstack/np.vstack.
    Add labels to each image with cv2.putText.
    Return the gallery image.
"""
    #add labels
    labels = ["Canny", "Harris Corners", "Shi Tomasi", "SIFT", "FAST", "ORB"]
    images = [f1, f2, f3, f4, f5, f6]
    
    for i, img in enumerate(images):
        cv2.putText(img, labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    row1 = np.hstack((f1, f2, f3))
    row2 = np.hstack((f4, f5, f6))  
    
    gallery = np.vstack((row1, row2))
    return gallery



# ----------------------------
# Main Program
# ----------------------------

# Use Tkinter to select user image
# Read image / display error
# Resize for easier viewing if needed

# Create a window with trackbars
# Add two+ trackbars for each filter using cv2.createTrackbar
# Display Keyboard Shortcuts if using any

def nothing(x):
    print("Trackbar value:", x)

if __name__ == "__main__":
    #opossum_img = cv2.imread('opossum.png')
    Tk().withdraw() # keep root window from appearing

    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    file_path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    if len(file_path):
        img=cv2.imread(file_path)

        if img is None:
            print("Error: Could not read image.")
        else:
            cv2.namedWindow("Filter Gallery")

            cv2.createTrackbar("Canny Threshold 1", "Filter Gallery", 50, 255, nothing)
            cv2.createTrackbar("Canny Threshold 2", "Filter Gallery", 50, 255, nothing)

            cv2.createTrackbar("Harris Block Size", "Filter Gallery", 5, 20, nothing)
            cv2.createTrackbar("Harris K Size", "Filter Gallery", 3, 7, nothing)

            cv2.createTrackbar("Shi Tomasi Max Corners", "Filter Gallery", 100, 1000, nothing)
            cv2.createTrackbar("Shi Tomasi Quality Level", "Filter Gallery", 1, 100, nothing)
            cv2.createTrackbar("Shi Tomasi Min Distance", "Filter Gallery", 10, 255, nothing)
            cv2.createTrackbar("Shi Tomasi Block Size", "Filter Gallery", 50, 255, nothing)

            cv2.createTrackbar("SIFT Contrast Threshold", "Filter Gallery", 4, 100, nothing)
            cv2.createTrackbar("SIFT Edge Threshold", "Filter Gallery", 1, 30, nothing)

            cv2.createTrackbar("FAST Threshold", "Filter Gallery", 10, 100, nothing)

            cv2.createTrackbar("ORB Scale Factor", "Filter Gallery", 10, 15, nothing)
            cv2.createTrackbar("ORB FAST Threshold", "Filter Gallery", 10, 100, nothing)

            while True:
                #read trackbar values
                canny_threshold1 = cv2.getTrackbarPos("Canny Threshold 1", "Filter Gallery")
                canny_threshold2 = cv2.getTrackbarPos("Canny Threshold 2", "Filter Gallery")

                harris_block = cv2.getTrackbarPos("Harris Block Size", "Filter Gallery")
                harris_k = cv2.getTrackbarPos("Harris K Size", "Filter Gallery")
                cv2.setTrackbarMin('Harris Block Size', 'Filter Gallery', 1)
                cv2.setTrackbarMin('Harris K Size', 'Filter Gallery', 1)

                #k needs odd kernel sizes > 1
                if harris_k % 2 == 0: 
                    harris_k = max(1, harris_k-1)
                    cv2.setTrackbarPos('Harris K Size', 'Filter Gallery', harris_k)

                st_max_corners = cv2.getTrackbarPos("Shi Tomasi Max Corners", "Filter Gallery")
                st_quality_level = cv2.getTrackbarPos("Shi Tomasi Quality Level", "Filter Gallery") / 100
                st_min_distance = cv2.getTrackbarPos("Shi Tomasi Min Distance", "Filter Gallery")
                st_block_size = cv2.getTrackbarPos("Shi Tomasi Block Size", "Filter Gallery")
                cv2.setTrackbarMin('Shi Tomasi Block Size', 'Filter Gallery', 1)
                
                sift_contrast_threshold = cv2.getTrackbarPos("SIFT Contrast Threshold", "Filter Gallery") / 1000
                sift_edge_threshold = cv2.getTrackbarPos("SIFT Edge Threshold", "Filter Gallery")
                cv2.setTrackbarMin('SIFT Edge Threshold', 'Filter Gallery', 1)

                fast_threshold = cv2.getTrackbarPos("FAST Threshold", "Filter Gallery")
                cv2.setTrackbarMin('FAST Threshold', 'Filter Gallery', 1)

                orb_scale_factor = cv2.getTrackbarPos("ORB Scale Factor", "Filter Gallery") / 10
                orb_fast_threshold = cv2.getTrackbarPos("ORB FAST Threshold", "Filter Gallery")
                cv2.setTrackbarMin('ORB Scale Factor', 'Filter Gallery', 10)
                cv2.setTrackbarMin('ORB FAST Threshold', 'Filter Gallery', 1)


                f1 = filter1(img.copy(), canny_threshold1, canny_threshold2)
                f2 = filter2(img.copy(), harris_block, harris_k)
                f3 = filter3(img.copy(), st_max_corners, st_quality_level, st_min_distance, st_block_size)
                f4 = filter4(img.copy(), sift_contrast_threshold, sift_edge_threshold)
                f5 = filter5(img.copy(), fast_threshold)
                f6 = filter6(img.copy(), orb_scale_factor, orb_fast_threshold)

                gallery = create_gallery(img.copy(), f1, f2, f3, f4, f5, f6)
                
                cv2.imshow("Filter Gallery", gallery)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            cv2.destroyAllWindows()
            
    else:
        print("No file chosen!")