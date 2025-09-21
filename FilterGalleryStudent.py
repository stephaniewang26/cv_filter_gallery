"""
Unit 1 Project: Computer Vision Filter Gallery
Requirements:
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
"""

import cv2
import numpy as np
from tkinter import Tk

# ----------------------------
# Filter Functions (TODOs)
# ----------------------------

def filter1(img, red_add, blue_subtract):
    warm_img = img.copy()
    blue_channel, green_channel, red_channel  = cv2.split(warm_img)
    red_channel = cv2.add(red_channel, red_add)
    blue_channel = cv2.subtract(blue_channel, blue_subtract)
    warm_img = cv2.merge((blue_channel, green_channel, red_channel))
    return warm_img
    #Return the resulting filtered image 
   
def filter2(img, x_k, y_k):
    sobel_img = img.copy()
    gray_img = cv2.cvtColor(sobel_img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=x_k)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=y_k)

    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    sobel_img = cv2.convertScaleAbs(gradient_magnitude)
    return sobel_img
    #Return the resulting filtered image 

def filter3(img, smoothing_strength, block_size):
    cartoon_img = img.copy()
    gray_img = cv2.cvtColor(cartoon_img, cv2.COLOR_BGR2GRAY)

    #reduce noise for thresholding
    blurred_img = cv2.medianBlur(gray_img, 7)
    #adaptive thresholding --> determines threshold for pixel based on region around it, not just one global value for threshold
    edges = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=block_size, C=5)
    
    #bilateral blurs while preserving edges
    color_img = cv2.bilateralFilter(cartoon_img, d=9, sigmaColor=smoothing_strength, sigmaSpace=smoothing_strength)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon_img = cv2.bitwise_and(color_img, edges_colored)
    return cartoon_img
    #Return the resulting filtered image 

def filter4(img, alpha, gamma):
    rdj_img = img.copy()

    rdj_overlay = cv2.imread('rdj_overlay.jpg')
    rdj_overlay = cv2.resize(rdj_overlay, (rdj_img.shape[1], rdj_img.shape[0]))
    alpha_val = alpha / 100
    beta_val = 1 - alpha_val
    rdj_img = cv2.addWeighted(rdj_img, alpha_val, rdj_overlay, beta_val, gamma)
    return rdj_img
    #Return the resulting filtered image 

def create_gallery(filter1, filter2, filter3, filter4):
    """
    Stack all images into a gallery layout using np.hstack/np.vstack.
    Add labels to each image with cv2.putText.
    Return the gallery image.
"""
    pass

if __name__ == "__main__":
    opossum_img = cv2.imread('opossum.png')
    #🆘🆘🆘🆘🆘are these of sufficient complexity?
    cv2.imshow('Original', opossum_img)
    # cv2.imshow('Filter 1', filter1(opossum_img, 0, 100))
    #cv2.imshow('Filter 2', filter2(opossum_img, 3, 3))
    #cv2.imshow('Filter 3', filter3(opossum_img, smoothing_strength=150, block_size=21))
    cv2.imshow('Filter 4', filter4(opossum_img, alpha=80, gamma=10))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------------
# Main Program
# ----------------------------

# Use Tkinter to select user image
# Read image / display error
# Resize for easier viewing if needed

# Create a window with trackbars
# Add two+ trackbars for each filter using cv2.createTrackbar
# Display Keyboard Shortcuts if using any

#while True:
    # Read current trackbar values
    # Apply filters
    # Show gallery
    # Check for save/quit
   # pass