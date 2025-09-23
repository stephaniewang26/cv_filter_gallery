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
    sobel_img = cv2.cvtColor(sobel_img, cv2.COLOR_GRAY2BGR)
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
    #weight of first image
    alpha_val = alpha / 100
    #weight of second image
    beta_val = 1 - alpha_val
    rdj_img = cv2.addWeighted(rdj_img, alpha_val, rdj_overlay, beta_val, gamma)
    return rdj_img
    #Return the resulting filtered image 

def create_gallery(og, f1, f2, f3, f4):
    """
    Stack all images into a gallery layout using np.hstack/np.vstack.
    Add labels to each image with cv2.putText.
    Return the gallery image.
"""
    #add labels
    labels = ["Original", "Warmth", "Sobel", "Cartoonify", "Rio de Janeiro"]
    images = [og, f1, f2, f3, f4]
    
    for i, img in enumerate(images):
        cv2.putText(img, labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    row1 = np.hstack((og, f1, f2))
    row2 = np.hstack((f3, f4, np.zeros_like(og)))  # blank space to match row1
    
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

            cv2.createTrackbar("Red Add", "Filter Gallery", 50, 255, nothing)
            cv2.createTrackbar("Blue Sub", "Filter Gallery", 50, 255, nothing)

            cv2.createTrackbar("Sobel ksize X", "Filter Gallery", 3, 7, nothing)
            cv2.createTrackbar("Sobel ksize Y", "Filter Gallery", 3, 7, nothing)

            cv2.createTrackbar("Smoothing strength", "Filter Gallery", 150, 255, nothing)
            cv2.createTrackbar("Outline thickness", "Filter Gallery", 21, 101, nothing)
            #🆘🆘idk what to put as the max i think it can just keep going

            cv2.createTrackbar("Alpha", "Filter Gallery", 80, 100, nothing)
            cv2.createTrackbar("Gamma", "Filter Gallery", 10, 100, nothing)

            while True:
                #read trackbar values
                red_add = cv2.getTrackbarPos("Red Add", "Filter Gallery")
                blue_sub = cv2.getTrackbarPos("Blue Sub", "Filter Gallery")

                x_k = cv2.getTrackbarPos("Sobel ksize X", "Filter Gallery")
                y_k = cv2.getTrackbarPos("Sobel ksize Y", "Filter Gallery")
                cv2.setTrackbarMin('Sobel ksize X', 'Filter Gallery', 1)
                cv2.setTrackbarMin('Sobel ksize Y', 'Filter Gallery', 1)

                #sobel needs odd kernel sizes > 1
                if x_k % 2 == 0: 
                    x_k = max(1, x_k-1)
                    cv2.setTrackbarPos('Sobel ksize X', 'Filter Gallery', x_k)
                if y_k % 2 == 0: 
                    y_k = max(1, y_k-1)
                    cv2.setTrackbarPos('Sobel ksize Y', 'Filter Gallery', y_k)

                smoothing_strength = cv2.getTrackbarPos("Smoothing strength", "Filter Gallery")
                block_size = cv2.getTrackbarPos("Outline thickness", "Filter Gallery")
                if block_size % 2 == 0: 
                    block_size = max(3, block_size-1)  # must be odd
                    cv2.setTrackbarPos('Outline thickness', 'Filter Gallery', block_size)
                
                alpha = cv2.getTrackbarPos("Alpha", "Filter Gallery")
                gamma = cv2.getTrackbarPos("Gamma", "Filter Gallery")

                f1 = filter1(img.copy(), red_add, blue_sub)
                f2 = filter2(img.copy(), x_k, y_k)
                f3 = filter3(img.copy(), smoothing_strength, block_size)
                f4 = filter4(img.copy(), alpha, gamma)

                gallery = create_gallery(img.copy(), f1, f2, f3, f4)
                cv2.imshow('Filter Gallery', gallery)

                key = cv2.waitKey(50) & 0xFF
                if key != 255:   
                    break

            cv2.destroyAllWindows()
            
    else:
        print("No file chosen!")


    #🆘🆘🆘🆘are these of sufficient complexity?
    #cv2.imshow('Original', opossum_img)
    # cv2.imshow('Filter 1', filter1(opossum_img, 0, 100))
    #cv2.imshow('Filter 2', filter2(opossum_img, 3, 3))
    #cv2.imshow('Filter 3', filter3(opossum_img, smoothing_strength=150, block_size=21))
    #cv2.imshow('Filter 4', filter4(opossum_img, alpha=80, gamma=10))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()