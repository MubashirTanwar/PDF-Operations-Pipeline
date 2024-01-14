import cv2 
import numpy as np
import os
import math
import sys
def get_string(img_path):
    
    img = cv2.imread(img_path)

    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        # bg_img = cv2.medianBlur(dilated_img, 15)
        bg_img = cv2.GaussianBlur(dilated_img, (5,5),0)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1) 

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # kernel_line = np.ones((5, 5), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_line)
    processed_img_path = 'processed.png'
    cv2.imwrite(processed_img_path, img)
    
    draw_contours(processed_img_path)

def draw_contours(img_path): 
    img = cv2.imread(img_path,0)
    original_img = img.copy() 
    
    img = cv2.medianBlur(img, 5)

    # Threshold the image
    # _, threshed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 1)

    # Remove horizontal table borders
    kernel = np.ones((4, 1), np.uintp) 
    opened = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)

    # Remove vertical table borders
    lines = cv2.HoughLinesP(opened, 1, np.pi/180, 200, minLineLength=120, maxLineGap=5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the angle of the line
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Ignore lines that are vertical or near-vertical
        if abs(angle) > 45:
            cv2.line(opened, (x1, y1), (x2, y2), (0, 0, 0), 4)        
    lines_removed=opened 

    #perform erosion for noise removal
    kernel = np.ones((3,2), np.uint8)
    erosion = cv2.erode(lines_removed, kernel)

    #dialation
    kernel = np.ones((2,48), np.uint8)
    dilated = cv2.dilate(erosion, kernel, iterations=1)

    #closing to fill the smalls gaps left by dialation
    kernel = np.ones((2,6), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    x1=1
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 1200:  # Set a minimum contour area
            _,_,w,h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            if 1.8 < aspect_ratio: 
                if h < img.shape[0] * 0.7 and w < img.shape[0] * 0.6:   # Avoid very large boxes that span most of the image height and width
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    
                    width = int(rect[1][0])
                    height = int(rect[1][1])

                    src_pts = box.astype("float32")
                    # Coordinate of the points in box points after the rectangle has been straightened
                    dst_pts = np.array([[0, height-1],
                                        [0, 0],
                                        [width-1, 0],
                                        [width-1, height-1]], dtype="float32")

                    # The perspective transformation matrix
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(original_img, M, (width, height))

                    if height > width:
                        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                    # Save the cropped image
                    cv2.imwrite(os.path.join(output_dir, f'line_{x1}.png'), warped)
                    x1+=1

if sys.argv != 4:
    print("usage: python getLines.py imagePath outputDirectory")
img_path = sys.argv[1]
output_dir =sys.argv[2]
get_string(img_path)
