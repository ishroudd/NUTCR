import cv2
import numpy as np
import argparse
import imutils

# Writing custom sort algorithms sucks
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    contour_boxes = list(sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    sorted_boxes = []
    temp_list = [contour_boxes[0]]
    for pair in contour_boxes[1:]:
        if pair[1][1] - temp_list[-1][1][1] in range(-6,6):
            temp_list.append(pair)
            continue
        elif len(temp_list) < 2:    # Lone contours are unlikely to be part of the spreadsheet
            temp_list = [pair]
            continue
        sorted_boxes.extend(sorted(temp_list, key=lambda b: b[1][0]))
        temp_list = [pair]
    # sorted_boxes.extend(blah) <- last contour is the entire image and is not included
    # for pair in sorted_boxes:
    #     print(pair[0])
    #     print(cv2.countNonZero(pair[0]))
    return ([pair[0] for pair in sorted_boxes], [pair[1] for pair in sorted_boxes])

def rank_contour(image, c, i):
    # compute the center of the contour area and draw a circle
    # representing the center
    if cv2.contourArea(c) < MIN_THRESH:
        return
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the countour number on the image
    cv2.putText(image, "#{}".format(i), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 2)

    # return the image with the contour number drawn on it
    return image

def content_check(contour):
    1

def extract_content(contours):
    orig = 221
    content = []
    for contour in contours:
        if abs(contour[0][0][0] - orig) <= 10: #the list of contours is weird, looks like [[[x y]] [[x y]]]
            content.append([])
            content[-1].append(content_check(contour))
        elif not content:
            continue
        content[-1].append(content_check(contour))

def content_to_csv(content):
    1


def box_extraction(img_for_box_extraction_path, cropped_dir_path):

    img = cv2.imread(cropped_dir_path + img_for_box_extraction_path,0)  # Read the image
    print(img[2:10,9:10]) #debug
    (thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # Thresholding the image

    # Defining a kernel length
    kernel_length = np.array(img).shape[1] // 40

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  # A kernel of (3 X 3) ones
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3) # Morphological operation to detect verticle lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3) # Morphological operation to detect horizontal lines from an image

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha

    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    img_final_temp = cv2.resize(img_final_bin, (x, y))
    cv2.imshow("img_final_bin.jpg", img_final_temp)
    cv2.waitKey(0)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approx = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx.append(cv2.approxPolyDP(contour, epsilon, True))
    cv2.drawContours(img, approx, -1, (0, 255, 0), 3)
    cv2.waitKey(0)
    # approx.sort(key=lambda contour: get_contour_precedence(contour, img.shape[1]))
    # for i in range(len(approx)):
    #     img_final_bin = cv2.putText(img_final_bin, str(i), cv2.boundingRect(approx[i])[:2], cv2.FONT_HERSHEY_DUPLEX, 2, [1])
    contours, bounding = sort_contours(contours, "bottom-to-top")
    # contours, bounding = sort_contours(contours, "left-to-right")
    for i,c in enumerate(contours,1):
        rank_contour(img_final_bin, c, i)


    img_final_bin = cv2.resize(img_final_bin, (x, y))
    cv2.imshow('result', img_final_bin)
    cv2.waitKey(0)

# Resolution
x = 1900
y = 1000

# Threshold for contour centers in draw_contour
MIN_THRESH = 5

filename = "scan_Page_1.png"
dirpath = r"F:\PycharmProjects\NUTCR\Workbench\\"
box_extraction(filename, dirpath) # make sure to double backslash your file path


# Get box template, super-expand the lines, then re-template to fix potential box breaks?
# Show template, have customer pick header/info sections, maybe even point out broken cells?

# Majorly erode everything
# check content of small area around center of each contour
