import cv2
import numpy as np

def get_contour_precedence(contour, cols):
    #tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // 1) * 1) * cols + origin[0]

def

def box_extraction(img_for_box_extraction_path, cropped_dir_path):

    img = cv2.imread(cropped_dir_path + img_for_box_extraction_path,0)  # Read the image
    print(img.shape)
    (thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # Thresholding the image
    #img_bin = 255 - img_bin  # Invert the image
    #cv2.imwrite("Image_bin.jpg",img_bin)

    # Defining a kernel length
    kernel_length = np.array(img).shape[1] // 40

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    #cv2.imwrite("verticle_lines.jpg", verticle_lines_img)  # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    #cv2.imwrite("horizontal_lines.jpg", horizontal_lines_img)

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
    contours.sort(key=lambda x: get_contour_precedence(x, img.shape[1]))
    for i in range(len(contours)):
        img_final_bin = cv2.putText(img_final_bin, str(i), cv2.boundingRect(contours[i])[:2], cv2.FONT_HERSHEY_DUPLEX, 2, [1])

    #img_final_bin = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    img_final_bin = cv2.resize(img_final_bin, (x, y))
    cv2.imshow('result', img_final_bin)
    cv2.waitKey(0)


x = 1900
y = 1050
box_extraction(r"SKEWTEST_Page_1.png", r"F:\PycharmProjects\OCR\Workbench\\")


# Get box template, super-expand the lines, then re-template to fix potential box breaks?