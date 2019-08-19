import cv2
import numpy as np

# Sorts contours
def sort_contours(contours):

    # List of (x,y,w,h) coords based on contours
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # Sorts contours and boxes by y-coord
    contour_boxes = list(sorted(zip(contours, bounding_boxes), key=lambda b: b[1][1], reverse=True))

    # Sorts all boxes in each row
    sorted_boxes = []
    temp_list = [contour_boxes[0]]
    for pair in contour_boxes[1:]:
        if pair[1][1] - temp_list[-1][1][1] in range(-10,10):
            temp_list.append(pair)
            continue
        elif len(temp_list) < 2:    # Lone contours are unlikely to be part of needed data
            temp_list = [pair]
            continue
        sorted_boxes.extend(sorted(temp_list, key=lambda b: b[1][0]))
        temp_list = [pair]

    contours, boxes = [None] * len(sorted_boxes), [None] * len(sorted_boxes)
    for i, pair in enumerate(sorted_boxes):
        contours[i] = pair[0]
        boxes[i] = pair[1]

    # Final output is sorted left to right, bottom to top
    return contours, boxes

def rank_contour(image, contour, rank):
    # compute the center of the contour area
    if cv2.contourArea(contour) < MIN_THRESH:
        return
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the contour number on the image
    cv2.putText(image, "#{}".format(rank), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 2)

    return image

def content_check(box, image):
    # For debugging
    img_test = np.empty_like(image)
    img_test[:] = image
    cv2.rectangle(img_test, (box[0] + box[2] // 3, box[1] + box[3] // 3), (box[0] + 2 * box[2] // 3, box[1] + 2 * box[3] // 3), (255, 255, 255), 1)
    img_test = cv2.resize(img_test, (x, y))
    cv2.imshow('result', img_test)
    cv2.waitKey(0)

    total_white = cv2.countNonZero(image[box[1] + box[3] // 3:box[1] + 2 * box[3] // 3, box[0] + box[2] // 3:box[0] + 2 * box[2] // 3])
    print(total_white)
    return total_white

def content_to_csv(content):
    1

    # ...

def box_extraction(filename, dirpath):

    # Read and threshold the image
    img = cv2.imread(dirpath + filename, 0)
    (thresh, img_bin) = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # kernels for morphological operations
    kernel_length = np.array(img).shape[1] // 40

    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))

    # Pull and recombine vertical and horizontal lines
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
    verticle_lines = cv2.dilate(img_temp1, verticle_kernel, iterations=1)

    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=1)
    horizontal_lines = cv2.dilate(img_temp2, hori_kernel, iterations=1)

    img_skeleton = cv2.addWeighted(verticle_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_skeleton = cv2.erode(~img_skeleton, kernel, iterations=2)
    (thresh, img_skeleton) = cv2.threshold(img_skeleton, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Detect all boxes in modified image
    contours, hierarchy = cv2.findContours(img_skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort boxes from left to right, bottom to top
    contours, bounding = sort_contours(contours)

    # Draw sorted contours for debugging
    for i,c in enumerate(contours,1):
        rank_contour(img_skeleton, c, i)
    img_skeleton = cv2.resize(img_skeleton, (x, y))
    cv2.imshow('result', img_skeleton)
    cv2.waitKey(0)

    img_eroded = cv2.erode(img_bin, kernel)
    img_test0 = cv2.resize(img_eroded, (x, y))
    cv2.imshow('result', img_test0)
    cv2.waitKey(0)

    # Extract content from boxes as csv
    content = []
    for box in bounding[:-4]:  # Last 4 boxes of test page are unneeded
        print(box)
        if box[0] - FIRST_COL_X in range(-10, 10):
            content.append([])
            content[-1].append(content_check(box, img_eroded))
        elif not content:
            continue
        else:
            content[-1].append(content_check(box, img_eroded))
    reversed(content) # Doesn't work
    print(content)

    content_to_csv(content)


# Resolution
x = 1900
y = 1000

# Threshold for contour centers in draw_contour
MIN_THRESH = 5
# X-coord of first row
FIRST_COL_X = 11

filename = r"scan_Page_1.png"
dirpath = r"F:\PycharmProjects\NUTCR\Workbench\\" # make sure to double backslash your file path
box_extraction(filename, dirpath)

# Get box template, super-expand the lines, then re-template to fix potential box breaks?
# Show template, have user pick header/info sections, maybe even point out broken cells?
