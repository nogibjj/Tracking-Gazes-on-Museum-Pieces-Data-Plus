test_path = r"C:\Users\ericr\Downloads\cat video test for ref image finder.mp4"


# SIFT Version

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Remember to imread the image with cv2.IMREAD_GRAYSCALE

# img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE) # queryImage
base_img = cv2.imread("../IMG_6015_Base_SIFT.jpg", cv2.IMREAD_GRAYSCALE)  # queryImage
# img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage
base_far = cv2.imread("../IMG_6016_BASE_FAR.jpg", cv2.IMREAD_GRAYSCALE)  # trainImage

copy_base_img = cv2.imread("../IMG_6015_Base_SIFT.jpg")  # queryImage
copy_base_far = cv2.imread("../IMG_6016_BASE_FAR.jpg")  # trainImage
# base_img = cv2.imread("../IMG_6015_Base_SIFT.jpg", cv2.IMREAD_COLOR)  # queryImage
# base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

# base_far = cv2.imread("../IMG_6016_BASE_FAR.jpg", cv2.IMREAD_COLOR)  # trainImage
# base_far = cv2.cvtColor(base_far, cv2.COLOR_BGR2RGB)

# base_lateral = cv2.imread(
#     "../IMG_6021_Base_LATERAL.jpg", cv2.IMREAD_COLOR
# )  # trainImage
# base_lateral = cv2.cvtColor(base_lateral, cv2.COLOR_BGR2RGB)

# base_side = cv2.imread("../IMG_6018_Base_SIDE.jpg", cv2.IMREAD_COLOR)  # trainImage
# base_side = cv2.cvtColor(base_side, cv2.COLOR_BGR2RGB)

# base_shift = cv2.imread("../IMG_6024_Base_SHIFT.jpg", cv2.IMREAD_COLOR)  # trainImage
# base_shift = cv2.cvtColor(base_shift, cv2.COLOR_BGR2RGB)

# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(base_img, None)
kp2, des2 = sift.detectAndCompute(base_far, None)
# kp3, des3 = sift.detectAndCompute(base_lateral, None)
# kp4, des4 = sift.detectAndCompute(base_side, None)
# kp5, des5 = sift.detectAndCompute(base_shift, None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]
# matches2 = bf.knnMatch(des1, des3, k=2)
# matches3 = bf.knnMatch(des1, des4, k=2)
# matches4 = bf.knnMatch(des1, des5, k=2)
# Apply ratio test
good = []
good_pairs = []
for i, (m, n) in enumerate(matches):
    # if m.distance < 0.75 * n.distance:
    if m.distance < 0.2 * n.distance:
        good.append([m])
        matchesMask[i] = [1, 0]
        ## Notice: How to get the index
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        good_pairs.append([pt1, pt2])

        if i % 5 == 0:
            ## Draw pairs in purple, to make sure the result is ok
            cv2.circle(base_img, (int(pt1[0]), int(pt1[1])), 10, (255, 0, 255), -1)
            cv2.circle(base_far, (int(pt2[0]), int(pt2[1])), 10, (255, 0, 255), -1)

# Draw match in blue, error in red
draw_params = dict(
    matchColor=(255, 0, 0),
    singlePointColor=(0, 0, 255),
    matchesMask=matchesMask,
    flags=0,
)


# cv.drawMatchesKnn expects list of lists as matches.


img3 = cv2.drawMatchesKnn(
    base_img,
    kp1,
    base_far,
    kp2,
    good,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
res = cv2.drawMatchesKnn(base_img, kp1, base_far, kp2, matches, None, **draw_params)
plt.figure(figsize=(20, 20))
plt.imshow(img3), plt.show()
# plt.imshow(res), plt.show()


# abstracting important pieces to other functions

# create the sift object and its keypoints
# and descriptors by abstracting the code into
# a function


def image_matcher(reference_frame, comparison_frame):
    """Find the keypoints and descriptors with SIFT.

    Original scripts considered reading the images in
    grayscale and with cv2.imread"""

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(reference_frame, None)
    kp2, des2 = sift.detectAndCompute(comparison_frame, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    return matches, kp1, kp2


# choosing the best pairs guarantees the best accuracy
# when mapping the gaze points from a random frame to
# our reference image finder's best reference image


def pair_generators(
    distance_multiplier=0.1, gaze_point=None, matches=None, kp1=None, kp2=None
):
    """Generate the list of keypoint pairs.

    These keypoints come from the SIFT algorithm output."""
    good_pairs = []
    for m, n in matches:
        if m.distance < distance_multiplier * n.distance:
            # getting the reference image and comparison image points
            # and the
            pt1 = kp1[m.queryIdx].pt
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = kp2[m.trainIdx].pt
            pt2 = (int(pt2[0]), int(pt2[1]))

            if pt1[0] == gaze_point[0] or pt2[0] == gaze_point[0]:
                continue

            elif pt1[1] == gaze_point[1] or pt2[1] == gaze_point[1]:
                continue

            good_pairs.append([pt1, pt2])

    return good_pairs


# loop function that finds you the ideal pair
def ideal_pair(
    dist_ranges=np.arange(0.05, 1.0, 0.05),
    gaze_point=None,
    matches=None,
    kp1=None,
    kp2=None,
):
    """Find the best pair for the gaze point.

    It will stop once it finds at least 2 pairs."""

    for value in dist_ranges:
        pairs_list = pair_generators(
            distance_multiplier=value,
            gaze_point=gaze_point,
            matches=matches,
            kp1=kp1,
            kp2=kp2,
        )

        if len(pairs_list) >= 2:
            return pairs_list[0:2]

        else:
            pass

    return None


def keypoints_finder(
    reference_frame=None,
    comparison_frame=None,
    gaze_point=None,
    dist_ranges=np.arange(0.05, 1.0, 0.05),
):
    """Find the best two pairs of query and train points
    for the gaze point.

    This algorithm is meant to facilitate the mapping of the
    comparison gaze point to the reference image's gaze point."""

    matches, kp1, kp2 = image_matcher(
        reference_frame=reference_frame, comparison_frame=comparison_frame
    )

    pairs_list = ideal_pair(
        dist_ranges=dist_ranges,
        gaze_point=gaze_point,
        matches=matches,
        kp1=kp1,
        kp2=kp2,
    )

    if pairs_list is None:
        return None

    else:
        return pairs_list
