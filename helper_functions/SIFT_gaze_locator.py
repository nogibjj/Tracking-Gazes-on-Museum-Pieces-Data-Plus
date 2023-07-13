# test_path = r"C:\Users\ericr\Downloads\cat video test for ref image finder.mp4"


# SIFT Version

import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Remember to imread the image with cv2.IMREAD_GRAYSCALE

# # img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE) # queryImage
# base_img = cv2.imread("../IMG_6015_Base_SIFT.jpg", cv2.IMREAD_GRAYSCALE)  # queryImage
# # img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage
# base_far = cv2.imread("../IMG_6016_BASE_FAR.jpg", cv2.IMREAD_GRAYSCALE)  # trainImage

# copy_base_img = cv2.imread("../IMG_6015_Base_SIFT.jpg")  # queryImage
# copy_base_far = cv2.imread("../IMG_6016_BASE_FAR.jpg")  # trainImage
# # base_img = cv2.imread("../IMG_6015_Base_SIFT.jpg", cv2.IMREAD_COLOR)  # queryImage
# # base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

# # base_far = cv2.imread("../IMG_6016_BASE_FAR.jpg", cv2.IMREAD_COLOR)  # trainImage
# # base_far = cv2.cvtColor(base_far, cv2.COLOR_BGR2RGB)

# # base_lateral = cv2.imread(
# #     "../IMG_6021_Base_LATERAL.jpg", cv2.IMREAD_COLOR
# # )  # trainImage
# # base_lateral = cv2.cvtColor(base_lateral, cv2.COLOR_BGR2RGB)

# # base_side = cv2.imread("../IMG_6018_Base_SIDE.jpg", cv2.IMREAD_COLOR)  # trainImage
# # base_side = cv2.cvtColor(base_side, cv2.COLOR_BGR2RGB)

# # base_shift = cv2.imread("../IMG_6024_Base_SHIFT.jpg", cv2.IMREAD_COLOR)  # trainImage
# # base_shift = cv2.cvtColor(base_shift, cv2.COLOR_BGR2RGB)

# # Initiate SIFT detector
# sift = cv2.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(base_img, None)
# kp2, des2 = sift.detectAndCompute(base_far, None)
# # kp3, des3 = sift.detectAndCompute(base_lateral, None)
# # kp4, des4 = sift.detectAndCompute(base_side, None)
# # kp5, des5 = sift.detectAndCompute(base_shift, None)
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
# matchesMask = [[0, 0] for i in range(len(matches))]
# # matches2 = bf.knnMatch(des1, des3, k=2)
# # matches3 = bf.knnMatch(des1, des4, k=2)
# # matches4 = bf.knnMatch(des1, des5, k=2)
# # Apply ratio test
# good = []
# good_pairs = []
# for i, (m, n) in enumerate(matches):
#     # if m.distance < 0.75 * n.distance:
#     if m.distance < 0.2 * n.distance:
#         good.append([m])
#         matchesMask[i] = [1, 0]
#         ## Notice: How to get the index
#         pt1 = kp1[m.queryIdx].pt
#         pt2 = kp2[m.trainIdx].pt
#         good_pairs.append([pt1, pt2])

#         if i % 5 == 0:
#             ## Draw pairs in purple, to make sure the result is ok
#             cv2.circle(base_img, (int(pt1[0]), int(pt1[1])), 10, (255, 0, 255), -1)
#             cv2.circle(base_far, (int(pt2[0]), int(pt2[1])), 10, (255, 0, 255), -1)

# # Draw match in blue, error in red
# draw_params = dict(
#     matchColor=(255, 0, 0),
#     singlePointColor=(0, 0, 255),
#     matchesMask=matchesMask,
#     flags=0,
# )


# # cv.drawMatchesKnn expects list of lists as matches.


# img3 = cv2.drawMatchesKnn(
#     base_img,
#     kp1,
#     base_far,
#     kp2,
#     good,
#     None,
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
# )
# res = cv2.drawMatchesKnn(base_img, kp1, base_far, kp2, matches, None, **draw_params)
# plt.figure(figsize=(20, 20))
# plt.imshow(img3), plt.show()
# # plt.imshow(res), plt.show()


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

            elif (pt1[0] == pt2[0]) and (pt1[1] == pt2[1]):
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


# Geometric solution of the problem


"""
From this point onwards, two pairs of points are needed to solve the problem.
The previous functions were designed to obtain two reference - comparison pairs of points.

The reference image is the image on which we want to map the gaze point. 
The comparison image is the image from which we want to map the gaze point.
The gaze point is a point that is frame dependent.

A coordinate in one frame may not be the same coordinate 
in the reference image, because spatial changes may have
taken place across time. 

For example, one comparison frame may be more zoomed in
when compared to the reference image, or it may be more
translated to the right or left of the field of vision.

For these reasons, we needed a scale invariant algorithm
like SIFT to find common points across both images to
then estimate the gaze point's location on the reference image
by using the comparison points and gaze point
from the comparison image.

"""


def slope_finder(comparison_point, gaze_point):
    """Find the slope of the line that connects two points.

    In this case, we want the slope of the line that connects
    the gaze point and the comparison point."""

    slope = (gaze_point[1] - comparison_point[1]) / (
        gaze_point[0] - comparison_point[0]
    )
    return slope


def intercept_finder(reference_point, comparison_slope):
    """Find the intercept of the line that connects two points.

    In this case, we want the intercept of the line that connects
    the future gaze point to be mapped unto the reference image
    and the reference point. We don't have the gaze point yet,
    but we can estimate the lines intercept by using the
    comparison slope and the reference point. Those two
    lines will preserve the same relationship (slope) with
    their gaze points."""

    intercept = reference_point[1] - (comparison_slope * reference_point[0])
    return intercept


def slope_intercept_finder(reference_point, comparison_point, gaze_point):
    """Find the slope and intercept of the line that connects two points.

    The slope is obtained from the comparison point and the gaze point.

    The intercept is obtained by using that slope and the reference point.

    """

    slope = slope_finder(comparison_point, gaze_point)
    intercept = intercept_finder(reference_point, slope)

    return slope, intercept


# make a tuple from the slope and intercept


def intersecting_point(slope_intercept_a, slope_intercept_b):
    """Find the point at which two lines intersect.

    The idea for solving this problem in a "coding friendly" way
    comes from here :

    https://www.cuemath.com/geometry/intersection-of-two-lines/

    Arguments :

    - slope_intercept_a :  a tuple containing the slope and intercept of the first reference point.

    - slope_intercept_b :  a tuple containing the slope and intercept of the first reference point.

    By obtaining the slopes and intercepts, we can effectively find
    the reference gaze point."""

    # First, identify the terms by using the standard form from the
    # equations of these lines. We start with the assumption that we are
    # converting the slope-intercept form of a line to standard form.
    # This is why the coefficient for y will be -1.
    # 0 = Ax + By + C

    a_1 = slope_intercept_a[0]
    b_1 = -1  # negative and constant because of conversion
    c_1 = slope_intercept_a[1]

    a_2 = slope_intercept_b[0]
    b_2 = -1  # negative and constant because of conversion
    c_2 = slope_intercept_b[1]

    # terms
    x0_t1 = b_1 * c_2 - b_2 * c_1
    x0_t2 = a_1 * b_2 - a_2 * b_1
    x0 = x0_t1 / x0_t2

    y0_t1 = c_1 * a_2 - c_2 * a_1
    y0_t2 = a_1 * b_2 - a_2 * b_1
    y0 = y0_t1 / y0_t2

    return x0, y0


def reference_gaze_point_mapper(
    reference_frame,
    comparison_frame,
    gaze_point,
    dist_ranges=np.arange(0.05, 1.0, 0.05),
):
    """Map the gaze point from the comparison image to the reference image.

    This function uses the slopes and intercepts of the reference
    and comparison points to map the comparison gaze point to
    a reference gaze point.
    """

    # find the paired reference and comparison points
    # the size of the list is 2
    best_pairs = keypoints_finder(
        reference_frame, comparison_frame, gaze_point, dist_ranges=dist_ranges
    )

    pair_1_ref_pt = best_pairs[0][0]
    pair_1_comparison_pt = best_pairs[0][1]

    pair_2_ref_pt = best_pairs[1][0]
    pair_2_comparison_pt = best_pairs[1][1]

    # find the slopes of the comparison lines that connect to
    # the gaze point

    slope_intercept_a = slope_intercept_finder(
        pair_1_ref_pt, pair_1_comparison_pt, gaze_point
    )

    slope_intercept_b = slope_intercept_finder(
        pair_2_ref_pt, pair_2_comparison_pt, gaze_point
    )

    # find the reference gaze point
    reference_gaze_point = intersecting_point(slope_intercept_a, slope_intercept_b)

    assert (
        reference_gaze_point[1]
        == slope_intercept_a[0] * reference_gaze_point[0] + slope_intercept_a[1]
    )

    assert (
        reference_gaze_point[1]
        == slope_intercept_b[0] * reference_gaze_point[0] + slope_intercept_b[1]
    )

    reference_gaze_point = (int(reference_gaze_point[0]), int(reference_gaze_point[1]))

    return reference_gaze_point
