from typing import Tuple, List

import cv2 as cv
import numpy as np
from skimage import measure

'''
Shadow removal code by Yalım Doğan

This software is implemented according to the methods presented in:

- Murali, Saritha, and V. K. Govindan. 
"Removal of shadows from a single image." 
the Proceedings of First International Conference on Futuristic Trends 
in Computer Science and Engineering. Vol. 4.

- Murali, Saritha, and V. K. Govindan. 
"Shadow detection and removal from a single image using LAB color space." 
Cybernetics and information technologies 13.1 (2013): 95-103.

'''


# TODO:
# Matplotlib rather than opencv for showing images
# Documentation

# Applies median filtering over given point
def median_filter(img, point, filter_size) -> List:
    # Obtain window indices
    indices = [[x, y]
               for x in range(point[1] - filter_size // 2, point[1] + filter_size // 2 + 1)
               for y in range(point[0] - filter_size // 2, point[0] + filter_size // 2 + 1)]

    indices = list(filter(lambda x: not (x[0] < 0 or x[1] < 0 or
                                         x[0] >= img.shape[0] or
                                         x[1] >= img.shape[1]), indices))

    pixel_values = [0, 0, 0]

    # Find the median of pixel values
    for channel in range(3):
        pixel_values[channel] = list(img[index[0], index[1], channel] for index in indices)
    pixel_values = list(np.median(pixel_values, axis=1))

    return pixel_values


# Applies median filtering on given contour pixels, the filter size is adjustable
def edge_median_filter(img, contours_list, filter_size=7) -> np.ndarray:
    temp_img = np.copy(img)

    for partition in contours_list:
        for point in partition:
            temp_img[point[0][1]][point[0][0]] = median_filter(img,
                                                               point[0],
                                                               filter_size)

    return cv.cvtColor(temp_img, cv.COLOR_HSV2BGR)


def process_regions(org_image,
                    lab_img,
                    mask,
                    lab_adjustment,
                    shadow_dilation_kernel_size,
                    shadow_dilation_iteration,
                    shadow_size_threshold,
                    verbose) -> np.ndarray:
    shadow_clear_img = np.copy(org_image)  # Used for constructing corrected image

    # We need connected components
    # Initialize the labels of the blobs in our binary image
    labels = measure.label(mask)

    non_shadow_kernel_size = (shadow_dilation_kernel_size, shadow_dilation_kernel_size)
    non_shadow_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, non_shadow_kernel_size)

    # Now, we will iterate over each label's pixels
    for label in np.unique(labels):
        if not label == 0:
            temp_filter = np.zeros(mask.shape, dtype="uint8")
            temp_filter[labels == label] = 255

            # Only consider blobs with size above threshold
            if cv.countNonZero(temp_filter) >= shadow_size_threshold:
                shadow_indices = np.where(temp_filter == 255)

                non_shadow_temp_filter = cv.dilate(temp_filter, non_shadow_kernel,
                                                   iterations=shadow_dilation_iteration)

                # Get the new set of indices and remove shadow indices from them
                non_shadow_temp_filter = cv.bitwise_xor(non_shadow_temp_filter, temp_filter)
                non_shadow_indices = np.where(non_shadow_temp_filter == 255)

                # Contours are used for extracting the edges of the current shadow region
                contours, hierarchy = cv.findContours(temp_filter, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                # Q: Rather than asking for RGB constants individually, why not adjust L only?
                # A: L component isn't enough to REVIVE the colors that were under the shadow.

                # Calculate average LAB and BGR values in current shadow region and non-shadow areas
                # Find the average of the non-shadow areas
                if lab_adjustment:

                    shadow_average_lab = np.mean(lab_img[shadow_indices[0], shadow_indices[1], :], axis=0)

                    # Get the average LAB from border areas
                    border_average_lab = np.mean(lab_img[non_shadow_indices[0], non_shadow_indices[1], :],
                                                 axis=0)

                    # Calculate ratios that are going to be used on clearing the current shadow region
                    # This is different for each region, therefore calculated each time
                    lab_ratio = border_average_lab / shadow_average_lab

                    shadow_clear_img = cv.cvtColor(shadow_clear_img, cv.COLOR_BGR2LAB)
                    shadow_clear_img[shadow_indices[0], shadow_indices[1]] = np.uint8(
                        shadow_clear_img[shadow_indices[0],
                                         shadow_indices[1]] * lab_ratio)
                    shadow_clear_img = cv.cvtColor(shadow_clear_img, cv.COLOR_LAB2BGR)

                else:
                    shadow_average_bgr = np.mean(org_image[shadow_indices[0], shadow_indices[1], :], axis=0)

                    # Get the average BGR from border areas
                    border_average_bgr = np.mean(org_image[non_shadow_indices[0], non_shadow_indices[1], :], axis=0)
                    bgr_ratio = border_average_bgr / shadow_average_bgr

                    # Adjust BGR
                    shadow_clear_img[shadow_indices[0], shadow_indices[1]] = np.uint8(
                        shadow_clear_img[shadow_indices[0],
                                         shadow_indices[1]] * bgr_ratio)

                # Then apply median filtering over edges to smooth them
                # At least on the images I tried, this doesn't work as intended.
                # It is possible that this is the result of using a high frequency image only

                # Image is converted to HSV before filtering, as BGR components of the image
                # is more interconnected, therefore filtering each channel independently wouldn't be correct
                shadow_clear_img = edge_median_filter(cv.cvtColor(shadow_clear_img, cv.COLOR_BGR2HSV),
                                                      contours)
                if verbose:
                    # For debugging, cut the current shadow region from the image
                    reverse_mask = cv.cvtColor(cv.bitwise_not(temp_filter), cv.COLOR_GRAY2BGR)
                    img_w_hole = org_image & reverse_mask

                    temp_filter = cv.cvtColor(temp_filter, cv.COLOR_GRAY2BGR)
                    cv.drawContours(temp_filter, contours, -1, (255, 0, 0), 3)

                    all_images = cv.hconcat([org_image, temp_filter, img_w_hole, shadow_clear_img])

                    cv.imshow(f"Shadow Region {label}", all_images)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

    return shadow_clear_img


def calculate_mask(lab_img,
                   means,
                   thresholds,
                   region_adjustment_kernel_size) -> np.ndarray:
    # If mean is below 256 (which is I think the max value for a channel)
    channel_max = 256

    # Apply threshold using only L
    if sum(means[1:]) <= channel_max:
        mask = cv.inRange(lab_img, (0, 0, 0), (thresholds[0], channel_max, channel_max))
    else:  # Else, also consider B channel
        mask = cv.inRange(lab_img, (0, 0, 0), (thresholds[0], channel_max, thresholds[2]))

    kernel_size = (region_adjustment_kernel_size, region_adjustment_kernel_size)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
    cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, mask)
    cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, mask)

    return mask


def remove_shadows(org_image,
                   lab_adjustment,
                   region_adjustment_kernel_size,
                   shadow_dilation_iteration,
                   shadow_dilation_kernel_size,
                   shadow_size_threshold,
                   verbose) -> Tuple[np.ndarray, np.ndarray]:
    lab_img = cv.cvtColor(org_image, cv.COLOR_BGR2LAB)

    # Calculate the mean values of A and B across all pixels
    means = [np.mean(lab_img[:, :, i]) for i in range(3)]
    thresholds = [means[i] - (np.std(lab_img[:, :, i]) / 3) for i in range(3)]

    mask = calculate_mask(lab_img, means, thresholds, region_adjustment_kernel_size)

    shadow_clear_img = process_regions(org_image,
                                       lab_img,
                                       mask,
                                       lab_adjustment,
                                       shadow_dilation_kernel_size,
                                       shadow_dilation_iteration,
                                       shadow_size_threshold,
                                       verbose)

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

    return shadow_clear_img, mask


def process_image_file(img_name,
                       save=False,
                       lab_adjustment=False,
                       region_adjustment_kernel_size=10,
                       shadow_dilation_kernel_size=5,
                       shadow_dilation_iteration=3,
                       shadow_size_threshold=2500,
                       verbose=False) -> Tuple[np.ndarray, np.ndarray]:
    org_image = cv.imread(img_name)
    print("Read the image {}".format(img_name))

    shadow_clear, mask = remove_shadows(org_image,
                                        lab_adjustment,
                                        region_adjustment_kernel_size,
                                        shadow_dilation_iteration,
                                        shadow_dilation_kernel_size,
                                        shadow_size_threshold,
                                        verbose=verbose)

    cv.imshow("Original - Shadow Mask - Cleared Shadows", cv.hconcat([org_image, mask, shadow_clear]))
    cv.waitKey(0)

    if save:
        f_name = img_name[:img_name.index(".")] + "_shadowClear" + img_name[img_name.index("."):]
        cv.imwrite(f_name, shadow_clear)
        print("Saved result as " + f_name)

    return org_image, shadow_clear
