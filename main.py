import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

class Frame():
    # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    # Apply a distortion correction to raw images.
    # Use color transforms, gradients, etc., to create a thresholded binary image.
    # Apply a perspective transform to rectify binary image ("birds-eye view").
    # Detect lane pixels and fit to find the lane boundary.
    # Determine the curvature of the lane and vehicle position with respect to center.
    # Warp the detected lane boundaries back onto the original image.
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

    def __init__(self, image=None, prev_line=None):

        self.prev_line = prev_line

        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Format - array of three values such that
        # f(y = Ay^2 + By + C
        # is the polynomial that describes the lane line curvature and
        # A is left_fit_m[0], B is left_fit_m[1], and C is left_fit_m[2] - same for right_fit_m
        self.left_fit_m = None  # Polynomial values in meters for left lane
        self.right_fit_m = None  # Polynomial values in meters for right lane

        # Format - array of three values such that
        # f(y = Ay^2 + By + C
        # is the polynomial that describes the lane line curvature and
        # A is left_fit_m[0], B is left_fit_m[1], and C is left_fit_m[2] - same for right_fit_m
        self.left_fit = None  # Polynomial values in pixels for left lane
        self.right_fit = None  # Polynomial values in pixels for right lane

        # Format - self.left_fitx and self.right_fitx are x values corresponding to y values in self.ploty
        # Used in matplotlib "plot" function
        # plt.plot(left_fitx, ploty) - same for right_fit_x
        self.left_fitx = None  # Polynomial for the left lane
        self.right_fitx = None  # Polynomial for the right lane
        self.ploty = None  # List of y values corresponding to x values in self.left_fitx and self.right_fitx

        self.left_curverad = None  # Curvature value in meters for left lane
        self.left_curverad = None  # Curvature value in meters for right lane
        self.avg_curverad = None  # Average curvature value in meters, represents the curvature in the middle of the road

        self.dist_between_lanes = None  # Distance between lanes in pixels at the bottom of the image
        self.dist_from_center_m = None  # Distance from the center of the lane in meters

        # For distortion calculation
        self.objpoints = []
        self.imgpoints = []

        # IMAGE PIPELINE
        # As much for debugging as anything else, has the image at every step of the process
        self.calculate_distortion()  # Calculate distortion coefficients
        self.src = np.float32([[590, 450], [250, 690], [1075, 690], [690, 450]])
        if image is not None:
            self.original_img = image  # Original
            self.image_shape = self.original_img.shape  # Image shape
            self.undist_img = self.undistort(self.original_img)  # After distortion correction
            self.gray_img = cv2.cvtColor(self.undist_img, cv2.COLOR_RGB2GRAY)  # Gray image
            self.gray_image_shape = (self.gray_img.shape[1], self.gray_img.shape[0])  # Gray image shape reversed
            self.dst_offset = 280
            self.dst = np.float32([[self.dst_offset, 0], [self.dst_offset, self.gray_image_shape[1]],
                                   [self.gray_image_shape[0] - self.dst_offset, self.gray_image_shape[1]],
                                   [self.gray_image_shape[0] - self.dst_offset, 0]])
            self.lines_img = self.isolate_lines()  # After lines isolated
            self.warped_original, M = self.transform_image(self.undist_img)  # After warped to birds-eye
            self.warped_lines, M = self.transform_image(self.lines_img)  # After polynomial drawn on warped image
            self.detect_lanes()  # Detect the lanes and fill polynomials
            self.find_curvature()  # Populate curvature values
            self.final_img = self.warp_boundaries()  # Original image with lines drawn
        else:
            self.original_img = None  # Original
            self.undist_img = None  # After distortion correction
            self.gray_img = None  # Gray image
            self.lines_img = None  # After lines isolated
            self.warped_original = None  # After warped to birds-eye
            self.warped_lines = None  # After polynomial drawn on warped image
            self.lane_lines = None
            self.final_img = None  # Original image with lines drawn

            self.image_shape = None  # Image shape
            self.gray_image_shape = None  # Gray image shape (reversed)

            self.dst_offset = 280
            self.dst = None

    def new_image(self, img):
        """
        Performs all operations on a new image
        :param img: New image to be processed
        :return: Final image with lane lines and curvature drawn
        """
        self.original_img = img  # Original
        self.image_shape = self.original_img.shape  # Image shape
        self.undist_img = self.undistort(self.original_img)  # After distortion correction
        self.gray_img = cv2.cvtColor(self.undist_img, cv2.COLOR_RGB2GRAY)  # Gray image
        self.gray_image_shape = (self.gray_img.shape[1], self.gray_img.shape[0])  # Gray image shape reversed
        self.dst_offset = 280
        self.dst = np.float32([[self.dst_offset, 0], [self.dst_offset, self.gray_image_shape[1]],
                               [self.gray_image_shape[0] - self.dst_offset, self.gray_image_shape[1]],
                               [self.gray_image_shape[0] - self.dst_offset, 0]])
        self.lines_img = self.isolate_lines()  # After lines isolated
        self.warped_original, M = self.transform_image(self.undist_img)  # After warped to birds-eye
        self.warped_lines, M = self.transform_image(self.lines_img)  # After polynomial drawn on warped image
        self.detect_lanes()  # Detect the lanes and fill polynomials
        self.find_curvature()  # Populate curvature values
        self.final_img = self.warp_boundaries()  # Original image with lines drawn
        self.prev_line = self.prep_next_line()
        return self.final_img

    def calculate_distortion(self):
        """
        Calculate distortion coefficients for use distorting future images
        :return: Nothing, automatically stores values in class variables
        """
        nx = 9
        ny = 6

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

    def undistort(self, img):
        """
        Undistorts an image according to values found in the calculate_distortion function
        :param img: Image to be undistorted
        :return: An undistorted image
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints,
                                                           self.imgpoints,
                                                           self.image_shape[1::],
                                                           None, None)
        return cv2.undistort(self.original_img, mtx, dist, None, mtx)

    def abs_sobel_thresh(self, orient='x', thresh=(0, 255)):
        """
        Apply absolute value of a sobel gradient in the x or y direction
        :param orient: Orientation of the sobel function
        :param thresh: Threshold of sobel gradient
        :return: A binary image thresholded according to the sobel gradient in the x or y direction
        """
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(self.gray_img, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(self.gray_img, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary_output

    def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        """
        Returns a thresholded image based on the magnitude of the gradient, useful for capturing lane edges
        :param image: Image to be processed
        :param sobel_kernel: Size of sobel kernel, larger values mean smoother/larger area
        :param mag_thresh: Threshold for magnitude of the gradient
        :return: A binary immage with activated pixels for all within threshold
        """
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi / 2)):
        """
        Returns an image thresholded based on the direction of the gradient
        :param image: Image to be thresholded
        :param sobel_kernel: Size of sobel kernel - larger values mean more smoothing
        :param thresh: Angular threshold in radians
        :return: A binary image containgin all pixesl with gradient direction in threshold
        """
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def hls_threshold(self, image, thresh=(0, 255)):
        """
        Threshold based on the saturation channel to isolate lines
        :param image: The image to be thresholded
        :param thresh: The threshold for the image
        :return: Returns a binary image where pixels within threshold are 1 and all others are 0
        """
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
        return s_binary

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # Return the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def isolate_lines(self, ksize=3):
        """
        Isolate lines using several functions based on the sobel gradient and combine them
        :param ksize: Size of the sobel kernel
        :return: A binary image with only the lane lines represented as a 1 and everything else as a 0
        """
        imshape = self.undist_img.shape

        vertices = np.array([[(150, imshape[0]), (550, 450), (730, 450), (imshape[1] - 100, imshape[0])]],
                            dtype=np.int32)
        # Following functions currently unused but left in just in case they are useful later
        # gradx = self.abs_sobel_thresh(region_masked, orient='x', thresh=(10, 140))
        # grady = self.abs_sobel_thresh(region_masked, orient='y', thresh=(10, 140))
        mag_binary = self.mag_thresh(self.gray_img, sobel_kernel=ksize, mag_thresh=(50, 200))
        dir_binary = self.dir_threshold(self.gray_img, sobel_kernel=ksize, thresh=(0.7, 1.3))
        hls_binary = self.hls_threshold(self.undist_img, thresh=(100, 255))

        combined = np.zeros_like(dir_binary)
        combined[(((mag_binary == 1) & (dir_binary == 1))) | (hls_binary == 1)] = 1

        region_masked = self.region_of_interest(combined, vertices)

        return region_masked

    def transform_image(self, img, inverse=False):
        """
        Transform an image based on src and dst as defined in the class variables self.src and self.dst
        :param img: Image to be transformed
        :param inverse: Boolean, if true imagge is transformed from dst to src
        :return: A transformed image
        """
        if inverse:
            M = cv2.getPerspectiveTransform(self.dst, self.src)
            warped = cv2.warpPerspective(img, M, self.gray_image_shape)
        else:
            M = cv2.getPerspectiveTransform(self.src, self.dst)
            warped = cv2.warpPerspective(img, M, self.gray_image_shape)
        return warped, M

    def fit_poly(self, leftx, lefty, rightx, righty):
        """
        Fit a second order polynomial to a list of pixels that represent the lane lines
        :param leftx: X values of pixels in the left lane
        :param lefty: Y values of pixels in the left lane corrospnding to leftx
        :param rightx: X values of pixels in the right lane
        :param righty: Y values of pixels in the right lane corrospnding to leftx
        :return: Nothing, stores lane fit in both pixel space and meters as class variables
        """
        # Fit a second order polynomial to each
        self.left_fit_m = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        self.right_fit_m = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, self.image_shape[0] - 1, self.image_shape[0])

        # Calculate both polynomials using ploty, left_fit and right_fit
        self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]


    def detect_lanes(self):
        """
        Detect the lanes from a transformed binary image and collect pixel locations for use in the fit_poly function
        :return: Nothing, performs operations and calls fit_poly function
        """
        if self.prev_line is not None:
            left_fit, right_fit = self.prev_line
            margin = 100

            # Grab activated pixels
            nonzero = self.warped_lines.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                           left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                                 left_fit[1] * nonzeroy + left_fit[
                                                                                     2] + margin)))
            right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                            right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                                   right_fit[1] * nonzeroy +
                                                                                   right_fit[
                                                                                       2] + margin)))

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit new polynomials
            self.fit_poly(leftx, lefty, rightx, righty)

            ## Visualization ##
            # Left in for debugging
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((self.warped_lines, self.warped_lines, self.warped_lines)) * 255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx - margin, self.ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx + margin,
                                                                            self.ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx - margin, self.ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx + margin,
                                                                             self.ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        else:
            histogram = np.sum(self.warped_lines[self.warped_lines.shape[0] // 2:, :], axis=0)
            out_img = np.dstack((self.warped_lines, self.warped_lines, self.warped_lines)) * 255

            midpoint = np.int(histogram.shape[0] // 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # HYPERPARAMETERS
            # Choose the number of sliding windows
            nwindows = 9
            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            minpix = 50

            # Set height of windows - based on nwindows above and image shape
            window_height = np.int(self.warped_lines.shape[0] // nwindows)

            # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
            nonzero = self.warped_lines.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Current positions to be updated later for each window in nwindows
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = self.warped_lines.shape[0] - (window + 1) * window_height
                win_y_high = self.warped_lines.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Draw the windows on the visualization image - Left in for debugging
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)

                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                # Recenter next window on the mean position if enough pixels found
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the pixels from left and right lanes
            try:
                left_lane_inds = np.concatenate(left_lane_inds)
                right_lane_inds = np.concatenate(right_lane_inds)
            except ValueError:
                # Avoids an error if the above is not implemented fully
                # Left this in just in case something malfunctions and nothing is found
                pass

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a polynomial to the pixel positions
            self.fit_poly(leftx, lefty, rightx, righty)


    def find_curvature(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define y-value where we want radius of curvature as the bottom of the image
        y_eval = np.max(self.ploty)

        # Calculation of radius of curvature
        self.left_curverad = ((1 + (
                2 * self.left_fit_m[0] * y_eval * self.ym_per_pix + self.left_fit_m[1]) ** 2) ** 1.5) / np.absolute(
            2 * self.left_fit_m[0])
        self.right_curverad = ((1 + (
                2 * self.right_fit_m[0] * y_eval * self.ym_per_pix + self.right_fit_m[
            1]) ** 2) ** 1.5) / np.absolute(
            2 * self.right_fit_m[0])

        self.avg_curverad = (self.left_curverad + self.right_curverad) / 2

        left_xint = self.left_fit[0] * y_eval ** 2 + self.left_fit[1] * y_eval + self.left_fit[2]
        right_xint = self.right_fit[0] * y_eval ** 2 + self.right_fit[1] * y_eval + self.right_fit[2]
        self.dist_between_lanes = right_xint - left_xint
        position = (left_xint + right_xint) / 2
        dist_from_center = self.image_shape[1] / 2 - position
        self.dist_from_center_m = dist_from_center * self.xm_per_pix

    def warp_boundaries(self):
        """
        Warp the lane lines back onto the undistorted image, draw curvature values on the image
        :return: Image with lane lines and curvature drawn
        """
        warp_zero = np.zeros_like(self.warped_lines).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])

        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))


        # Warp the blank back to original image space using inverse perspective matrix
        newwarp, M = self.transform_image(color_warp, inverse=True)
        # Combine the result with the original image
        weighted = cv2.addWeighted(self.undist_img, 1, newwarp, 0.3, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        curve_string_l = "L: {:,.2f}".format(self.left_curverad)
        curve_string_r = "R: {:,.2f}".format(self.right_curverad)
        curve_string_a = "A: {:,.2f}".format(self.avg_curverad)
        curve_string_d = "Dist from center: {:,.4f}".format(self.dist_from_center_m)
        cv2.putText(weighted, curve_string_l, (30, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(weighted, curve_string_r, (30, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(weighted, curve_string_a, (30, 110), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(weighted, curve_string_d, (30, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return weighted

    def prep_next_line(self):
        """Bundle polynomial values and store for use in the next frame"""
        return self.left_fit, self.right_fit


if __name__ == '__main__':

    """#Perform operations on multiple images
    
    images = glob.glob('test_images/*.jpg')

    for num, image in enumerate(images):
        img = mpimg.imread(image)
        frame = Frame(img)
        outpath = "output_images/test" + str(num) + ".jpg"
        plt.imshow(frame.final_img)
        plt.savefig(outpath)
        plt.clf()"""


    """# Perform operations on one image and adjust output size
    
    img = mpimg.imread('test_images/test2.jpg')
    frame = Frame(img)
    plt.imshow(frame.final_img)

    plt.show()"""


    # Perform operations on a video
    frame = Frame()
    project_video_out = 'project_video_out.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(frame.new_image)
    white_clip.write_videofile(project_video_out, audio=False)
