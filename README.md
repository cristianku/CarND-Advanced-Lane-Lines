Udacity Advanced Lane lines finding
-----------------------------------

<http://www.udacity.com/drive>

 

\*\*\* Official Udacity Review :
<https://review.udacity.com/#!/reviews/843707/shared> \*\*\*\*

\*\*\* Youtube Video : <https://www.youtube.com/watch?v=syV9GxVa9c0>

 

The goal is to write a software pipeline to identify the lane boundaries in a
video.

 

The Project
-----------

 

The goals / steps of this project are the following:

-   Compute the camera calibration matrix and distortion coefficients given a
    set of chessboard images.

-   Apply a distortion correction to raw images.

-   Use color transforms, gradients, etc., to create a thresholded binary image.

-   Apply a perspective transform to rectify binary image ("birds-eye view").

-   Detect lane pixels and fit to find the lane boundary.

-   Determine the curvature of the lane and vehicle position with respect to
    center.

-   Warp the detected lane boundaries back onto the original image.

-   Output visual display of the lane boundaries and numerical estimation of
    lane curvature and vehicle position.

 

Dependencies
------------

**Anaconda environment:**

-   [CarND Term1 Starter
    Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

    The lab enviroment can be created with CarND Term1 Starter Kit. Click
    [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md)
    for the details.

 

Details About Files In This Directory
-------------------------------------

-   `camera_cal` **folder**

    images for camera calibration.

-   `test_images` **folder**

    Images for testing the pipeline on single frames from Udacity

-   `output_images` **folder**

    test_images processes by the pipeline

-   `write_up_images` **folder**

    images for write_up.md

-   wide_dist_pickle.p

    Pickle file used to store the Calibration data ( Objpoints and Imgpoint)

-   `test_images2`**folder**\` \`

    Test images extracted from **challenge_video.mp4**

-   `test_images3`**folder**

    Test images extracted from **project_video.mp4**

-   `project_video.mp4`

    video the pipeline should work well on

     

-   **project_video_output.mp4**

    `project_video.mp4` processed by the pipeline

 

-   `challenge_video.mp4`

    optional video to process

-   `harder_challenge_video.mp4`

    optional video to process

     

Installation
------------

1.  Clone the repository

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ sh
git clone https://github.com/cristianku/CarND-Advanced-Lane-Lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

Running the project
-------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ sh
cd CarND-Advanced-Lane-Lines
source activate carnd-term1
jupyter notebook lanefind.ipynb
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 
-
