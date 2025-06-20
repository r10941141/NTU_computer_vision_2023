Computer Vision 2023 by 簡韶逸 at NTU EE


-- HW1 --
Part 1: Difference of Gaussian (DoG)
This module implements a Difference of Gaussian (DoG) detector that extracts keypoints based on local extrema in scale-space across multiple octaves. Detected keypoints represent regions of significant intensity change.

Part 2: Joint Bilateral Filter
This module implements a Joint Bilateral Filter that performs edge-preserving smoothing on grayscale images and produces a filtered output image.


-- HW2 --
Part 1: BoW Scene Recognition
This module implements a Bag-of-Words (BoW) based scene recognition pipeline using SIFT descriptors and a nearest neighbor classifier.

Part 2: CNN Image Classification
This module implements deep CNN training for image classification on CIFAR-10-like datasets using a custom CNN (MyNet) and a pretrained ResNet18 as backbone options. It includes logging, validation, and learning curve visualization.


-- HW3 --
Part 1: Homography Estimation
This script implements homography estimation to warp multiple source images onto a target canvas.

Part 2: Marker-Based Planar AR
This script performs marker-based planar augmented reality by detecting ArUco markers in each video frame, estimating homography with a reference image.

Part 3: Unwarp the QR Code
This script performs backward warping to unwarp two QR code images, enabling extraction of their links and comparison of the warped results for analysis of similarity or difference.

Part 4: Panorama
This script implements image stitching by detecting and matching ORB features between consecutive images, estimating homography with RANSAC, and warping to create a stitched panorama images.


-- HW4 --
Stereo Matching
This script implements stereo matching by computing a disparity map using cost volume construction with absolute difference, applying joint bilateral filtering, left-right consistency check, hole filling, and refinement with a weighted median filter to produce a smooth disparity result.

