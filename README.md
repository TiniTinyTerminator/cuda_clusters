# The Canny Edge Filter and Connected Component Labeling

Edge detection is a fundamental task in image processing and computer vision, crucial for identifying object boundaries within images. The Canny Edge Filter is a widely used edge detection algorithm due to its ability to detect edges with optimal accuracy. This guide will walk you through the steps involved in the Canny Edge Filter, from noise reduction to edge tracking by hysteresis.

Additionally, we will explore the algorithm for finding clusters or connected components in a binary image, focusing on the conceptual steps. Understanding these concepts is essential for applications in image segmentation, object detection, and pattern recognition.

## Steps of the Canny Edge Filter

### 1. Noise Reduction

Images are often contaminated with noise, which can lead to false edge detection. To reduce this noise, the Canny edge detector applies a Gaussian filter to the image.

#### Gaussian Filter

The Gaussian filter is a low-pass filter that smooths the image by reducing the intensity variations between neighboring pixels. The 2D Gaussian filter is represented by the following equation:

$$ G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}} $$

For instance, a $5 \times 5$ Gaussian kernel with $\sigma = 1$ can be represented as:

$$
\begin{bmatrix}
1 & 4 & 7 & 4 & 1 \\
4 & 16 & 26 & 16 & 4 \\
7 & 26 & 41 & 26 & 7 \\
4 & 16 & 26 & 16 & 4 \\
1 & 4 & 7 & 4 & 1
\end{bmatrix}
$$

This kernel is then normalized by dividing each element by the sum of all elements (273 in this case), resulting in:

$$
\begin{bmatrix}
0.004 & 0.015 & 0.026 & 0.015 & 0.004 \\
0.015 & 0.058 & 0.095 & 0.058 & 0.015 \\
0.026 & 0.095 & 0.150 & 0.095 & 0.026 \\
0.015 & 0.058 & 0.095 & 0.058 & 0.015 \\
0.004 & 0.015 & 0.026 & 0.015 & 0.004
\end{bmatrix}
$$

### 2. Gradient Calculation

After smoothing, the next step is to find the intensity gradient of the image. This is done using Sobel filters, which calculate the gradient in the x and y directions.

#### Sobel Filters

The Sobel operator uses two \(3 \times 3\) kernels which are convolved with the image to compute the gradients in the x and y directions.

$$
G_x =
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
$$
$$
G_y =
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$

For each pixel in the image, the gradients in the x (\(G_x\)) and y (\(G_y\)) directions are computed by convolving the image with these kernels.

#### Example Calculation of Gradients

Consider a small section of the image:

$$
\begin{bmatrix}
10 & 10 & 10 \\
10 & 50 & 10 \\
10 & 10 & 10
\end{bmatrix}
$$

Applying the Sobel \(G_x\) kernel:

$$
G_x =
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
\times
\begin{bmatrix}
10 & 10 & 10 \\
10 & 50 & 10 \\
10 & 10 & 10
\end{bmatrix}
= (-10 + 10 + 50 - 50 + 10 - 10) = 0
$$

Applying the Sobel \(G_y\) kernel:

$$
G_y =
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
\times
\begin{bmatrix}
10 & 10 & 10 \\
10 & 50 & 10 \\
10 & 10 & 10
\end{bmatrix}
= (-10 - 20 - 10 + 10 + 20 + 10) = 0
$$

The gradient magnitude \(G\) and the gradient direction \(\theta\) are computed as follows:

$$ G = \sqrt{G_x^2 + G_y^2} = \sqrt{0^2 + 0^2} = 0 $$
$$ \theta = \arctan\left(\frac{G_y}{G_x}\right) = \arctan\left(\frac{0}{0}\right) = 0 $$

### 3. Non-Maximum Suppression

Non-maximum suppression is applied to thin the edges. This step involves scanning the image to remove pixels that are not considered to be part of an edge. The gradient direction is used to determine whether a pixel is a local maximum along the direction of the gradient.

#### Example Non-Maximum Suppression

Consider the gradient magnitude matrix:

$$
\begin{bmatrix}
0 & 0 & 0 \\
0 & 100 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

Here, the central pixel (100) is compared with its neighbors in the gradient direction. If it is the maximum, it is kept; otherwise, it is suppressed.

### 4. Double Threshold

The edge pixels are classified into strong, weak, and non-relevant pixels using two thresholds: a high threshold \(T_h\) and a low threshold \(T_l\).

- Strong pixels: Gradient magnitude > \(T_h\)
- Weak pixels: \(T_l\) < Gradient magnitude < \(T_h\)
- Non-relevant pixels: Gradient magnitude < \(T_l\)

For example, if \(T_h = 75\) and \(T_l = 25\):

$$
\begin{bmatrix}
0 & 0 & 0 \\
0 & 100 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

The pixel with 100 is a strong pixel (since 100 > 75), and the others are non-relevant.

### 5. Edge Tracking by Hysteresis

Finally, edge tracking by hysteresis is used to determine the final edges. Strong pixels are immediately considered as edges. Weak pixels are considered edges only if they are connected to strong pixels. This helps in eliminating spurious weak edges.

#### Example Edge Tracking by Hysteresis

Using the previous classification:

$$
\begin{bmatrix}
0 & 0 & 0 \\
0 & 100 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

Only the central pixel (100) remains as an edge.

The Canny Edge Filter provides a comprehensive approach to edge detection, balancing sensitivity and noise reduction, making it highly effective for various image processing applications.

Sure, let's explain the process of finding clusters (connected components) in a binary image using the implemented algorithm, but without delving into the specifics of CUDA or the code. Instead, we'll focus on the conceptual steps involved in the algorithm.

## Steps to Find Clusters Using the Implemented Algorithm

### 1. Initialization

The goal is to label all connected components (clusters) in a binary image. A binary image contains two pixel values: foreground (typically represented as 255) and background (represented as 0).

1. **Label Assignment**: 
   - Each foreground pixel is initially assigned a unique label (its own index in the image).
   - Background pixels are assigned a label of -1 (indicating they are not part of any foreground component).

### 2. Label Propagation

The purpose of this step is to ensure that all pixels belonging to the same connected component (cluster) end up with the same label.

1. **Iterative Propagation**:
   - For each foreground pixel, examine its 8-connected neighbors (adjacent pixels horizontally, vertically, and diagonally).
   - Propagate the minimum label among the pixel and its neighbors to the pixel itself. This means if a pixel has a neighbor with a lower label, it adopts that lower label.

2. **Repetition**:
   - This propagation step is repeated multiple times (or until no further changes occur), ensuring that labels spread across all pixels in the same connected component.

### 3. Flattening Labels

After propagation, it's possible that labels within a connected component are not uniformly the same but form a hierarchy. Flattening ensures each component has a uniform label.

1. **Flatten Hierarchies**:
   - For each pixel, trace back its label to the root label by following the chain of labels until a pixel points to itself.
   - Update each pixel's label to this root label, ensuring all pixels in a component have the same label.

### 4. Identifying Boundary Pixels

Boundary pixels are important for defining the edges of clusters.

1. **Boundary Detection**:
   - For each foreground pixel, check its 4-connected neighbors (adjacent pixels horizontally and vertically).
   - If any neighbor is a background pixel or belongs to a different cluster, mark the current pixel as a boundary pixel.

### 5. Contour Extraction

Contours represent the outlines of each connected component.

1. **Group Boundary Pixels**:
   - Group boundary pixels by their labels, forming sets of boundary pixels for each connected component.

2. **Form Contours**:
   - Convert these sets into contours, which can be used for further analysis or visualization.

### Conceptual Summary

- **Initialization**: Assign unique labels to foreground pixels and a special label to background pixels.
- **Label Propagation**: Spread the minimum label across connected pixels iteratively to unify labels within the same component.
- **Flattening Labels**: Ensure all pixels in a connected component share the same label by collapsing label hierarchies.
- **Identifying Boundary Pixels**: Detect pixels that form the boundary of each connected component by checking their neighbors.
- **Contour Extraction**: Group boundary pixels by their labels to form the contours of each connected component.

By following these steps, the algorithm effectively labels all connected components in a binary image, identifies their boundaries, and extracts their contours for further processing or analysis. This method is robust and ensures that all pixels in the same cluster are correctly labelled and that the boundaries are accurately defined.

# Explanation of the CUDA Kernels for Image Processing Operations

This header file, `cuda_kernel.h`, contains CUDA kernels for various image processing operations such as converting a color image to grayscale, applying the Sobel operator, performing double threshold hysteresis, tracking edges in hysteresis, performing non-maximum suppression, labeling connected components, detecting boundaries, and path interpolation. . These operations are crucial for tasks like edge detection in images. CUDA (Compute Unified Device Architecture) allows for parallel processing on NVIDIA GPUs, making these operations highly efficient for large images.

## Imaging

### 1. `colorToBW` Kernel

Converts a color image to grayscale. Each pixel in the input color image (in RGB format) is processed independently by a thread. The kernel calculates the grayscale value using the standard luminance conversion formula, which is a weighted sum of the red, green, and blue components:
$$ \text{Grayscale} = 0.299 \times R + 0.587 \times G + 0.114 \times B $$
This grayscale value is then stored in the output image.

### 2. `SobelOperator` Kernel

Applies the Sobel operator to a grayscale image to calculate the gradient magnitude and direction at each pixel. The Sobel operator detects edges by computing the horizontal (Gx) and vertical (Gy) gradients using convolution with Sobel kernels. The gradient magnitude (G) is then calculated as:
$$ G = \sqrt{Gx^2 + Gy^2} $$
The gradient direction (\(\theta\)) is computed using:
$$ \theta = \arctan\left(\frac{Gy}{Gx}\right) $$
The results are stored in the gradient and direction arrays.

### 3. `DoubleThresholdHysteresis` Kernel

Applies double thresholding to the gradient magnitudes to classify pixels as strong edges, weak edges, or non-edges. Pixels with gradient magnitudes above the high threshold are considered strong edges, those below the low threshold are non-edges, and those in between are weak edges. This classification helps in the subsequent edge tracking step.

### 4. `HysteresisTrackEdges` Kernel

Tracks edges by hysteresis to finalize edge detection. Weak edges (classified in the previous step) are considered as edges only if they are connected to strong edges. This kernel checks the 8-connected neighbors of each weak edge pixel to see if any of them is a strong edge. If so, the weak edge is promoted to a strong edge; otherwise, it is discarded.

### 5. `NonMaxSuppression` Kernel

Performs non-maximum suppression to thin the edges. For each pixel, the kernel compares its gradient magnitude with the magnitudes of the two neighboring pixels in the direction of the gradient. If the pixel's magnitude is not greater than both neighbors, it is suppressed (set to zero). This step ensures that the edges are one pixel wide and accurately represent the boundaries.

### 6. `convert_color_to_bw` Function

Converts a color image (in RGB format) to a grayscale image using the `colorToBW` kernel. The function allocates memory on the GPU, copies the input image data to the device, launches the kernel, and then copies the result back to the host. This function encapsulates the entire process of color-to-grayscale conversion using CUDA.

### 7. `canny_edge_detection` Function

Performs Canny edge detection on a grayscale image using a sequence of CUDA kernels:

1. **Sobel Operator**: Computes the gradient magnitude and direction.
2. **Non-Maximum Suppression**: Thins the edges to one pixel width.
3. **Double Threshold Hysteresis**: Classifies pixels as strong edges, weak edges, or non-edges.
4. **Hysteresis Edge Tracking**: Finalizes edge detection by promoting weak edges connected to strong edges.

The function allocates memory on the GPU, copies the input image data, launches each kernel in sequence, and copies the final edge-detected image back to the host. This function encapsulates the entire Canny edge detection process using CUDA.

---

## Clustering

This part of the document will explain the individual parts for clustering of the pixels

### 1. `BoundaryPixel` Struct

The `BoundaryPixel` struct holds information about boundary pixels:

- `label`: The label of the connected component.
- `point`: The coordinates of the boundary pixel.

### 2. `DeviceHypot` Function

A device-compatible function to calculate the Euclidean distance (`hypot`). This function computes the hypotenuse of a right-angled triangle given the lengths of the other two sides, using the formula \(\sqrt{x^2 + y^2}\).

### 3. `InitLabeling` Kernel

Initializes labels for connected component labeling. Each pixel in the binary image is assigned a label based on whether it is part of the foreground (255) or background (0). Foreground pixels are labeled with their index, while background pixels are labeled with -1. This kernel ensures that each pixel is independently labeled by its own thread.

### 4. `LabelPropagation` Kernel

Propagates labels for connected component labeling by updating each pixel's label to the minimum label of its 8-connected neighbors (the eight surrounding pixels). This kernel iteratively updates the labels to ensure that all connected pixels share the same label, effectively grouping them into connected components.

### 5. `FlattenLabels` Kernel

Flattens labels by propagating the minimum label value until each label points to itself. This is done by repeatedly updating each pixel's label to the label of its label until the labels converge. This step ensures that all pixels in a connected component have the same label.

### 6. `CalculateSegmentLengthsKernel` Kernel

Calculates segment lengths of a path, which is an array of points. Each thread computes the Euclidean distance between consecutive points in the path, storing the result in a lengths array. The last segment, being a single point, is assigned a length of zero.

### 9. `IdentifyBoundaryPixels` Kernel

Identifies boundary pixels in a labeled image by checking if a pixel has neighbors with different labels or is on the edge of the image. Each thread checks the four-connected neighbors (left, right, above, below) of its assigned pixel. If any neighbor has a different label or is out of bounds, the pixel is marked as a boundary pixel and added to the boundary pixels list.

### 10. `compute_distances` Kernel

Computes distances between points and a reference point, marking visited points with a maximum distance. Each thread calculates the distance from its assigned point to the reference point using the `DeviceHypot` function. Visited points are assigned a maximum distance to exclude them from further processing.

### 11. `label_components` Function

Labels connected components in a binary image using CUDA by repeatedly invoking the `InitLabeling`, `LabelPropagation`, and `FlattenLabels` kernels. The function initializes labels, then iteratively propagates and flattens them to group pixels into connected components. Finally, it copies the labeled image back to the host.

### 12. `extract_contours` Function

Extracts contours from a labeled image using CUDA by identifying boundary pixels. The function allocates memory on the GPU, copies the labeled image, and launches the `IdentifyBoundaryPixels` kernel to find boundary pixels. It then copies the boundary pixels back to the host, groups them by labels, and creates contours. These contours represent the boundaries of the connected components in the image.
