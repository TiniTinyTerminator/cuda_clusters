# Overview

This application processes an input image or a video stream from a camera using CUDA-accelerated image processing techniques. It includes functionalities such as converting color images to grayscale, applying Gaussian filters, performing Canny edge detection, labeling connected components, and extracting contours. Processed images are saved to a specified output directory.

## The Canny Edge Filter and Connected Component Labeling

Edge detection is a fundamental task in image processing and computer vision, crucial for identifying object boundaries within images. The Canny Edge Filter is a widely used edge detection algorithm due to its ability to detect edges with optimal accuracy. This application implements the Canny Edge Filter, guiding you through the steps involved, from noise reduction to edge tracking by hysteresis. Additionally, it uses the nearest neighbor algorithm to find clusters or connected components in a binary image, enabling the identification and labeling of distinct regions within the image.

## Prerequesties

1. cmake
2. cuda
3. clang
4. opencv (used for image/camera loading)
5. argparse

### loading git submodules

2 extra git modules are loaded in which are used for arg parsing and error handling. To load these in, do the following:

```sh
git submodule update --init --recursive
```

## How to Use the Image Processing Application

### Command-Line Arguments

The application uses `argparse` to handle command-line arguments:

- `-i`, `--image`: Path to the input image.
- `-o`, `--output`: Directory to save the processed images. Defaults to the current directory.
- `--min-cluster-size`: Minimum size of the cluster (contour) to be processed. Defaults to 10.

### Steps to Use the Application

#### Processing an Input Image

1. **Run the application** with the path to an image:

    ```sh
    ./image_processor -i path/to/image.png -o output_directory --min-cluster-size 10
    ```

2. **Output Files**:
    - `bw_image.png`: Grayscale version of the input image.
    - `blurred_image.png`: Gaussian blurred image.
    - `canny_image.png`: Image after Canny edge detection.
    - `filtered_image.png`: Image with small clusters removed and remaining clusters colored.

#### Processing a Video Stream from a Camera

1. **Run the application** without specifying an image path:

    ```sh
    ./image_processor -o output_directory --min-cluster-size 10
    ```

2. **Real-Time Display**:
    - `color`: Original camera feed.
    - `bw`: Grayscale version of the camera feed.
    - `blurred image`: Gaussian blurred camera feed.
    - `canny image`: Camera feed after Canny edge detection.
    - `removed small clusters and color clusters`: Processed feed with small clusters removed and remaining clusters colored.

3. **Exit the application** by pressing the 'c' key.

### Example Usage

To process an image:

```sh
./image_processor -i sample_image.jpg -o ./results --min-cluster-size 15
```

### To process video from a camera

```sh
./image_processor --min-cluster-size 15
```

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

The Sobel operator uses two $3 \times 3$ kernels which are convolved with the image to compute the gradients in the x and y directions.

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

For each pixel in the image, the gradients in the x ($G_{x}$) and y ($G_{y}$) directions are computed by convolving the image with these kernels.

#### Example Calculation of Gradients

Consider a small section of the image:

$$
\begin{bmatrix}
10 & 10 & 10 \\
10 & 50 & 10 \\
10 & 10 & 10
\end{bmatrix}
$$

Applying the Sobel $G_x$ kernel:

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

Applying the Sobel $G_y$ kernel:

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

The gradient magnitude $G$ and the gradient direction $\theta$ are computed as follows:

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

The edge pixels are classified into strong, weak, and non-relevant pixels using two thresholds: a high threshold $T_h$ and a low threshold $T_l$.

- Strong pixels: Gradient magnitude > $T_h$
- Weak pixels: $T_l$ < Gradient magnitude < $T_h$
- Non-relevant pixels: Gradient magnitude < $T_l$

For example, if $T_h = 75$ and $T_l = 25$:

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

## Explanation of the implemented functions for Image Processing

This header file, `cuda_kernel.h`, contains CUDA kernels for various image processing operations such as converting a color image to grayscale, applying the Sobel operator, performing double threshold hysteresis, tracking edges in hysteresis, performing non-maximum suppression, labeling connected components, detecting boundaries, and path interpolation.

### `map_and_color_points`

A function that maps points to an image using CUDA. It accepts a vector of points, a color in BGR order, and the size of the output image. The function allocates device memory for points and the color image, copies points from host to device, initializes the color image with zeros, converts the color to a `color_t` structure, and launches the CUDA kernel. After synchronizing the device and checking for errors, it creates the output image, copies the color image from device to host, frees device memory, and returns the output image.

### `apply_gaussian_filter`

This function applies a Gaussian filter to a grayscale image using CUDA. It creates a Gaussian kernel, allocates device memory, copies data to the GPU, and launches the `GaussianFilter` kernel. The function then copies the result back to the host and frees GPU memory.

### `convert_color_to_bw`

This function converts a color image to grayscale using CUDA. It allocates device memory, copies the input image to the GPU, and launches the `colorToBW` kernel. The function synchronizes the device, checks for errors, copies the result back to the host, and frees GPU memory.

### `canny_edge_detection`

This function performs Canny edge detection on a grayscale image using CUDA. It applies the `SobelOperator` to calculate gradient magnitude and direction, performs non-maximum suppression, applies double thresholding, and tracks edges by hysteresis. The function synchronizes the device, checks for errors, copies the result back to the host, and frees GPU memory.

### `label_components`

This function labels connected components in a binary image using CUDA. It allocates device memory, copies the input image to the GPU, and initializes labels with `InitLabeling`. The function iteratively propagates and flattens labels using `LabelPropagation` and `FlattenLabels` kernels. Finally, it copies the labeled image back to the host and frees GPU memory.

### `extract_contours`

This function extracts contours from a labeled image using CUDA. It allocates device memory, copies the labels to the GPU, and initializes the boundary count to zero. The function identifies boundary pixels with `IdentifyBoundaryPixels` kernel, copies the boundary pixels back to the host, and groups them by labels to create contours. The resulting contours are returned as a vector of points.

## Cuda Kernels

### `GaussianFilter`

A CUDA kernel that applies a Gaussian filter to an image. It takes input image data, output image data, image dimensions, a Gaussian kernel, and the kernel size. The kernel calculates the weighted sum of the neighboring pixels to apply the Gaussian blur, ensuring that pixel values are within image bounds.

### `colorToBW`

A CUDA kernel that converts a color image (in RGB format) to grayscale. It processes each pixel using the standard luminance conversion formula to compute the grayscale value. The kernel checks pixel coordinates to ensure they are within the image bounds.

### `SobelOperator`

A CUDA kernel that applies the Sobel operator to an image to detect edges. It calculates horizontal and vertical gradients for each pixel using the Sobel operator, computes the gradient magnitude and direction, and stores these values in output arrays. The kernel excludes pixels at the image borders.

### `DoubleThresholdHysteresis`

A CUDA kernel that performs double thresholding for edge detection. It classifies pixels into strong edges, weak edges, or non-edges based on high and low threshold values. The kernel processes each pixel and assigns edge classification accordingly.

### `HysteresisTrackEdges`

A CUDA kernel that tracks edges in the hysteresis phase of edge detection. It checks the neighbors of weak edge pixels to determine if they should be classified as strong edges. The kernel ensures it processes pixels within image bounds, excluding border pixels.

### `NonMaxSuppression`

A CUDA kernel that performs non-maximum suppression for edge detection. It compares the gradient magnitude of a pixel with its neighbors based on the gradient direction to suppress non-maximum values, keeping only the local maxima as edges. The kernel processes each pixel within the image bounds.

### `DeviceHypot`

A device-compatible function to calculate the hypotenuse. It computes the Euclidean distance given x and y coordinates.

### `InitLabeling`

A CUDA kernel that initializes labels for connected component labeling. It assigns unique labels to foreground pixels (value 255) and sets the background pixels to -1. The function processes each pixel to initialize labels based on their values.

### `LabelPropagation`

A CUDA kernel for label propagation in connected component labeling. It propagates the minimum label among the 8-connected neighbors for each foreground pixel. This iterative process helps in labeling connected components.

### `FlattenLabels`

A CUDA kernel that flattens labels in connected component labeling. It propagates the minimum label value to achieve unique labels for each connected component. The function ensures that each pixel's label is updated to the minimum label of its component.

### `CalculateSegmentLengths`

A CUDA kernel that calculates the segment lengths of a path. It computes the Euclidean distance between consecutive points in a path, storing the lengths in an output array. The last segment has no length.

### `IdentifyBoundaryPixels`

A CUDA kernel that identifies boundary pixels in a labeled image. It checks each pixel's 4-connected neighbors to determine if it is a boundary pixel. Boundary pixels are added to a list, and the boundary count is incremented atomically.

### `MapPoints`

A CUDA kernel function designed to map points to an image with a specified color. It takes an array of points, an output image, the image's width and height, the number of points, and the color to be used. The function computes the index for each thread, checks if the index is out of bounds, and maps points to the image if they are within image bounds, coloring them accordingly.
