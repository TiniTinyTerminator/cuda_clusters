#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <opencv2/opencv.hpp>
template <class T>
struct Point_t
{
    T x, y;

};

typedef Point_t<int32_t> IntPoint_t;
typedef Point_t<float> FloatPoint_t;

/**
 * @brief Converts a color image to a grayscale image using CUDA.
 * 
 * @param input The input color image (cv::Mat3b).
 * @return cv::Mat1b The output grayscale image.
 * 
 * @throws std::runtime_error If a CUDA error occurs.
 */
cv::Mat1b convert_color_to_bw(const cv::Mat3b &input);

/**
 * @brief Applies a Gaussian filter to an input image using CUDA.
 * 
 * @param input The input image.
 * @param kernel_size The size of the Gaussian kernel (must be odd).
 * @param sigma The standard deviation of the Gaussian distribution.
 * 
 * @return cv::Mat1b The filtered output image.
 * 
 * @throws std::runtime_error If a CUDA error occurs.
 */
cv::Mat1b apply_gaussian_filter(const cv::Mat1b &input, const int kernel_size, const float sigma);


/**
 * @brief Performs Canny edge detection using CUDA.
 * 
 * @param input The input grayscale image (cv::Mat1b).
 * @param low_thresh The low threshold value for edge detection.
 * @param high_thresh The high threshold value for edge detection.
 * 
 * @return cv::Mat1b The output edge-detected image.
 * 
 * @throws std::runtime_error If a CUDA error occurs.
 */
cv::Mat1b canny_edge_detection(const cv::Mat1b &input, unsigned char low_thresh, unsigned char high_thresh);

/**
 * @brief Label connected components in a binary image using CUDA.
 * 
 * @param input The input binary image (cv::Mat1b).
 * 
 * @return std::vector<int> The labeled image as a flat vector.
 * 
 * @throws std::runtime_error If a CUDA error occurs.
 */
std::vector<int> label_components(const cv::Mat1b &input);

/**
 * @brief Extract contours from a labeled image using CUDA.
 * 
 * @param labels The labeled image as a flat vector.
 * @param width Width of the image.
 * @param height Height of the image.
 * 
 * @return std::vector<std::vector<IntPoint_t>> The extracted contours.
 * 
 * @throws std::runtime_error If a CUDA error occurs.
 */
std::vector<std::vector<IntPoint_t>> extract_contours(const std::vector<int> &labels, size_t width, size_t height);

/**
 * @brief Scales points and maps them to an image with a specified color.
 * 
 * @param points A vector of points to be mapped.
 * @param color The color used for mapping points (in BGR order).
 * @param size The size of the output image.
 * 
 * @return cv::Mat3b The output image with mapped points.
 * 
 * @throws std::runtime_error If a CUDA error occurs.
 */
cv::Mat3b map_and_color_points(const std::vector<IntPoint_t> &points, cv::Scalar color, cv::Size size);