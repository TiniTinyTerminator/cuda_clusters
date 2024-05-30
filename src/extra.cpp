#include "cuda_kernel.h"

void scale_and_map_points(const std::vector<IntPoint_t>& points, cv::Mat& image) {
    for (const auto& point : points) {
        cv::circle(image, cv::Point(point.x, point.y), 0, cv::Scalar(255, 255, 255), -1);
    }
}