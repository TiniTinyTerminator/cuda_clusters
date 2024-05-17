#include <iostream>
#include <thread>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "cuda_kernel.h"

using namespace std::chrono_literals;

int main(int argc, char const *argv[])
{
    std::cout << "hello world!!" << std::endl;

    cv::VideoCapture cam(0);
    cv::Mat image, bw_image, canny_image, small_cluster_filter;

    bw_image = cv::Mat(cam.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT), cam.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH), CV_8UC1);
    canny_image = cv::Mat(cam.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT), cam.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH), CV_8UC1);
    small_cluster_filter = cv::Mat(cam.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT), cam.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH), CV_8UC1);

    std::vector<int> labels(image.cols * image.rows);

    while (true)
    {

        cam.read(image);

        cv::imshow("color", image);

        cv::GaussianBlur(image, image, cv::Size(3, 3), 5);

        convert_color_to_bw(image.data, bw_image.data, image.cols, image.rows);
        canny_edge_detection(bw_image.data, canny_image.data, bw_image.cols, bw_image.rows, 10, 50);
        remove_small_clusters(canny_image.data, small_cluster_filter.data, canny_image.cols, canny_image.rows, 3);

        label_components(small_cluster_filter.data, labels.data(), small_cluster_filter.cols, small_cluster_filter.rows);

        auto contours = extract_contours(labels.data(), small_cluster_filter.cols, small_cluster_filter.rows);
        std::vector<Point> interpolated_path; 

        // Solve TSP for each contour and apply FFT
        for (const auto& contour : contours)
        {
            std::vector<Point> tsp_path = solve_tsp(contour);

            // Interpolate the path to a fixed number of points using CUDA
            size_t num_points = 128; // Adjust the number of points as needed
            std::vector<Point> interpolated_path;
            interpolate_path(tsp_path, interpolated_path, num_points);

            // Apply FFT to the interpolated path
            apply_fft(interpolated_path);
        }        

        cv::imshow("bw", bw_image);
        cv::imshow("canny image", canny_image);
        cv::imshow("removed small clusters", small_cluster_filter);

        int key = cv::waitKey(1);
        if (key == 'c')
            break;

        /* code */
    }

    return 0;
}
