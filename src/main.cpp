#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <string>
#include <algorithm>

#include <argparse/argparse.hpp>
#include <opencv2/opencv.hpp>

#include "cuda_kernel.h"

using namespace std::chrono_literals;

// Define a list of colors using cv::Scalar
std::vector<cv::Scalar> colors = {
    cv::Scalar(255, 0, 0),    // Blue
    cv::Scalar(0, 255, 0),    // Green
    cv::Scalar(0, 0, 255),    // Red
    cv::Scalar(255, 255, 0),  // Cyan
    cv::Scalar(255, 0, 255),  // Magenta
    cv::Scalar(0, 255, 255),  // Yellow
    cv::Scalar(255, 255, 255) // White
};

int main(int argc, char const *argv[])
{
    argparse::ArgumentParser program("image_processor");

    program.add_argument("-i", "--image")
        .help("Path to the input image")
        .default_value(std::string(""));

    program.add_argument("-o", "--output")
        .help("Directory to save the processed images")
        .default_value(std::string("."));

    program.add_argument("--min-cluster-size")
        .help("Minimum size of the cluster (contour) to be processed")
        .default_value(10)
        .scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return -1;
    }

    std::string image_path = program.get<std::string>("--image");
    std::string output_dir = program.get<std::string>("--output");
    int min_cluster_size = program.get<int>("--min-cluster-size");

    // Ensure output directory exists
    std::filesystem::create_directories(output_dir);

    cv::Mat image;

    if (!image_path.empty())
    {
        image = cv::imread(image_path);
        if (image.empty())
        {
            std::cerr << "Error: Could not open or find the image!\n";
            return -1;
        }

        cv::Mat1b bw_image = convert_color_to_bw(image);
        cv::Mat1b blurred_image = apply_gaussian_filter(bw_image, 5, 1);
        cv::Mat1b canny_image = canny_edge_detection(blurred_image, 10, 50);
        std::vector<int> labels = label_components(canny_image);
        auto contours = extract_contours(labels, canny_image.cols, canny_image.rows);
        
        cv::Mat3b output(canny_image.size());
        output.setTo(cv::Scalar(0,0,0));

        for (size_t i = 0; i < contours.size(); i++)
        {
            std::vector<IntPoint_t> &contour = contours[i];

            if (contour.size() < min_cluster_size)
                continue;

            output += scale_and_map_points(contour, colors[i % colors.size()], output.size());
        }

        cv::imwrite(output_dir + "/bw_image.png", bw_image);
        cv::imwrite(output_dir + "/blurred_image.png", blurred_image);
        cv::imwrite(output_dir + "/canny_image.png", canny_image);
        cv::imwrite(output_dir + "/filtered_image.png", output);
    }
    else
    {
        cv::VideoCapture cam(0);
        if (!cam.isOpened())
        {
            std::cerr << "Error: Could not open the camera!\n";
            return -1;
        }

        while (true)
        {
            cam.read(image);
            if (image.empty())
            {
                std::cerr << "Error: Could not read frame from camera!\n";
                return -1;
            }

            cv::imshow("color", image);

            cv::Mat1b bw_image = convert_color_to_bw(image);
            cv::Mat1b blurred_image = apply_gaussian_filter(bw_image, 5, 1);
            cv::Mat1b canny_image = canny_edge_detection(blurred_image, 10, 50);
            std::vector<int> labels = label_components(canny_image);
            auto contours = extract_contours(labels, canny_image.cols, canny_image.rows);

            cv::Mat3b output(canny_image.size());
            output.setTo(cv::Scalar(0,0,0));

            for (size_t i = 0; i < contours.size(); i++)
            {
                std::vector<IntPoint_t> &contour = contours[i];

                if (contour.size() < min_cluster_size)
                    continue;

                output += scale_and_map_points(contour, colors[i % colors.size()], output.size());
            }

            cv::imshow("bw", bw_image);
            cv::imshow("blurred image", blurred_image);
            cv::imshow("canny image", canny_image);
            cv::imshow("removed small and colored clusters", output);

            int key = cv::waitKey(1);
            if (key == 'c')
                break;
        }
    }

    return 0;
}
