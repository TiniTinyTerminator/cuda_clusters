#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>

struct Point
{
    int32_t x, y;
};

void convert_color_to_bw(unsigned char * input, unsigned char * output, size_t width, size_t height);

void canny_edge_detection(unsigned char *input, unsigned char *output, size_t width, size_t height, unsigned char low_thresh, unsigned char high_thresh);

void remove_small_clusters(unsigned char *input, unsigned char *output, size_t width, size_t height, int min_size);

void label_components(unsigned char *input, int *output, size_t width, size_t height);

std::vector<std::vector<Point>> extract_contours(const int *labels, size_t width, size_t height);

void interpolate_path(const std::vector<Point>& path, std::vector<Point>& interpolated_path, size_t num_points);

std::vector<Point> solve_tsp(const std::vector<Point>& points);

void apply_fft(const std::vector<Point>& path);
