#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <set>
#include <string>
#include <vector>
#include <values.h>

using namespace cv;
using namespace cv::xfeatures2d;

template<class T>
T mymin(std::initializer_list<T> ilist) {  // min function for multiple arguments
    return *std::min_element(ilist.begin(), ilist.end());
}

struct Direction {
    int x, y;
    Direction(int a = 0, int b = 0): x(a), y(b) {}
};

void leaveBestPoints(std::vector<KeyPoint>& input) {
    if (input.size() > 100) {
        std::sort(input.rbegin(), input.rend(), [](KeyPoint a, KeyPoint b){return a.response * a.size < b.response * b.size;});
        input.resize(100);
    }
}

void measureBorders(Mat& first, Mat& second, int& min_disp, int& max_disp) {  // измеряет границы значений диспаритета
    Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.1, 5, 1.6);
    std::vector<KeyPoint> firstKeys, secKeys;
    f2d -> detect(first, firstKeys);
    f2d -> detect(second, secKeys);
    leaveBestPoints(firstKeys);
    leaveBestPoints(secKeys);
    
    Mat firstDescriptor, secDescriptor;
    f2d -> compute (first, firstKeys, firstDescriptor);
    f2d -> compute (second, secKeys, secDescriptor);
    
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(firstDescriptor, secDescriptor, matches);
    
    double min = 100, max = 0;
    for (size_t i = 0; i != matches.size(); ++i) {
        if (matches[i].distance < min) {
            min = matches[i].distance;
        }
        if (matches[i].distance > max) {
            max = matches[i].distance;
        }
    }
    
    max_disp = static_cast<int>(ceil(max));
    min_disp = static_cast<int>(ceil(min));
}

void countUnary(Mat& first, Mat& second, int max_disparity, std::vector<std::vector<std::vector<int>>>& unaries) {
    for (size_t y = 0; y != first.rows; ++y) {
        for (size_t x = 0; x != first.cols; ++x) {
            for (int d = 0; d <= 2 * max_disparity; ++d) {
                int currd = d - max_disparity;
                unaries[x][y][d] = (x + currd >= 0 && x + currd < first.cols ?
                                   abs(static_cast<int>(first.at<uchar>(y, x)) - static_cast<int>(second.at<uchar>(y, x + currd))) : 0);
            }
        }
    }
}

void countDirections(std::vector<std::vector<std::vector<int>>>& unaries, std::vector<std::vector<std::vector<int>>>& temp,
                     std::vector<std::vector<std::vector<int>>>& aggregated, Direction r, int max_disparity) {
    
    double penaltyEqual = 8;  // just an example of values, not tested
    double penaltyDiffer = 64;
    
    int alpha, beta, ksi, eps, c, k;
    
    if (r.x < 0) {
        alpha = unaries.size() - 1;
        beta = -1;
        c = -1;
    } else {
        alpha = 0;
        beta = unaries.size();
        c = 1;
    }
    if (r.y < 0) {
        ksi = unaries[0].size() - 1;
        eps = 0;
        k = -1;
    } else {
        ksi = 0;
        eps = unaries[0].size();
        k = 1;
    }
    
    for (int x = alpha; x != beta; x += c) {
        for (int y = ksi; y != eps; y += k) {
            int newX = x - r.x;
            int newY = y - r.y;
            if (newX < 0 || newY < 0 || newX >= unaries.size() || newY >= unaries[0].size()) {
                for (size_t d = 0; d <= 2 * max_disparity; ++d)
                    temp[x][y][d] = unaries[x][y][d];
            } else {
                int minimum = *std::min_element(temp[newX][newY].begin(), temp[newX][newY].end());
                
                for (size_t d = 0; d <= 2 * max_disparity; ++d) {
                    if (d == 2 * max_disparity) {
                        temp[x][y][d] = unaries[x][y][d] + static_cast<int>(mymin<double>({temp[newX][newY][d],
                                                                                           temp[newX][newY][d - 1] +
                                                                                           penaltyEqual,
                                                                                           minimum +
                                                                                           penaltyDiffer}));
                    } else if (d == 0) {
                        temp[x][y][d] = unaries[x][y][d] + static_cast<int>(mymin<double>({temp[newX][newY][d],
                                                                                           temp[newX][newY][d + 1] +
                                                                                           penaltyEqual,
                                                                                           minimum +
                                                                                           penaltyDiffer}));
                    } else {
                        temp[x][y][d] = unaries[x][y][d] + static_cast<int>(mymin<double>({temp[newX][newY][d],
                                                                                           temp[newX][newY][d - 1] +
                                                                                           penaltyEqual,
                                                                                           temp[newX][newY][d + 1] +
                                                                                           penaltyEqual,
                                                                                           minimum +
                                                                                           penaltyDiffer}));
                    }
                }
            }
        }
    }
    
    for (size_t x = 0; x != aggregated.size(); ++x) {
        for (size_t y = 0; y != aggregated[0].size(); ++y) {
            for (size_t d = 0; d != 2 * max_disparity; ++d) {
                aggregated[x][y][d] += temp[x][y][d];
            }
        }
    }
    
    for (size_t x = 0; x != unaries.size(); ++x) {
        for (size_t y = 0; y != unaries[0].size(); ++y) {
            std::fill(temp[x][y].begin(), temp[x][y].end(), 0);
        }
    }
}

int main() {  // there will be console flags in final version
    std::string firstSource, secSource;
//    std::cin >> firstSource >> secSource;
    firstSource = "/home/sanity-seeker/Programming/stereo_vision/dpth_map_dataset/aloe_left.jpg";
    secSource = "/home/sanity-seeker/Programming/stereo_vision/dpth_map_dataset/aloe_right.jpg";
    Mat first = imread(firstSource, CV_LOAD_IMAGE_UNCHANGED);
    Mat second = imread(secSource, CV_LOAD_IMAGE_UNCHANGED);
    
    int min_disp, max_disp;
    measureBorders(first, second, min_disp, max_disp);
    max_disp = std::min(3 * min_disp, max_disp); // approximately
    
    std::vector<std::vector<std::vector<int>>> ctable(first.cols, (std::vector<std::vector<int>> (first.rows, std::vector<int>(2 * max_disp, 0))));
    std::vector<std::vector<std::vector<int>>> stable(first.cols, (std::vector<std::vector<int>> (first.rows, std::vector<int>(2 * max_disp, 0))));
    std::vector<std::vector<std::vector<int>>> temtable(first.cols, (std::vector<std::vector<int>> (first.rows, std::vector<int>(2 * max_disp, 0))));
    std::vector<std::vector<int>> disparities(first.cols, (std::vector<int>(first.rows, 0)));
    // unary potential
    countUnary(first, second, max_disp, ctable);
    
    // binary potential
    Direction up(0, 1), down(0, -1), left(-1, 0), right(1, 0), right_down(1, -1), right_up(1, 1), left_down(-1, -1), left_up(-1, 1);
    countDirections(ctable, temtable, stable, up, max_disp);
    countDirections(ctable, temtable, stable, down, max_disp);
    countDirections(ctable, temtable, stable, left, max_disp);
    countDirections(ctable, temtable, stable, right, max_disp);
    countDirections(ctable, temtable, stable, right_down, max_disp);
    countDirections(ctable, temtable, stable, right_up, max_disp);
    countDirections(ctable, temtable, stable, left_down, max_disp);
    countDirections(ctable, temtable, stable, left_up, max_disp);
    
    // select right disparity and create a grayscale image
    int smax = MININT;
    int smin = MAXINT;
    for (size_t x = 0; x != first.cols; ++x) {
        for (size_t y = 0; y != first.rows; ++y) {
            int min_index = std::min_element(stable[x][y].begin(), stable[x][y].end()) - stable[x][y].begin();
            disparities[x][y] = min_index;
            if (min_index > smax) {
                smax = min_index;
            }
            if (min_index < smin) {
                smin = min_index;
            }
        }
    }
    
    Mat stereo(first.rows, first.cols, CV_8UC1, Scalar(0));
    for (size_t x = 0; x != first.cols; ++x) {
        for (size_t y = 0; y != first.rows; ++y) {
            stereo.at<uchar> (y, x) = disparities[x][y] * 255 / (smax - smin);
        }
    }
    
    medianBlur(stereo, stereo, 3);
    
//    namedWindow("disparity map", CV_WINDOW_AUTOSIZE);
//    imshow("disparity map", stereo);
//    waitKey(0);
//    destroyWindow("disparity map");
    
    imwrite("/home/sanity-seeker/Programming/stereo_vision/results/aloe5.jpg", stereo);
    return 0;
}
