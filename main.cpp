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

double penaltyEqual = 1;  // just an example of values, not tested
double penaltyDiffer = 2;

template<class T>
T mymin(std::initializer_list<T> ilist) {  // min function for multiple arguments
    return *std::min_element(ilist.begin(), ilist.end());
}

struct Direction {
    int x, y;
    Direction(int a = 0, int b = 0): x(a), y(b) {}
};

void leaveBestPoints(std::vector<KeyPoint>& input) {
    if (input.size() > 300) {
        std::sort(input.rbegin(), input.rend(), [](KeyPoint a, KeyPoint b){return a.response * a.size < b.response * b.size;});
        input.resize(300);
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
    
    std::vector<std::vector<std::vector<int>>> ctable(first.cols, (std::vector<std::vector<int>> (first.rows, std::vector<int>(2 * max_disp + 1, 0))));
    std::vector<std::vector<std::vector<int>>> stable(first.cols, (std::vector<std::vector<int>> (first.rows, std::vector<int>(2 * max_disp + 1, 0))));
    std::vector<std::vector<std::vector<int>>> temtable(first.cols, (std::vector<std::vector<int>> (first.rows, std::vector<int>(2 * max_disp + 1, 0))));
    std::vector<std::vector<int>> disparities(first.cols, (std::vector<int> (first.rows, 0)));
    // unary potential
    for (size_t y = 0; y != first.rows; ++y) {
        for (size_t x = 0; x != first.cols; ++x) {
            for (int d = 0; d <= 2 * max_disp; ++d) {
                int currd = d - max_disp;
                ctable[x][y][d] = (x + currd >= 0 && x + currd < first.cols ?
                         abs(static_cast<int>(first.at<uchar>(y, x)) - static_cast<int>(second.at<uchar>(y, x + currd))) : 0);
            }
        }
    }
    
    // binary potential
    Direction up(0, 1), down(0, -1), left(-1, 0), right(1, 0), right_down(1, -1), right_up(1, 1), left_down(-1, -1), left_up(-1, 1);
    
    // right
    for (size_t x = 0; x != first.cols; ++x) {
        for (size_t y = 0; y != first.rows; ++y) {
            int newX = x - right.x;
            
            if (newX < 0) {
                for (size_t d = 0; d <= 2 * max_disp; ++d)
                temtable[x][y][d] = ctable[x][y][d];
            } else {
                std::vector<int> disps (4, 0);  // top 4 minimums of previous pixel with their disparities  to know exactly
                std::vector<int> minimums (4, MAXINT);  // min(i) and min(k)
                for (size_t i = 0; i <= 2 * max_disp; ++i) {
                    for (size_t j = 0; j != minimums.size(); ++j) {
                        if (temtable[newX][y][i] < minimums[j]) {
                            for (size_t m = j + 1; m < minimums.size(); ++m) {
                                minimums[m] = minimums[m - 1];
                                disps[m] = disps[m - 1];
                            }
                            minimums[j] = temtable[newX][y][i];
                            disps[j] = i;
                            break;
                        }
                    }
                }
                
                for (size_t d = 0; d <= 2 * max_disp; ++d) {
                    int minimum;
                    for(size_t j = 0; j != disps.size(); ++j) {
                           if (abs(d - disps[j]) > 1) {
                               minimum = minimums[j];
                               break;
                           }
                    }
                    if (d == 2 * max_disp) {
                        temtable[x][y][d] = ctable[x][y][d] + static_cast<int>(mymin<double>({temtable[newX][y][d],
                                                                                              temtable[newX][y][d - 1] +
                                                                                              penaltyEqual,
                                                                                              minimum +
                                                                                              penaltyDiffer})) - minimums.front();
                    } else if (d == 0) {
                        temtable[x][y][d] = ctable[x][y][d] + static_cast<int>(mymin<double>({temtable[newX][y][d],
                                                                                              temtable[newX][y][d + 1] +
                                                                                              penaltyEqual,
                                                                                              minimum +
                                                                                              penaltyDiffer})) - minimums.front();
                    } else {
                        temtable[x][y][d] = ctable[x][y][d] + static_cast<int>(mymin<double>({temtable[newX][y][d],
                                                                                              temtable[newX][y][d - 1] +
                                                                                              penaltyEqual,
                                                                                              temtable[newX][y][d + 1] +
                                                                                              penaltyEqual,
                                                                                              minimum +
                                                                                              penaltyDiffer})) - minimums.front();
                    }
                }
            }
        }
    }
    
    for (size_t x = 0; x != first.cols; ++x) {
        for (size_t y = 0; y != first.rows; ++y) {
            for (size_t d = 0; d != 2 * max_disp; ++d) {
                stable[x][y][d] += temtable[x][y][d];
            }
        }
    }
    temtable.clear();
    
    // select right disparity (not sure)
    for (size_t x = 0; x != first.cols; ++x) {
        for (size_t y = 0; y != first.rows; ++y) {
            int min_index = std::min_element(stable[x][y].begin(), stable[x][y].end()) - stable[x][y].begin() - max_disp;
            disparities[x][y] = min_index;
        }
    }
    
    return 0;
}
