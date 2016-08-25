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

void countLcost(const std::vector<std::vector<std::vector<int>>>& unaries
                , std::vector<std::vector<std::vector<int>>>& aggregated
                , std::vector<std::vector<std::vector<int>>>& temp
                , int x, int y, int d, const Direction r, const int max_disparity) {
    if (temp[y][x][d] == 0) {
        temp[y][x][d] = unaries[y][x][d];
        aggregated[y][x][d] += unaries[y][x][d];  // add to final array
        
        int currd = d - max_disparity;
        int newX = x - r.x;
        int newY = y - r.y;
        
        if (newX >= 0 && newX < unaries[0].size() && newY >= 0 && newY < unaries.size()) {
            double minimum = MAXINT;
            double kmin = MAXINT;
            for (int i = -max_disparity; i <= max_disparity; ++i) {
                countLcost(unaries, aggregated, temp, newX, newY, i + max_disparity, r, max_disparity);
                if (abs(currd - i) > 1 && temp[newY][newX][i + max_disparity] < minimum) {
                    minimum = temp[newY][newX][i + max_disparity];
                } else if (temp[newY][newX][i + max_disparity] < kmin) {
                    kmin = temp[newY][newX][i + max_disparity];
                }
            }
            
            if (currd == -max_disparity) {
                temp[y][x][d] += static_cast<int>(ceil(mymin<double>({temp[newY][newX][d]
                        , temp[newY][newX][d + 1] + penaltyEqual
                        , minimum + penaltyDiffer}) - kmin));
                
                aggregated[y][x][d] += static_cast<int>(ceil(mymin<double>({temp[newY][newX][d]
                        , temp[newY][newX][d + 1] + penaltyEqual
                        , minimum + penaltyDiffer}) - kmin));
            } else if (currd == max_disparity) {
                temp[y][x][d] += static_cast<int>(ceil(mymin<double>({temp[newY][newX][d]
                        , temp[newY][newX][d - 1] + penaltyEqual
                        , minimum + penaltyDiffer}) - kmin));
                
                aggregated[y][x][d] += static_cast<int>(ceil(mymin<double>({temp[newY][newX][d]
                        , temp[newY][newX][d - 1] + penaltyEqual
                        , minimum + penaltyDiffer}) - kmin));
            } else {
                temp[y][x][d] += static_cast<int>(ceil(mymin<double>({temp[newY][newX][d]
                        , temp[newY][newX][d - 1] + penaltyEqual
                        , temp[newY][newX][d + 1] + penaltyEqual
                        , minimum + penaltyDiffer}) - kmin));
    
                aggregated[y][x][d] += static_cast<int>(ceil(mymin<double>({temp[newY][newX][d]
                             , temp[newY][newX][d - 1] + penaltyEqual
                       , temp[newY][newX][d + 1] + penaltyEqual
                       , minimum + penaltyDiffer}) - kmin));
            }
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
    
    std::vector<std::vector<std::vector<int>>> ctable(first.rows, (std::vector<std::vector<int>> (first.cols, std::vector<int>(2 * max_disp + 1, 0))));
    std::vector<std::vector<std::vector<int>>> stable(first.rows, (std::vector<std::vector<int>> (first.cols, std::vector<int>(2 * max_disp + 1, 0))));
    std::vector<std::vector<std::vector<int>>> temtable(first.rows, (std::vector<std::vector<int>> (first.cols, std::vector<int>(2 * max_disp + 1, 0))));
    std::vector<std::vector<int>> disparities(first.rows, (std::vector<int> (first.cols, 0)));
    // unary potential
    for (size_t y = 0; y != first.rows; ++y) {
        for (size_t x = 0; x != first.cols; ++x) {
            for (int d = 0; d <= 2 * max_disp; ++d) {
                int currd = d - max_disp;
                ctable[y][x][d] = (x + currd >= 0 && x + currd < first.cols ?
                         abs(static_cast<int>(first.at<uchar>(y, x)) - static_cast<int>(second.at<uchar>(y, x + currd))) : 0);
            }
        }
    }
    
    // binary potential
    Direction up(0, -1), down(0, 1), left(1, 0), right(-1, 0), right_down(-1, 1), right_up(-1, -1), left_down(1, 1), left_up(-1, -1);
    for (size_t y = 0; y != first.rows; ++y) {
        for (size_t x = 0; x != first.cols; ++x) {
            for (int d = 0; d <= 2 * max_disp; ++d) {
                countLcost(ctable, stable, temtable, x, y, d, right, max_disp);  // не совсем понятно, как поступать с суммой
                temtable.clear();
                countLcost(ctable, stable, temtable, x, y, d, left, max_disp);
                temtable.clear();
                countLcost(ctable, stable, temtable, x, y, d, up, max_disp);
                temtable.clear();
                countLcost(ctable, stable, temtable, x, y, d, down, max_disp);
                temtable.clear();
                countLcost(ctable, stable, temtable, x, y, d, right_up, max_disp);
                temtable.clear();
                countLcost(ctable, stable, temtable, x, y, d, right_down, max_disp);
                temtable.clear();
                countLcost(ctable, stable, temtable, x, y, d, left_up, max_disp);
                temtable.clear();
                countLcost(ctable, stable, temtable, x, y, d, left_down, max_disp);
                temtable.clear();
            }
        }
    }
    
    // select right disparity (not sure)
    for (size_t y = 0; y != first.rows; ++y) {
        for (size_t x = 0; x != first.cols; ++x) {
            int min_index =
                    std::min_element(stable[y][x].begin(), stable[y][x].end()) - stable[y][x].begin() - max_disp;
            disparities[y][x] = min_index;
        }
    }
    
    return 0;
}
