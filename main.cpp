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

int penaltyEqual = 5;  // just an example of values, not tested
int penaltyDiffer = 60;

template<class T>
T mymin(std::initializer_list<T> ilist) {  // min function for multiple arguments
    return *std::min_element(ilist.begin(), ilist.end());
}

struct Direction {
    int x, y;
    Direction(int a = 0, int b = 0): x(a), y(b) {}
};

cv::Mat censusTransform3x3(const Mat& image) {
    cv::Mat censusMap(image.size(), CV_8U, Scalar(0));
    std::vector<const unsigned char*> readLines(3);
    for (size_t j = 1; j < image.rows - 1; ++j) {
        for (size_t l = 0; l != 3; ++l) {
            readLines[l] = image.ptr<unsigned char>(j + l - 1);
        }
        unsigned char * censusWriteLine = censusMap.ptr<unsigned char>(j);
        for (size_t i = 1; i < image.cols - 1; ++i) {
            int curValue = readLines[1][i];
            unsigned char censusValue = 0;
            for (size_t l = 0; l != 3; ++l) {
                for (size_t k = 0; k != 3; ++k) {
                    if (k == 0 && l == 0) {
                        continue;
                    }
                    bool less = readLines[l][i + k - 1] < curValue;
                    censusValue <<= 1;
                    if (less) censusValue += 1;
                }
                censusWriteLine[i] = censusValue;
            }
        }
    }
    return censusMap;
}

int HammingDistance (int A, int B) {
    int S = A ^ B;
    int BitCounter = 0;
    if( A < 0 || B < 0 ) {
        return 8 * sizeof(int);
    }
    while (S) {
        if (S & 1)
            ++BitCounter;
        S >>= 1;
    }
    return BitCounter;
}

cv::Mat createDisparityImage(std::vector<std::vector<int>>& disparities, int max, int min) {
    Mat stereo(disparities[0].size(), disparities.size(), CV_8UC1, Scalar(0));
    for (size_t x = 0; x != disparities.size(); ++x) {
        for (size_t y = 0; y != disparities[0].size(); ++y) {
            stereo.at<uchar> (y, x) = static_cast<uchar>(std::floor(disparities[x][y] * 255 / static_cast<double>(max - min)));
//            stereo.at<uchar> (y, x) = disparities[x][y];
        }
    }
    medianBlur(stereo, stereo, 3);
    return stereo;
}

void leaveBestPoints(std::vector<KeyPoint>& input) {
    if (input.size() > 15) {
        std::sort(input.rbegin(), input.rend(), [](KeyPoint a, KeyPoint b){return a.response * a.size < b.response * b.size;});
        input.resize(15);
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
    
    for (size_t i = 0; i != matches.size(); ++i) {  // сравнить по координатам
       int delta = static_cast<int>(std::ceil(firstKeys[matches[i].queryIdx].pt.x - secKeys[matches[i].trainIdx].pt.x));
       if (min_disp > delta) {
           min_disp = delta;
       }
       if (max_disp < delta) {
           max_disp = delta;
       }
    }
}

void countDirections(std::vector<std::vector<std::vector<int>>>& unaries, std::vector<std::vector<std::vector<int>>>& temp
        , std::vector<std::vector<std::vector<int>>>& aggregated, Direction r
        , int max_disparity, int min_disparity) {
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
        eps = -1;
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
                for (size_t d = 0; d < max_disparity - min_disparity + 1; ++d)
                    temp[x][y][d] = unaries[x][y][d];
            } else {
                int minimum = *std::min_element(temp[newX][newY].begin(), temp[newX][newY].end());
                
                for (size_t d = 0; d <= max_disparity - min_disparity + 1; ++d) {
                    if (d == max_disparity - min_disparity) {
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
            for (size_t d = 0; d != max_disparity - min_disparity + 1; ++d) {
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

void countUnary(Mat& first, Mat& second, int max_disparity, int min_disparity, std::vector<std::vector<std::vector<int>>>& unaries) {
    Mat censusFirst = censusTransform3x3(first);
    Mat censusSecond = censusTransform3x3(second);
    for (size_t y = 0; y != censusFirst.rows; ++y) {
        for (size_t x = 0; x != censusFirst.cols; ++x) {
            for (int d = 0; d < max_disparity - min_disparity + 1; ++d) {
                int currd = d + min_disparity;
//                unaries[x][y][d] = (x + currd >= 0 && x + currd < first.cols ?
//                                   abs(static_cast<int>(first.at<uchar>(y, x)) - static_cast<int>(second.at<uchar>(y, x + currd))) : 0);
                  unaries[x][y][d] = (x + currd >= 0 && x + currd < censusFirst.cols ?
                                      HammingDistance(censusFirst.at<uchar>(y, x), censusSecond.at<uchar>(y, x + currd)) : 0);
            }
        }
    }
}

void countBinary(std::vector<std::vector<std::vector<int>>>& ctable,  std::vector<std::vector<std::vector<int>>>& stable
                , std::vector<std::vector<std::vector<int>>>& temtable, int max_disp, int min_disp) {
    
    Direction up(0, 1), down(0, -1), left(-1, 0), right(1, 0), right_down(1, -1), right_up(1, 1), left_down(-1, -1), left_up(-1, 1),
            up_diag_left(-2, 1), up_diag_right(2, 1), down_diag_left(-2, -1), down_diag_right(2, -1), diag_right(1, 2), diag_left(-1, 2),
            diag_down_right(1, -2), diag_down_left(-1, -2);
    
    countDirections(ctable, temtable, stable, up, max_disp, min_disp);
    countDirections(ctable, temtable, stable, down, max_disp, min_disp);
    countDirections(ctable, temtable, stable, left, max_disp, min_disp);
    countDirections(ctable, temtable, stable, right, max_disp, min_disp);
    countDirections(ctable, temtable, stable, right_down, max_disp, min_disp);
    countDirections(ctable, temtable, stable, right_up, max_disp, min_disp);
    countDirections(ctable, temtable, stable, left_down, max_disp, min_disp);
    countDirections(ctable, temtable, stable, left_up, max_disp, min_disp);
    
    countDirections(ctable, temtable, stable, up_diag_left, max_disp, min_disp);
    countDirections(ctable, temtable, stable, up_diag_right, max_disp, min_disp);
    countDirections(ctable, temtable, stable, down_diag_left, max_disp, min_disp);
    countDirections(ctable, temtable, stable, down_diag_right, max_disp, min_disp);
    countDirections(ctable, temtable, stable, diag_right, max_disp, min_disp);
    countDirections(ctable, temtable, stable, diag_left, max_disp, min_disp);
    countDirections(ctable, temtable, stable, diag_down_right, max_disp, min_disp);
    countDirections(ctable, temtable, stable, diag_down_left, max_disp, min_disp);
}

int main() {  // there will be console flags in final version
    std::string firstSource, secSource;
//    std::cin >> firstSource >> secSource;
    firstSource = "/home/sanity-seeker/Programming/stereo_vision/dpth_map_dataset/aero/source_left.jpg";
    secSource = "/home/sanity-seeker/Programming/stereo_vision/dpth_map_dataset/aero/source_right.jpg";
    Mat first = imread(firstSource, CV_LOAD_IMAGE_GRAYSCALE);
    Mat second = imread(secSource, CV_LOAD_IMAGE_GRAYSCALE);
    
     resize(first, first, Size(600, 800), 0, 0, INTER_LINEAR);
     resize(second, second, Size(600, 800), 0, 0, INTER_LINEAR);
    
    int min_disp, max_disp;
    // measureBorders(first, second, min_disp, max_disp);
    min_disp = -10;
    max_disp = 30;
    
    std::vector<std::vector<std::vector<int>>> ctable(first.cols, (std::vector<std::vector<int>> (first.rows, std::vector<int>(max_disp - min_disp + 1, 0))));
    std::vector<std::vector<std::vector<int>>> stable(first.cols, (std::vector<std::vector<int>> (first.rows, std::vector<int>(max_disp - min_disp + 1, 0))));
    std::vector<std::vector<std::vector<int>>> temtable(first.cols, (std::vector<std::vector<int>> (first.rows, std::vector<int>(max_disp - min_disp + 1, 0))));
    std::vector<std::vector<int>> base_disparities(first.cols, (std::vector<int>(first.rows, 0)));
    std::vector<std::vector<int>> match_disparities(second.cols, (std::vector<int>(second.rows, 0)));
    // unary potential
    countUnary(first, second, max_disp, min_disp, ctable);
    
    // binary potential
    countBinary(ctable, stable, temtable, max_disp, min_disp);
    
    // select right disparity
    int smax = MININT;
    int smin = MAXINT;
    for (size_t x = 0; x != first.cols; ++x) {
        for (size_t y = 0; y != first.rows; ++y) {
            int min_index = std::min_element(stable[x][y].begin(), stable[x][y].end()) - stable[x][y].begin();
            base_disparities[x][y] = min_index;
            if (min_index > smax) {
                smax = min_index;
            }
            if (min_index < smin) {
                smin = min_index;
            }
        }
    }
    Mat base = createDisparityImage(base_disparities, smax, smin);
    
    // doing all calculations assuming match image is a base one
    for (size_t x = 0; x != first.cols; ++x) {
        for (size_t y = 0; y != first.rows; ++y) {
            std::fill(ctable[x][y].begin(), ctable[x][y].end(), 0);
            std::fill(stable[x][y].begin(), stable[x][y].end(), 0);
        }
    }
    
    int match_max_disp = std::max(-max_disp, -min_disp);
    int match_min_disp = std::min(-min_disp, -max_disp);
    countUnary(second, first, match_max_disp, match_min_disp, ctable);
    countBinary(ctable, stable, temtable, match_max_disp, match_min_disp);
    
    for (size_t x = 0; x != second.cols; ++x) {
        for (size_t y = 0; y != second.rows; ++y) {
            int min_index = std::min_element(stable[x][y].begin(), stable[x][y].end()) - stable[x][y].begin();
            match_disparities[x][y] = min_index;
        }
    }
    Mat match = createDisparityImage(match_disparities, smax, smin);
    
    for (size_t y = 0; y != match.rows; ++y) {
        for (size_t x = 0 ; x != match.cols; ++x) {
            match.at<uchar>(y, x) = 255 - match.at<uchar>(y, x);
        }
    }
    
    // calculating final disparity image
    Mat disparity(first.size(), CV_8UC1, Scalar(0));
    
    int invalid = 100;
    
    for (size_t x = 0; x != first.cols; ++x) {
        for (size_t y = 0; y != first.rows; ++y) {
            if (!(x + base_disparities[x][y] - min_disp < 0 || x + base_disparities[x][y] - min_disp >= first.cols)) {
                if (abs(base.at<uchar>(y, x) - match.at<uchar>(y, x + base_disparities[x][y] - min_disp)) <= 30) {
                    disparity.at<uchar>(y, x) = base.at<uchar>(y, x);
                } else {
                    disparity.at<uchar>(y, x) = invalid;
                }
            }
        }
    }
    //medianBlur(disparity, disparity, 3);
    
    namedWindow("base", CV_WINDOW_AUTOSIZE);
    namedWindow("match", CV_WINDOW_AUTOSIZE);
    namedWindow("final", CV_WINDOW_AUTOSIZE);
    imshow("match", match);
    imshow("base", base);
    imshow("final", disparity);
    waitKey(0);
    destroyAllWindows();
    
//    imwrite("/home/sanity-seeker/Programming/stereo_vision/results/aero/source++_" + std::to_string(min_disp) + '_' + std::to_string(max_disp) + "_" +
//            std::to_string(penaltyEqual) + "_" + std::to_string(penaltyDiffer) + ".jpg", stereo);
    return 0;
}
