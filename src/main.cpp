#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
int main()
{
    const int imagesCount = 6;
    const int colors = 3;
    const int imgPartWidth = 84;
    const int imgPartHeight = 63;
    const int imgPadding = 2;

    std::string imagePath = "img/321141404_575114514442518_1353911691399830858_n.jpg";
    Mat fullImage = imread(imagePath, IMREAD_COLOR);
    if (fullImage.empty()) {
        std::cout << "Could not read the image: " << imagePath << std::endl;
        return 1;
    }
    imshow("Full image", fullImage);

    Mat imageA = fullImage(Range(0, imgPartHeight - 1), Range(0, imgPartWidth - 1));
    Mat imageB = fullImage(Range(0, imgPartHeight - 1), Range(imgPadding + imgPartWidth, imgPadding + imgPartWidth * 2 - 1));
    Mat imageC = fullImage(Range(0, imgPartHeight - 1), Range(imgPadding * 2 + imgPartWidth * 2, imgPadding * 2 + imgPartWidth * 3 - 1));
    Mat imageD = fullImage(Range(imgPadding + imgPartHeight, imgPadding + imgPartHeight * 2 - 1), Range(0, imgPartWidth - 1));
    Mat imageE = fullImage(Range(imgPadding + imgPartHeight, imgPadding + imgPartHeight * 2 - 1), Range(imgPadding + imgPartWidth, imgPadding + imgPartWidth * 2 - 1));
    Mat imageF = fullImage(Range(imgPadding + imgPartHeight, imgPadding + imgPartHeight * 2 - 1), Range(imgPadding * 2 + imgPartWidth * 2, imgPadding * 2 + imgPartWidth * 3 - 1));
    Mat images[imagesCount] = {imageA, imageB, imageC, imageD, imageE, imageF};
    
    Mat separateColorspace[imagesCount][colors];
    for (int i = 0; i < imagesCount; i++) {
        Mat colorSplitImage[colors];
        split(images[i], colorSplitImage);
        for (int j = 0; j < colors; j++) {
            Mat imageWithSingleColor[colors];
            for (int k = 0; k < colors; k++) {
                if (k == j) {
                    imageWithSingleColor[k] = colorSplitImage[k].clone();
                } else {
                    imageWithSingleColor[k] = Mat::zeros(colorSplitImage[k].rows, colorSplitImage[k].cols, CV_8UC1);
                }
            }
            Mat tmpResult;
            merge(imageWithSingleColor, colors, tmpResult);
            separateColorspace[i][j] = tmpResult;
        }
    }

    Mat grayImages[imagesCount][colors];
    for (int i = 0; i < imagesCount; i++) {
        for (int j = 0; j < colors; j++) {
            cvtColor(separateColorspace[i][j], grayImages[i][j], COLOR_BGR2GRAY);
            equalizeHist(grayImages[i][j], grayImages[i][j]);
        }
    }

    Mat gaussianBlurImages[imagesCount][colors];
    int kernelSize = 5;
    for (int i = 0; i < imagesCount; i++) {
        for (int j = 0; j < colors; j++) {
            GaussianBlur(grayImages[i][j], gaussianBlurImages[i][j], Size(kernelSize, kernelSize), 0);
        }
    }

    Mat edges[imagesCount][colors];
    int lowThreshold = 1;
    int highThreshold = 5;
    int apertureSize = 3;
    bool l2Gradient = false;
    for (int i = 0; i < imagesCount; i++) {
        for (int j = 0; j < colors; j++) {
            Canny(grayImages[i][j], edges[i][j], lowThreshold, highThreshold, apertureSize, l2Gradient);
        }
    }

    int rho = 1;
    double theta = M_PI / 180;
    int threshold = 25;
    int minLineLength = 15;
    int maxLineGap = 0;
    std::vector<Vec4i> lines[imagesCount][colors];
    for (int i = 0; i < imagesCount; i++) {
        for (int j = 0; j < colors; j++) {
            HoughLinesP(edges[i][j], lines[i][j], rho, theta, threshold, minLineLength, maxLineGap);
        }
    }

    Mat separateColorspaceOutImages[imagesCount][colors];
    for (int i = 0; i < imagesCount; i++) {
        for (int j = 0; j < colors; j++) {
            separateColorspaceOutImages[i][j] = separateColorspace[i][j].clone();
        }
    }

    for (int i = 0; i < imagesCount; i++) {
        for (int j = 0; j < colors; j++) {
            for(size_t k = 0; k < lines[i][j].size(); k++) {
                line(separateColorspaceOutImages[i][j], Point(lines[i][j][k][0], lines[i][j][k][1]), Point(lines[i][j][k][2], lines[i][j][k][3]), Scalar(255, 255, 255), 2, LINE_AA);
            }
        }
    }

    Mat outImages[imagesCount];
    for (int i = 0; i < imagesCount; i++) {
        outImages[i] = images[i].clone();
    }

    
    for (int i = 0; i < imagesCount; i++) {
        for (int j = 0; j < colors; j++) {
            imshow(std::to_string(i) + " " + std::to_string(j), separateColorspaceOutImages[i][j]);
        }
    }

    imshow("A", outImages[0]);
    imshow("B", outImages[1]);
    imshow("C", outImages[2]);
    imshow("D", outImages[3]);
    imshow("E", outImages[4]);
    imshow("F", outImages[5]);

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
