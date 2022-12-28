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
    
    Mat grayImages[imagesCount];
    for (int i = 0; i < imagesCount; i++) {
        cvtColor(images[i], grayImages[i], COLOR_BGR2GRAY);
    }

    Mat gaussianBlurImages[imagesCount];
    int kernelSize = 5;
    for (int i = 0; i < imagesCount; i++) {
        GaussianBlur(grayImages[i], gaussianBlurImages[i], Size(kernelSize, kernelSize), 0);
    }

    Mat edges[imagesCount];
    int lowThreshold = 50;
    int highThreshold = 150;
    int apertureSize = 3;
    bool l2Gradient = false;
    for (int i = 0; i < imagesCount; i++) {
        Canny(grayImages[i], edges[i], lowThreshold, highThreshold, apertureSize, l2Gradient);
    }

    int rho = 1;
    double theta = M_PI / 180;
    int threshold = 25;
    int minLineLength = 30;
    std::vector<Vec4i> lines[imagesCount];
    for (int i = 0; i < imagesCount; i++) {
        HoughLinesP(edges[i], lines[i], rho, theta, threshold, minLineLength);
    }

    Mat outImages[imagesCount];
    for (int i = 0; i < imagesCount; i++) {
        outImages[i] = images[i].clone();
    }

    for (int i = 0; i < imagesCount; i++) {
        for(size_t j = 0; j < lines[i].size(); j++) {
            line(outImages[i], Point(lines[i][j][0], lines[i][j][1]), Point(lines[i][j][2], lines[i][j][3]), Scalar(0, 0, 255), 3, LINE_AA);
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
