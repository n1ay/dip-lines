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
    const int imgPartWidth = 85;
    const int imgPartHeight = 64;
    const int imgPadding = 1;

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

    Mat grayA, grayB, grayC, grayD, grayE, grayF; 
    cvtColor(imageA, grayA, COLOR_BGR2GRAY);
    cvtColor(imageB, grayB, COLOR_BGR2GRAY);
    cvtColor(imageC, grayC, COLOR_BGR2GRAY);
    cvtColor(imageD, grayD, COLOR_BGR2GRAY);
    cvtColor(imageE, grayE, COLOR_BGR2GRAY);
    cvtColor(imageF, grayF, COLOR_BGR2GRAY);

    int kernelSize = 5;
    Mat gaussianBlurA, gaussianBlurB, gaussianBlurC, gaussianBlurD, gaussianBlurE, gaussianBlurF;
    GaussianBlur(grayA, gaussianBlurA, Size(kernelSize, kernelSize), 0);
    GaussianBlur(grayB, gaussianBlurB, Size(kernelSize, kernelSize), 0);
    GaussianBlur(grayC, gaussianBlurC, Size(kernelSize, kernelSize), 0);
    GaussianBlur(grayD, gaussianBlurD, Size(kernelSize, kernelSize), 0);
    GaussianBlur(grayE, gaussianBlurE, Size(kernelSize, kernelSize), 0);
    GaussianBlur(grayF, gaussianBlurF, Size(kernelSize, kernelSize), 0);

    int lowThreshold = 50;
    int highThreshold = 150;
    int apertureSize = 3;
    bool l2Gradient = false;
    Mat edgesA, edgesB, edgesC, edgesD, edgesE, edgesF;
    Canny(gaussianBlurA, edgesA, lowThreshold, highThreshold, apertureSize, l2Gradient);
    Canny(gaussianBlurB, edgesB, lowThreshold, highThreshold, apertureSize, l2Gradient);
    Canny(gaussianBlurC, edgesC, lowThreshold, highThreshold, apertureSize, l2Gradient);
    Canny(gaussianBlurD, edgesD, lowThreshold, highThreshold, apertureSize, l2Gradient);
    Canny(gaussianBlurE, edgesE, lowThreshold, highThreshold, apertureSize, l2Gradient);
    Canny(gaussianBlurF, edgesF, lowThreshold, highThreshold, apertureSize, l2Gradient);

    int rho = 1;
    double theta = M_PI / 180;
    int threshold = 25;
    int minLineLength = 30;
    std::vector<Vec4i> linesA, linesB, linesC, linesD, linesE, linesF;
    HoughLinesP(edgesA, linesA, rho, theta, threshold, minLineLength);
    HoughLinesP(edgesB, linesB, rho, theta, threshold, minLineLength);
    HoughLinesP(edgesC, linesC, rho, theta, threshold, minLineLength);
    HoughLinesP(edgesD, linesD, rho, theta, threshold, minLineLength);
    HoughLinesP(edgesE, linesE, rho, theta, threshold, minLineLength);
    HoughLinesP(edgesF, linesF, rho, theta, threshold, minLineLength);

    Mat outA = imageA.clone();
    Mat outB = imageB.clone();
    Mat outC = imageC.clone();
    Mat outD = imageD.clone();
    Mat outE = imageE.clone();
    Mat outF = imageF.clone();

    for(size_t i = 0; i < linesA.size(); i++) {
        line(outA, Point(linesA[i][0], linesA[i][1]), Point(linesA[i][2], linesA[i][3]), Scalar(0, 0, 255), 3, LINE_AA);
    }
    for(size_t i = 0; i < linesB.size(); i++) {
        line(outB, Point(linesB[i][0], linesB[i][1]), Point(linesB[i][2], linesB[i][3]), Scalar(0, 0, 255), 3, LINE_AA);
    }
    for(size_t i = 0; i < linesC.size(); i++) {
        line(outC, Point(linesC[i][0], linesC[i][1]), Point(linesC[i][2], linesC[i][3]), Scalar(0, 0, 255), 3, LINE_AA);
    }
    for(size_t i = 0; i < linesD.size(); i++) {
        line(outD, Point(linesD[i][0], linesD[i][1]), Point(linesD[i][2], linesD[i][3]), Scalar(0, 0, 255), 3, LINE_AA);
    }
    for(size_t i = 0; i < linesE.size(); i++) {
        line(outE, Point(linesE[i][0], linesE[i][1]), Point(linesE[i][2], linesE[i][3]), Scalar(0, 0, 255), 3, LINE_AA);
    }
    for(size_t i = 0; i < linesF.size(); i++) {
        line(outF, Point(linesF[i][0], linesF[i][1]), Point(linesF[i][2], linesF[i][3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    imshow("A", outA);
    imshow("B", outB);
    imshow("C", outC);
    imshow("D", outD);
    imshow("E", outE);
    imshow("F", outF);

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
