#include <stdio.h>
#include <opencv2/opencv.hpp>

int main(){
    cv::VideoCapture cap1(0);
    if (!cap1.isOpened()){
        printf("Error opening video stream or file\n");
        return -1;
    }

    cv::VideoCapture cap2(1);
    if (!cap2.isOpened()){
        printf("Error opening video stream or file\n");
        return -1;
    }

    cv::namedWindow("Camera 1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Camera 2", cv::WINDOW_AUTOSIZE);

    while(true) {
        cv::Mat f1, f2;

        cap1 >> f1;
        cap2 >> f2;

        if(f1.empty() || f2.empty()) {
            printf("Empty frame captured. \n");
            break;
        }

        cv::imshow("Camera 1", f1);
        cv::imshow("Camera 2", f2);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    cap1.release();
    cap2.release();
    cv::destroyAllWindows();
}
