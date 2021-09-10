#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "inc/StereoVision.h"

using namespace std;
using namespace cv;

float baseLine = 7.0;
float focalLength = 6.0;
float alpha = 56.6;


int main()
{
	Mat leftFrame, rightFrame;
	VideoCapture capLeft(1);
	VideoCapture capRight(0);

	if (!capLeft.isOpened()) {
		cout << "Cannot Open Left Camera" << endl;
		return -1;
	}

	waitKey(0);

	if (!capRight.isOpened()) {
		cout << "Cannot Open Right Camera" << endl;
		return -1;
	}

	waitKey(0);

	StereoVision stereoVision(baseLine, alpha, focalLength);

	Mat leftMsk, rightMsk;
	Mat leftResFrame, rightResFrame;

	Point leftCircle, rightCircle;
	float ballDepth = 0;

	while (true) {

		capLeft.read(leftFrame);
		capRight.read(rightFrame);

		// Calibration of the frames
		/*stereoVision.undistort_frame(leftFrame);
		stereoVision.undistort_frame(rightFrame);*/


		// Applying HSV-filter
		leftMsk = stereoVision.add_HSV_filter(leftFrame, 0);
		rightMsk = stereoVision.add_HSV_filter(rightFrame, 1);


		// Frames after applyting HSV-filter mask
		bitwise_and(leftFrame, leftFrame, leftResFrame, leftMsk);
		bitwise_and(rightFrame, rightFrame, rightResFrame, rightMsk);


		// Detect Circles - Hough Transforms can be used aswell or some neural network to do object detection
		leftCircle = stereoVision.find_ball(leftFrame, leftMsk);
		rightCircle = stereoVision.find_ball(rightFrame, rightMsk);



		// Calculate the depth of the ball

	   // If no ball is detected in one of the cameras - show the text "tracking lost"
		if (!leftCircle.x || !rightCircle.x) {
			putText(leftFrame, "Tracking Lost", { 75, 50 }, FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
			putText(rightFrame, "Tracking Lost!", { 75, 75 }, FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
		}
		else {

			// Vector of all depths in case of several balls detected.
			// All formulas used to find depth is in the video presentation
			ballDepth = stereoVision.find_depth(leftCircle, rightCircle, leftFrame, rightFrame);

			putText(leftFrame, "Tracking!", { 75, 50 }, FONT_HERSHEY_SIMPLEX, 0.7, (125, 250, 0), 2);
			putText(rightFrame, "Tracking!", { 75, 75 }, FONT_HERSHEY_SIMPLEX, 0.7, (125, 250, 0), 2);


			// Multiply computer value with 205.8 to get real - life depth in[cm]. The factor was found manually.
			cout << "Ball depth: " << ballDepth << endl;

		}

		// Show the frames
		imshow("Left Frame", leftFrame);
		imshow("Right Frame", rightFrame);
		imshow("Left Mask", leftMsk);
		imshow("Right Mask", rightMsk);

		// Hit "q" to close the window
		if ((waitKey(1) & 0xFF) == 'q') {
			break;
		}
	}

	// Release and destroy all windows before termination
	capLeft.release();
	capRight.release();

	destroyAllWindows();

	return 0;
}