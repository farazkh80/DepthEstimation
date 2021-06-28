#include "../inc/StereoVision.h"

void StereoVision::undistort_frame(Mat& frame) {

	Mat cameraMatrix, newCameraMatrix;
	vector<double> distortionParameters;

	newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distortionParameters, { frame.cols, frame.rows }, 1);

	undistort(frame, frame, cameraMatrix, distortionParameters, newCameraMatrix);

}

Mat StereoVision::add_HSV_filter(Mat& frame, int camera) {

	// Blurring the frame for noise reduction
	GaussianBlur(frame, frame, { 5,5 }, 0);

	// Convert to HSV
	cvtColor(frame, frame, COLOR_BGR2HSV);

	Mat msk;

	// limits for red ball
	vector<int> lowerLimitRedLeft = { 60, 110, 50 };
	vector<int> upperLimitRedLeft = { 255, 255, 255 };
	vector<int> lowerLimitRedRight = { 140, 110, 50 };
	vector<int> upperLimitRedRight = { 255, 255, 255 };

	if (camera == 1)
		inRange(frame, lowerLimitRedRight, upperLimitRedRight, msk);
	else
		inRange(frame, lowerLimitRedLeft, upperLimitRedLeft, msk);

	// Methalogy for better image processing
	erode(msk, msk, (3, 3));
	dilate(msk, msk, (3, 3));

	return msk;


}

Point StereoVision::find_ball(Mat& frame, Mat& msk) {
	vector<vector<Point> > contours;

	findContours(msk, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Sort the contours to find the biggest one
	sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2) {
		return contourArea(c1, false) < contourArea(c2, false);
		});

	// if we have any valid contours
	if (contours.size() > 0) {

		// get the largest contour only
		vector<Point> largestContour = contours[contours.size() - 1];
		Point2f center;
		float radius;
		// find the biggest enclosed circle in the largest contour
		minEnclosingCircle(largestContour, center, radius);
		// get the center of the contour
		Moments m = moments(largestContour);
		// The center point of the contour
		Point centerPoint(m.m10 / m.m00, m.m01 / m.m00);

		// Only preceed if the radius is grater than a minimum threshold
		if (radius > MINIMAL_CONTOUR_THRESHOLD) {
			// Draw the circle and centroid on the frame
			circle(frame, center, int(radius), (0, 255, 255), 2);
			circle(frame, centerPoint, 5, (0, 0, 255), -1);
		}

		return centerPoint;
	}

	return { 0,0 };
}


float StereoVision::find_depth(Point circleLeft, Point circleRight, Mat& leftFrame, Mat& rightFrame) {

	int focal_pixels = 0;

	if (rightFrame.cols == leftFrame.cols) {

		// Convert focal lenght f from [mm] to [pixel]
		focal_pixels = (rightFrame.cols * 0.5) / tan(alpha * 0.5 * CV_PI / 180.0);
	}
	else {
		cout << "Left and Right Camera frames do not have the same pixel width" << endl;
	}

	int xLeft = circleLeft.x;
	int xRight = circleRight.x;

	// Calculate the disparity
	int disparity = xLeft - xRight;

	// Calculate the Depth Z
	float zDepth = (baseline * float(focal_pixels)) / float(disparity);    // Depth in [cm]

	return abs(zDepth);

}