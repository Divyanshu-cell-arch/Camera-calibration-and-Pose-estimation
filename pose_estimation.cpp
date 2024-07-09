#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

void draw(Mat &img, vector<Point2f> &corners, vector<Point2f> &imgpts) {
    line(img, corners[0], imgpts[0], Scalar(255, 0, 0), 5);
    line(img, corners[0], imgpts[1], Scalar(0, 255, 0), 5);
    line(img, corners[0], imgpts[2], Scalar(0, 0, 255), 5);
    putText(img, "X", imgpts[0], FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
    putText(img, "Y", imgpts[1], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    putText(img, "Z", imgpts[2], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    putText(img, "O", corners[0], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
}

vector<double> rotationMatrixToEulerAngles(Mat &R) {
    double sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return {x, y, z};
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: ./pose_estimate <calibration_file.xml> <image_directory>" << endl;
        return -1;
    }

    string calibrationFile = argv[1];
    string imageDirectory = argv[2];

    // Load camera matrix and distortion coefficients
    FileStorage fs(calibrationFile, FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "Failed to open calibration file." << endl;
        return -1;
    }

    Mat cameraMatrix, distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    // Load the board size, square size and pattern type from the XML file
    int boardWidth, boardHeight, squareSize;
    string patternType;
    fs["BoardSize_Width"] >> boardWidth;
    fs["BoardSize_Height"] >> boardHeight;
    fs["Square_Size"] >> squareSize;
    fs["Calibrate_Pattern"] >> patternType;
    fs.release();

    // Define the chessboard pattern size and 3D points
    Size patternSize(boardWidth, boardHeight);
    vector<Point3f> objp;
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            objp.push_back(Point3f(j * squareSize, i * squareSize, 0.0f));
        }
    }

    vector<Point3f> axis = { Point3f(3 * squareSize, 0, 0), Point3f(0, 3 * squareSize, 0), Point3f(0, 0, -3 * squareSize) };

    // Open CSV file to write results
    ofstream csvFile("output_data.csv");
    csvFile << "translation_x,translation_y,translation_z,rotation_x,rotation_y,rotation_z,euler_x,euler_y,euler_z,inputImageFile" << endl;

    // Iterate over images in the directory
    for (const auto &entry : fs::directory_iterator(imageDirectory)) {
        string inputImageFile = entry.path().string();
        Mat img = imread(inputImageFile);
        if (img.empty()) {
            cout << "Failed to load image: " << inputImageFile << endl;
            continue;
        }

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, patternSize, corners);

        if (found) {
            cout << "Chessboard corners found in " << inputImageFile << endl;
            drawChessboardCorners(img, patternSize, Mat(corners), found);

            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), 
                        TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));

            Mat rvec, tvec;
            solvePnP(objp, corners, cameraMatrix, distCoeffs, rvec, tvec);

            vector<Point2f> imgpts;
            projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs, imgpts);

            draw(img, corners, imgpts);

            Mat R;
            Rodrigues(rvec, R);
            vector<double> euler = rotationMatrixToEulerAngles(R);

            csvFile << tvec.at<double>(0) << "," << tvec.at<double>(1) << "," << tvec.at<double>(2) << ","
                    << rvec.at<double>(0) << "," << rvec.at<double>(1) << "," << rvec.at<double>(2) << ","
                    << euler[0] << "," << euler[1] << "," << euler[2] <<","<< inputImageFile << endl;

            imwrite("output_" + entry.path().filename().string(), img);
        } else {
            cout << "Chessboard corners not found in " << inputImageFile << endl;
        }
    }

    csvFile.close();
    return 0;
}
