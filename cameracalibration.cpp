#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>


int CHECKERBOARD[2]{9, 6};

int main()
{
   
    std::vector<std::vector<cv::Point3f>> objpoints;

    
    std::vector<std::vector<cv::Point2f>> imgpoints;

    
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CHECKERBOARD[1]; i++)
    {
        for (int j = 0; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j, i, 0));
    }

   
    std::vector<cv::String> images;
    
    std::string path = "/Users/divyanshu/iitj_project_image_data/Camaracalibratio /*.png";

    cv::glob(path, images);

    cv::Mat frame, gray;
    
    std::vector<cv::Point2f> corner_pts;
    bool success;

    
    std::vector<double> fx_array, fy_array, cx_array, cy_array;
    std::vector<cv::Mat> camera_matrices;
    std::vector<cv::Mat> dist_coeffs_array;
    std::vector<double> mean_errors;

    
    for (int batch_size = 5; batch_size <= 155; batch_size += 5)
    {
        objpoints.clear();
        imgpoints.clear();

        
        for (int i = 0; i < std::min(batch_size, (int)images.size()); i++)
        {
            frame = cv::imread(images[i]);
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            
            success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts,
                                                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

            
            if (success)
            {
                cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001);

                
                cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

               
                cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

                objpoints.push_back(objp);
                imgpoints.push_back(corner_pts);
            }
            else
            {
                std::cout << "Failed to process image: " << images[i] << std::endl;
            }

            cv::imshow("Image", frame);
            cv::waitKey(1000);
        }

        cv::destroyAllWindows();

        if (objpoints.empty() || imgpoints.empty())
        {
            std::cout << "No valid checkerboard images found in the first " << batch_size << " images." << std::endl;
            continue;
        }

        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F); 
        cv::Mat distCoeffs;
        std::vector<cv::Mat> rvecs, tvecs;

        
        cv::calibrateCamera(objpoints, imgpoints, gray.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
        std::cout << "distCoeffs : " << distCoeffs << std::endl;

       
        double fx = cameraMatrix.at<double>(0, 0);
        double fy = cameraMatrix.at<double>(1, 1);
        double cx = cameraMatrix.at<double>(0, 2);
        double cy = cameraMatrix.at<double>(1, 2);

        fx_array.push_back(fx);
        fy_array.push_back(fy);
        cx_array.push_back(cx);
        cy_array.push_back(cy);

        camera_matrices.push_back(cameraMatrix.clone());
        dist_coeffs_array.push_back(distCoeffs.clone());

       
        double total_error = 0;
        double total_points = 0;
        for (size_t i = 0; i < objpoints.size(); i++)
        {
            std::vector<cv::Point2f> imgpoints2;
            cv::Mat rvec = rvecs[i];
            cv::Mat tvec = tvecs[i];
            cv::projectPoints(objpoints[i], rvec, tvec, cameraMatrix, distCoeffs, imgpoints2);
            double error = cv::norm(imgpoints[i], imgpoints2, cv::NORM_L2);
            double per_image_error = std::sqrt(error * error / objpoints[i].size());
            total_error += error * error;
            total_points += objpoints[i].size();
        }
        double mean_error = std::sqrt(total_error / total_points);
        mean_errors.push_back(mean_error);
        std::cout << "Mean reprojection error: " << mean_error << std::endl;
    }

    
    std::cout << "fx_array: ";
    for (double fx : fx_array) std::cout << fx << " ";
    std::cout << std::endl;

    std::cout << "fy_array: ";
    for (double fy : fy_array) std::cout << fy << " ";
    std::cout << std::endl;

    std::cout << "cx_array: ";
    for (double cx : cx_array) std::cout << cx << " ";
    std::cout << std::endl;

    std::cout << "cy_array: ";
    for (double cy : cy_array) std::cout << cy << " ";
    std::cout << std::endl;

    
    for (size_t i = 0; i < camera_matrices.size(); i++)
    {
        std::cout << "Batch " << (i + 1) * 5 << ":" << std::endl;
        std::cout << "cameraMatrix : " << camera_matrices[i] << std::endl;
        std::cout << "distCoeffs : " << dist_coeffs_array[i] << std::endl;
        std::cout << "Mean reprojection error: " << mean_errors[i] << std::endl;
    }

    return 0;
}
