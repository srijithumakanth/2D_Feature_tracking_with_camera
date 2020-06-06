/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

// #include "boost/circular_buffer/space_optimized.hpp"
#include "boost/circular_buffer.hpp"
#include <map>

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    // vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

    vector<string> Selectors = {"SEL_NN" ,"SEL_KNN"};
    vector<string> Matchers = {"MAT_BF", "MAT_FLANN"};
    vector<string> Detectors= {"SHITOMASI", "HARRIS","SIFT", "FAST", "BRISK", "ORB", "AKAZE"};
    vector<string> Descriptors= {"SIFT", "BRISK", "BRIEF", "ORB", "FREAK", "AKAZE"};
    
    bool bVis = true;            // visualize results
    bool bFocusOnVehicle = true; // only keep keypoints on the preceding vehicle
    bool bLimitKpts = false;
    
    //Variables for Task MP.7 MP.8 and MP.9
    map<string, int> totalKptsOnVehicle;
    map<string, int> totalMatchedKpts;
    map<string, double> timeforKptsDetect;
    map<string, double> timeforKptsDescript;

    for (auto selectorType : Selectors) //Loop Over Selector
    {
        for (auto matcherType : Matchers) //Loop Over Matchers
        {
            for (auto detectorType : Detectors) //Loop Over Detector
            {
                for (auto descriptorType : Descriptors) //Loop Over Descritpor
                {
                    // Ring buffer for memory optimization
                    boost::circular_buffer<DataFrame> dataBuffer(dataBufferSize);
                    // boost::circular_buffer_space_optimized<DataFrame> dataBuffer(dataBufferSize);
                    
                    // Detector of type SIFT and descriptor of type ORB are incompatible
                    if(detectorType.compare("SIFT")==0 && descriptorType.compare("ORB")==0) continue;

                    // Descriptor of type AKAZE only works with detector of type AKAZE
                    if(detectorType.compare("AKAZE")!=0 && descriptorType.compare("AKAZE")==0) continue;
                    
                    /* MAIN LOOP OVER ALL IMAGES */
                    int sumKptsonVehcile = 0; //to record kpts on vehicle from all 10 images
                    int sumMatchedKpts = 0;
                    double timeKptsDetect = 0;
                    double timeKptsDescript = 0;
                    double t_matchDesc = 0;

                    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
                    {
                        cout<<"Selector is: "<<selectorType<<", Matcher is: "<<matcherType<<", Detector is: "<<detectorType<<", Descriptor is: "<<descriptorType<<", Image Index is: "<<imgIndex<<endl;
                        
                        /******************  Step 1: READ IMAGE & CONVERT TO GRAYSCALE  *********************************/
                        cv::Mat img, imgGray;
                        ostringstream imgNumber;
                        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                        // load image from file and convert to grayscale

                        img = cv::imread(imgFullFilename);
                        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
                        
                        /* LOAD IMAGE INTO BUFFER */
                        // push image into data frame buffer
                        DataFrame frame;
                        frame.cameraImg = imgGray;

                        dataBuffer.push_back(frame);
                        /* // If using just a std::vector<DataFrame> dataBuffer
                        
                        // Pop old images
                        if (dataBuffer.size() > dataBufferSize)
                        {
                            dataBuffer.erase(dataBuffer.begin());
                        }
                        */

                        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                        /******************  Step 2: DETECT IMAGE KEYPOINTS  *********************************/

                        // extract 2D keypoints from current image
                        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

                        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                        timeKptsDetect += detKeypointsModern(keypoints, imgGray, detectorType, false);
                        
                        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                        cv::Rect vehicleRect(535, 180, 180, 150);
                        if (bFocusOnVehicle)
                        {
                            vector<cv::KeyPoint> filteredKeypoints;
                            vector<cv::KeyPoint> neighbourhoodKeypoints;
                            for (auto kp : keypoints)
                            {
                                if (vehicleRect.contains(kp.pt))
                                {
                                    filteredKeypoints.push_back(kp);
                                }
                                else 
                                {
                                    neighbourhoodKeypoints.push_back(kp);
                                }
                            }
                            keypoints = filteredKeypoints;
                            // keypoints = neighbourhoodKeypoints;
                        }
                        sumKptsonVehcile += keypoints.size();

                        if (bLimitKpts)
                        {
                            int maxKeypoints = 50;

                            if (detectorType.compare("SHITOMASI") == 0)
                            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                            }
                            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                            cout << " NOTE: Keypoints have been limited!" << endl;
                        }

                        // push keypoints and descriptor for current frame to end of data buffer
                        (dataBuffer.end() - 1)->keypoints = keypoints;
                        cout << "#2 : DETECT KEYPOINTS done" << endl;
                        
                        /******************  Step 3: EXTRACT KEYPOINT DESCRIPTORS  *********************************/

                        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                        cv::Mat descriptors;
                        timeKptsDescript += descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

                        // push descriptors for current frame to end of data buffer
                        (dataBuffer.end() - 1)->descriptors = descriptors;

                        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                        if (dataBuffer.size() > 1) // wait until at least two images have been processed
                        {

                            /******************  Step 4: MATCH KEYPOINT DESCRIPTORS  *********************************/
                            
                            vector<cv::DMatch> matches;
                            // string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                            // string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
                            // string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN
                            string descriptorCategory= descriptorType.compare("SIFT")==0? "DES_HOG":"DES_BINARY";

                            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                            t_matchDesc = matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                            (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                            matches, descriptorCategory, matcherType, selectorType);

                            // store matches in current data frame
                            (dataBuffer.end() - 1)->kptMatches = matches;
                            sumMatchedKpts += matches.size();

                            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                            // visualize matches between current and previous image
                            if (bVis)
                            {
                                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                                matches, matchImg,
                                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                                string windowName = "Matching keypoints between two camera images";
                                cv::namedWindow(windowName, 2);
                                cv::imshow(windowName, matchImg);
                                cout << "Press key to continue to next image" << endl;
                                cv::waitKey(1); // wait for key to be pressed
                            }
                        }

                    } // eof loop over all images
                    // Only record once , keypoints for each detector
                    if(totalKptsOnVehicle.find(detectorType)==totalKptsOnVehicle.end())
                    {
                        totalKptsOnVehicle.insert(make_pair(detectorType,sumKptsonVehcile));
                    }
                    string tmpKey = detectorType +"," + descriptorType;
                    totalMatchedKpts.insert(make_pair(tmpKey,sumMatchedKpts));
                    timeforKptsDetect.insert(make_pair(tmpKey, timeKptsDetect));
                    timeforKptsDescript.insert(make_pair(tmpKey, timeKptsDescript));

                } // eof loop descriptor

            } // eof loop detector

            if(matcherType.compare("MAT_BF") == 0) //Only check MAT_BF as reqired in MP.8
            {
                string numOfMatchedKptsFileName =  "MP_8.csv";
                ofstream numOfMatchedKptsFile(numOfMatchedKptsFileName);
                numOfMatchedKptsFile<<"Detector"<<","<<"Descriptor"<<","<<"Matched Key Points"<<endl;

                for(auto it=totalMatchedKpts.begin();it!=totalMatchedKpts.end();++it)
                {
                    numOfMatchedKptsFile<<it->first<<","<<it->second<<endl;
                }
                
                numOfMatchedKptsFile.close();
            }
            
            ///Write MP.9 Result
            string timeForKptsFileName ="totalTimeForKpts_"+ selectorType +"_" + matcherType+".csv";
            ofstream timeForKptsFile(timeForKptsFileName);
            timeForKptsFile<<"Detector"<<",Descritptor,"<<"Detector Time"<<","<<"Descriptor Time,"<<"Time per KPts (in ms)"<<endl;
            auto it2 =timeforKptsDescript.begin();
            auto it3=totalMatchedKpts.begin();
            for(auto it1=timeforKptsDetect.begin();  it1!=timeforKptsDetect.end();++it1, ++it2,++it3 )
            {
                timeForKptsFile<<it1->first<<","<<it1->second<<","<<it2->second<<","<<(it1->second + it2->second)/it3->second<<endl;
            }
            timeForKptsFile.close();

            totalMatchedKpts.clear();
            timeforKptsDetect.clear();
            timeforKptsDescript.clear();

        } // eof loop matcher
    } // eof selector loop

    ///Write MP.7 Result
    string numOfKptsFileName = "MP_7.csv";
    ofstream numOfKptsFile(numOfKptsFileName);
    numOfKptsFile<<"Detector,"<<"Number of Kpts"<<endl;
    for(auto it=totalKptsOnVehicle.begin();it!=totalKptsOnVehicle.end();++it)
    {
      numOfKptsFile<<it->first<<","<<it->second<<endl;
    }
    numOfKptsFile.close();

    return 0;
}
