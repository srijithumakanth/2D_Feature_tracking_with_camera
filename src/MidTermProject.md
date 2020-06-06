## Camera Based 2D Feature Tracking

### MP.1 Data Buffer Optimization

---
```boost::circular_buffer``` , was used to achieve the required functionality of data buffer optimization. Necessary changes on the project's CMake files were made to link boost target libraries.

Boost documentation for the same: [boost circular buffer](https://www.boost.org/doc/libs/1_61_0/doc/html/circular_buffer.html)

## MP.2 Keypoint Detection
<font size="3">
Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.</font>

---
### HARRIS:
```C++
double detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    
    double t = (double)cv::getTickCount();

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Look for prominent corners and instantiate keypoints
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Hariss detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    return (1000 * t / 1.0);
```
---
All of the other detectors are implemented inside ```detKeypointsModern()``` function.
### FAST:
```C++
int threshold = 30;    // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;      // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
```
---
### BRISK:
```C++
int threshold = 30;        //   AGAST detection threshold score
        int octaves = 3;           // detection octaves
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint
        detector = cv::BRISK::create(threshold, octaves, patternScale);
```
---
### ORB:
```C++
 int   nfeatures = 500;     // The maximum number of features to retain.
        float scaleFactor = 1.2f;  // Pyramid decimation ratio, greater than 1.
        int   nlevels = 8;         // The number of pyramid levels.
        int   edgeThreshold = 31;  // This is size of the border where the features are not detected.
        int   firstLevel = 0;      // The level of pyramid to put source image to.
        int   WTA_K = 2;           // The number of points that produce each element of the oriented BRIEF descriptor.
        auto  scoreType = cv::ORB::HARRIS_SCORE; // HARRIS_SCORE / FAST_SCORE algorithm is used to rank features.
        int   patchSize = 31;      // Size of the patch used by the oriented BRIEF descriptor.
        int   fastThreshold = 20;  // The FAST threshold.
        detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                               firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
``` 

---
### AKAZE:
```C++
// Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT,
        //                                   DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
        auto  descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        int   descriptor_size = 0;        // Size of the descriptor in bits. 0 -> Full size
        int   descriptor_channels = 3;    // Number of channels in the descriptor (1, 2, 3).
        float threshold = 0.001f;         //   Detector response threshold to accept point.
        int   nOctaves = 4;               // Maximum octave evolution of the image.
        int   nOctaveLayers = 4;          // Default number of sublevels per scale level.
        auto  diffusivity = cv::KAZE::DIFF_PM_G2; // Diffusivity type. DIFF_PM_G1, DIFF_PM_G2,
        //                   DIFF_WEICKERT or DIFF_CHARBONNIER.
        detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                 threshold, nOctaves, nOctaveLayers, diffusivity);

```
---
### SIFT:
```C++
 int nfeatures = 0; // The number of best features to retain.
        int nOctaveLayers = 3; // The number of layers in each octave. 3 is the value used in D. Lowe paper.
        // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
        double contrastThreshold = 0.04;
        double edgeThreshold = 10; // The threshold used to filter out edge-like features.
        double sigma = 1.6; // The sigma of the Gaussian applied to the input image at the octave \#0.

        detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
}

```

## MP.3 Keypoint Removal
<font size="3">
Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.

The following code achieves the above mentioned functionality:</font>

```C++

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
```


## MP.4 Keypoint Descriptors
<font size="3">
Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.
</font>

All of the descriptors are implemented inside ```descKeypoints()``` function.

```C++
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        int bytes = 32;               // legth of the descriptor in bytes
        bool use_orientation = false; // sample patterns using key points orientation

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        int   nfeatures = 500;     // The maximum number of features to retain.
        float scaleFactor = 1.2f;  // Pyramid decimation ratio, greater than 1.
        int   nlevels = 8;         // The number of pyramid levels.
        int   edgeThreshold = 31;  // This is size of the border where the features are not detected.
        int   firstLevel = 0;      // The level of pyramid to put source image to.
        int   WTA_K = 2;           // The number of points that produce each element of the oriented BRIEF descriptor.
        auto  scoreType = cv::ORB::HARRIS_SCORE; // HARRIS_SCORE / FAST_SCORE algorithm is used to rank features.
        int   patchSize = 31;      // Size of the patch used by the oriented BRIEF descriptor.
        int   fastThreshold = 20;  // The FAST threshold.

        extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        bool orientationNormalized = true; // Enable orientation normalization.
        bool scaleNormalized = true;       // Enable scale normalization.
        float patternScale = 22.0f;        // Scaling of the description pattern.
        int nOctaves = 4;                  // Number of octaves covered by the detected keypoints.
        const std::vector<int>& selectedPairs = std::vector<int>(); // (Optional) user defined selected pairs indexes.

        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale,
                                               nOctaves, selectedPairs);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        // Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT,
        //                                   DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
        auto  descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        int   descriptor_size = 0;        // Size of the descriptor in bits. 0 -> Full size
        int   descriptor_channels = 3;    // Number of channels in the descriptor (1, 2, 3).
        float threshold = 0.001f;         //   Detector response threshold to accept point.
        int   nOctaves = 4;               // Maximum octave evolution of the image.
        int   nOctaveLayers = 4;          // Default number of sublevels per scale level.
        auto  diffusivity = cv::KAZE::DIFF_PM_G2; // Diffusivity type. DIFF_PM_G1, DIFF_PM_G2,
        //                   DIFF_WEICKERT or DIFF_CHARBONNIER.
        extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                  threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        int nfeatures = 0; // The number of best features to retain.
        int nOctaveLayers = 3; // The number of layers in each octave. 3 is the value used in D. Lowe paper.
        // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
        double contrastThreshold = 0.04;
        double edgeThreshold = 10; // The threshold used to filter out edge-like features.
        double sigma = 1.6; // The sigma of the Gaussian applied to the input image at the octave \#0.

        extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
    else
    {
        throw std::invalid_argument("Unknown descriptor type: " + descriptorType);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    return (1000 * t / 1.0);
}
```

## MP.5 Descriptor Matching
<font size="3">
Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.
    </font>

## MP.6 Descriptor Distance Ratio
<font size="3">
Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.</font>'

---
### FLANN matching
```C++
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround :
        //     convert binary descriptors to floating point due to a bug in current OpenCV implementation
        descSource.convertTo(descSource, CV_32F);
        }

        if (descRef.type() != CV_32F)
        {
        descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::cout<<"FLANN Matching" <<endl;
    }
```

---
### k-nearest neighbor selection
<font size="3">
Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints </font>

```C++
else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        int k = 2;
        vector<vector<cv::DMatch>> knn_matches;

        matcher->knnMatch(descSource, descRef, knn_matches, k);  // finds the 2 best matches

        std::cout << " (KNN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        // Filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        // std::cout << " KNN # keypoints removed = " << knn_matches.size() - matches.size() << endl;
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    }

    return (1000 * t / 1.0);
```

## MP.7 Performance Evaluation 1


[//]: # (Image References)

[image1_1]: ../images/SHI-TOMASI.png "SHI-TOMASI"


<font size="3">
Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.</font>



| Detector        | # of Key Points   | 
|:-------------:|:-------------:| 
| SHITOMASI     | 1179        | 
| HARRIS      | 248      |
| SIFT     | 1386      |
| BRISK      | 2762       |
| FAST      | 1491       |
| ORB      | 1161      |
| AKAZE      | 1670      |

### Neighborhood Size

#### SHITOMASI
<figure>
    <img  src="../images/SHI-TOMASI.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### HARRIS
<figure>
    <img  src="../images/HARRIS.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### SIFT
<figure>
    <img  src="../images/SIFT.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### FAST
<figure>
    <img  src="../images/FAST.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### BRISK
<figure>
    <img  src="../images/BRISK.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### ORG
<figure>
    <img  src="../images/ORB.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### AKAZE
<figure>
    <img  src="../images/AKAZE.png" alt="Drawing" style="width: 1000px;"/>
</figure>

## MP.8 Performance
<font size="3">
Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.
    </font>
    
    
   |  Mached Kpts  |SIFT       | ORB      | FREAK    |    BRISK  | BRIEF    |     AKAZE |
   |:-------:|:---------:|:--------:|:--------:|:---------:|:--------:|:---------:|  
   |SHITOMASI| 927       | 908      | 768      | 767       | 944      |  N/A   |
   |HARRIS| 163       | 162      | 144      | 142       | 173      |  N/A   |
   |SIFT| 800       | N/A      | 593      | 592       | 702      |  N/A   |
   |FAST| 1046       | 1071      | 878      | 899       | 1099      |  N/A   |
   |ORB| 763       | 763      | 420      | 751       | 545      |  N/A   |
   |BRISK| 1646       | 1514      | 1524      | 1570       | 1704      |  N/A   |
   |AKAZE| 1270       | 1182      | 1187      | 1215       | 1266      |  1259   |

## MP.9 Performance

<font size="3">
Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.</font>
    
All the results are stored and hig;ighted in yellow inside [evaluations](../evaluations/) folder.

The TOP3 detector / descriptor combinations:

 1) <font color=gree>**FAST** and **BRIEF** </font>

 2) <font color=gree>**FAST** and **ORB** </font>

 3) <font color=gree>**FAST** and **BRISK** </font>
