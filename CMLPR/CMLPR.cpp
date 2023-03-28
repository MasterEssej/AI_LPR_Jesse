// CMLPR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <baseapi.h>
#include <allheaders.h>

using namespace cv;
using namespace std;

#pragma region functions
Mat RGB2Grey(Mat RGB)
{
    Mat grey = Mat::zeros(RGB.size(), CV_8UC1); // Matrix of zeroes (black image), size of input image, 8 bit unsigned 1 channel (grey scale)

    for (int i = 0; i < RGB.rows; i++)
    {
        for (int j = 0; j < RGB.cols * 3; j += 3) // Cols*3 and jump 3 every time since RGB has 3 values for each pixel
        {
            grey.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2)) / 3; // j/3 in first to make it correct channel for grey scale, take average of RGB values to get 1 value for grey scale
        }
    }

    return grey;
}

Mat Grey2Binary(Mat Grey, int threshold)
{
    Mat binary = Mat::zeros(Grey.size(), CV_8UC1);

    for (int i = 0; i < Grey.rows; i++)
    {
        for (int j = 0; j < Grey.cols; j++)
        {
            if (Grey.at<uchar>(i, j) > threshold)
            {
                binary.at<uchar>(i, j) = 255;
            }
        }
    }

    return binary;
}

Mat InverseGrey(Mat Grey)
{
    Mat inverted = Mat::zeros(Grey.size(), CV_8UC1);

    for (int i = 0; i < Grey.rows; i++)
    {
        for (int j = 0; j < Grey.cols; j++)
        {
            inverted.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);
        }
    }

    return inverted;
}

Mat stepFunction(Mat Grey, int th1, int th2)
{
    Mat stepped = Mat::zeros(Grey.size(), CV_8UC1);

    for (int i = 0; i < Grey.rows; i++)
    {
        for (int j = 0; j < Grey.cols; j++)
        {
            if (Grey.at<uchar>(i, j) >= th1 && Grey.at<uchar>(i, j) <= th2)
            {
                stepped.at<uchar>(i, j) = 255;
            }
        }
    }

    return stepped;
}

Mat AvgMask(Mat Grey, int neighborsize) // Output average, exclude border
{
    Mat AvgImg = Mat::zeros(Grey.size(), CV_8UC1);

    int totPixels = pow((neighborsize * 2) + 1, 2);
    for (int i = neighborsize; i < Grey.rows - neighborsize; i++)
    {
        for (int j = neighborsize; j < Grey.cols - neighborsize; j++)
        {
            int sum = 0;
            int count = 0;

            for (int ii = - neighborsize; ii <= neighborsize; ii++)
            {
                for (int jj = - neighborsize; jj <= neighborsize; jj++)
                {
                    sum += Grey.at<uchar>(i + ii, j + jj);
                    count++;
                }
            }

            AvgImg.at<uchar>(i, j) = sum / count;
        }
    }

    return AvgImg;
}

Mat MaxMask(Mat Grey, int neighborsize) // Output max, exclude border
{
    Mat MaxImg = Mat::zeros(Grey.size(), CV_8UC1);

    for (int i = neighborsize; i < Grey.rows - neighborsize; i++)
    {
        for (int j = neighborsize; j < Grey.cols - neighborsize; j++)
        {
            int max = 0;

            for (int ii = -neighborsize; ii <= neighborsize; ii++)
            {
                for (int jj = -neighborsize; jj <= neighborsize; jj++)
                {
                    if (Grey.at<uchar>(i + ii, j + jj) > max)
                    {
                        max = Grey.at<uchar>(i + ii, j + jj);
                    }
                }
            }

            MaxImg.at<uchar>(i, j) = max;
        }
    }

    return MaxImg;
}

Mat MinMask(Mat Grey, int neighborsize) // Output min, exclude border
{
    Mat MinImg = Mat::zeros(Grey.size(), CV_8UC1);

    for (int i = neighborsize; i < Grey.rows - neighborsize; i++)
    {
        for (int j = neighborsize; j < Grey.cols - neighborsize; j++)
        {
            int min = 255;

            for (int ii = -neighborsize; ii <= neighborsize; ii++)
            {
                for (int jj = -neighborsize; jj <= neighborsize; jj++)
                {
                    if (Grey.at<uchar>(i + ii, j + jj) < min)
                    {
                        min = Grey.at<uchar>(i + ii, j + jj);
                    }
                }
            }

            MinImg.at<uchar>(i, j) = min;
        }
    }

    return MinImg;
}

Mat Edge(Mat Grey, int th)
{
    Mat EdgeImg = Mat::zeros(Grey.size(), CV_8UC1);

    for (int i = 1; i < Grey.rows - 1; i++)
    {
        for (int j = 1; j < Grey.cols - 1; j++)
        {
            int leftavg = (Grey.at<uchar>(i - 1, j - 1) + Grey.at<uchar>(i, j - 1) + Grey.at<uchar>(i + 1, j - 1)) / 3;
            int rightavg = (Grey.at<uchar>(i - 1, j + 1) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i + 1, j + 1)) / 3;
            if (abs(leftavg - rightavg) > th)
            {
                EdgeImg.at<uchar>(i, j) = 255;
            }
        }
    }
    return EdgeImg;
}

Mat Dilation(Mat EdgeImg, int n)
{
    Mat dilated = Mat::zeros(EdgeImg.size(), CV_8UC1);

    for (int i = n; i < EdgeImg.rows - n; i++)
    {
        for (int j = n; j < EdgeImg.cols - n; j++)
        {
            
            if (EdgeImg.at<uchar>(i, j) == 0)
            {

                for (int ii = -n; ii <= n; ii++)
                {
                    for (int jj = -n; jj <= n; jj++)
                    {
                        if (EdgeImg.at<uchar>(i + ii, j + jj) == 255)
                        {
                            dilated.at<uchar>(i, j) = 255;
                            break;
                        }
                    }
                }
            }
            else if (EdgeImg.at<uchar>(i, j) == 255)
            {
                dilated.at<uchar>(i, j) = 255;
            }
            
        }
    }
    return dilated;
}

Mat Erosion(Mat EdgeImg, int n)
{
    Mat eroded = Mat::zeros(EdgeImg.size(), CV_8UC1);

    for (int i = n; i < EdgeImg.rows - n; i++)
    {
        for (int j = n; j < EdgeImg.cols - n; j++)
        {

            if (EdgeImg.at<uchar>(i, j) == 255)
            {
                eroded.at<uchar>(i, j) = 255;
                for (int ii = -n; ii <= n; ii++)
                {
                    for (int jj = -n; jj <= n; jj++)
                    {
                        if (EdgeImg.at<uchar>(i + ii, j + jj) == 0)
                        {
                            eroded.at<uchar>(i, j) = 0;
                            break;
                        }
                    }
                }
            }
        }
    }
    return eroded;
}

Mat EqHist(Mat Grey)
{
    Mat equalized = Mat::zeros(Grey.size(), CV_8UC1);

    //count
    int count[256] = { 0 };
    for (int i = 0; i < Grey.rows; i++)
    {
        for (int j = 0; j < Grey.cols; j++)
        {
            count[Grey.at<uchar>(i, j)]++;
        }
    }

    //prob
    float prob[256] = { 0.0 };
    for (int i = 0; i < 256; i++)
    {
        prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);
    }

    //accprob
    float accprob[256] = { 0.0 };
    accprob[0] = prob[0];
    for (int i = 1; i < 256; i++)
    {
        accprob[i] = prob[i] + accprob[i - 1];
    }

    // new == 255 * accprob
    int newvalue[256] = { 0 };
    for (int i = 0; i < 256; i++)
    {
        newvalue[i] = 255 * accprob[i];
    }

    for (int i = 0; i < Grey.rows; i++)
    {
        for (int j = 0; j < Grey.cols; j++)
        {
            equalized.at<uchar>(i, j) = newvalue[Grey.at<uchar>(i, j)];
        }
    }

    return equalized;
}
int OTSU(Mat Grey)
{

    //count
    int count[256] = { 0 };
    for (int i = 0; i < Grey.rows; i++)
    {
        for (int j = 0; j < Grey.cols; j++)
        {
            count[Grey.at<uchar>(i, j)]++;
        }
    }

    //prob
    float prob[256] = { 0.0 };
    for (int i = 0; i < 256; i++)
    {
        prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);
    }

    //accprob
    float theta[256] = { 0.0 };
    theta[0] = prob[0];
    for (int i = 1; i < 256; i++)
    {
        theta[i] = prob[i] + theta[i - 1];
    }

    //meu
    float meu[256] = { 0.0 };
    for (int i = 1; i < 256; i++)
    {
        meu[i] = i*prob[i] + meu[i - 1];
    }


    float sigma[256] = { 0.0 };
    for (int i = 0; i < 256; i++)
    {
        sigma[i] = pow(meu[255] * theta[i] - meu[i], 2) / (theta[i]*(1 - theta[i]));
    }

    int index = 0;
    float maxVal = 0;

    for (int i = 0; i < 256; i++)
    {
        if (sigma[i] > maxVal)
        {
            maxVal = sigma[i];
            index = i;
        }
    }

    return index;
}

bool BP(Mat image, int th)
{
    float b = 0;
    float bp;

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (image.at<uchar>(i, j) == 0)
            {
                b++;
            }
        }
    }

    bp = (b / (float)(image.rows * image.cols)) * 100;

    if (bp > th)
    {
        return true;
    }
    else
    {
        return false;
    }
}

Mat padImage(Mat Grey, int padn, int color)
{
    Size size(Grey.cols + padn*2, Grey.rows + padn*2);
    Mat padded = Mat::zeros(size, CV_8UC1);
    

    for (int i = 0; i < padded.rows; i++)
    {
        for (int j = 0; j < padn; j++)
        {
            padded.at<uchar>(i, j) = color;
            padded.at<uchar>(i, padded.cols - j - 1) = color;
        }
    }
    for (int j = 0; j < padded.cols; j++)
    {
        for (int i = 0; i < padn; i++)
        {
            padded.at<uchar>(i, j) = color;
            padded.at<uchar>(padded.rows - i - 1, j) = color;
        }
    }
    for (int i = padn, x = 0; i < Grey.rows + padn; i++, x++)
    {
        for (int j = padn, y = 0; j < Grey.cols + padn; j++, y++)
        {
            padded.at<uchar>(i, j) = Grey.at<uchar>(x, y);
        }
    }
    return padded;
}

#pragma endregion



int main()
{
    /*
    tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();

    if (api->Init("C:\\Users\\jesse.viitanen\\AppData\\Local\\Tesseract-OCR\\tessdata", "eng")) {
        fprintf(stderr,"Could not initialize tesseract.\n");
    exit(1);}*/

    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI;
    api->Init("C:\\Users\\jesse.viitanen\\AppData\\Local\\Tesseract-OCR\\tessdata", "eng", tesseract::OEM_DEFAULT);
    api->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
    api->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890");
    //api->SetVariable("user_defined_dpi", "71");
    //api->SetVariable("debug_file", "/dev/null");

    vector<cv::String> fn;
    glob("C:\\Users\\jesse.viitanen\\Pictures\\Dataset\\*.jpg", fn, false);

    vector<Mat> images;
    for (int i = 0; i < 20; i++)
    {
        images.push_back(imread(fn[i]));
    }

    vector<Mat> plates(images.size());

    vector<int> skip{2, 3, 5, 7, 8, 10, 12, 13, 14, 15, 17};
    
    for (int k = 0; k < images.size(); k++)
    {
        if (find(skip.begin(), skip.end(), k+1) != skip.end()) {continue;}

        Mat GreyImg = RGB2Grey(images[k]);
        //imshow("Grey image", GreyImg);

        Mat EQImg = EqHist(GreyImg);
        //imshow("Equalized Grey image", EQImg);

        Mat AvgImg = AvgMask(EQImg, 1);
        //imshow("Average Mask image (Blur)", AvgImg);

        Mat BlurEdgeImg = Edge(AvgImg, 50);
        //imshow("Blur Edge image", BlurEdgeImg);

        Mat DilatedImg = Dilation(BlurEdgeImg, 4);
        //imshow("Dilated image", DilatedImg);


        Mat DilatedImgCopy;
        DilatedImgCopy = DilatedImg.clone();
        vector<vector<Point>> contours1; // 2d vector. Number of segments and all points in each segment.
        vector<Vec4i> hierarchy1;
        findContours(DilatedImg, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
        Mat dst = Mat::zeros(GreyImg.size(), CV_8UC3);

        if (!contours1.empty())
        {
            for (int i = 0; i < contours1.size(); i++)
            {
                Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
                drawContours(dst, contours1, i, colour, -1, 8, hierarchy1);
            }
        }
        //imshow("Segmented Image", dst);


        Mat plate;
        Rect rect;
        Scalar black = CV_RGB(0, 0, 0);
        for (int i = 0; i < contours1.size(); i++)
        {
            rect = boundingRect(contours1[i]);
            float ratio = ((float)rect.width / (float)rect.height);

            if (rect.width < 40 || rect.height > 150 || rect.x < 0.2 * GreyImg.cols || rect.x > 0.8 * GreyImg.cols || rect.y < 0.1 * GreyImg.rows || rect.y > 0.9 * GreyImg.rows || ratio < 1.5 || BP(dst(rect), 35) == true)
            {
                drawContours(DilatedImgCopy, contours1, i, black, -1, 8, hierarchy1);
            }
            else
            {
                plates[k] = GreyImg(rect);
                //cout << endl << ratio << endl;
            }
        }

        //imshow("Filtered image", DilatedImgCopy);
        //cout << endl << k + 1 << endl; // Image number

        if (plates[k].rows != 0 || plates[k].cols != 0) //prevent crashing if plate is empty
        {
            imshow("Plate" + to_string(k + 1), plates[k]);
        }
        //waitKey();

        
        int OTSUTH = OTSU(plates[k]);

        Mat EqPlate = EqHist(plates[k]);
        //imshow("Equalized plate" + to_string(k+1), EqPlate);

        Mat BinPlate = Grey2Binary(EqPlate, OTSUTH+90);
        //imshow("Binarized plate" + to_string(k + 1), BinPlate);

        /*
        api->SetImage(BinPlate.data, BinPlate.rows, BinPlate.cols, 1, BinPlate.step);
        string outtext = std::string(api->GetUTF8Text());

        cout << outtext;*/


        //cout << endl << OTSUTH << endl;

        Mat BinPlateCopy;
        BinPlateCopy = BinPlate.clone();
        vector<vector<Point>> contours2; // 2d vector. Number of segments and all points in each segment.
        vector<Vec4i> hierarchy2;
        findContours(BinPlate, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
        Mat dst2 = Mat::zeros(plates[k].size(), CV_8UC3);

        if (!contours2.empty())
        {
            for (int i = 0; i < contours2.size(); i++)
            {
                Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
                drawContours(dst2, contours2, i, colour, -1, 8, hierarchy2);
            }
        }
        imshow("Segmented Plate" + to_string(k + 1), dst2);
        //waitKey();

        
        Size nsize(300, 400);
        Mat nChar;

        Mat Char;
        int c = 0;
        for (int i = 0; i < contours2.size(); i++)
        {
            rect = boundingRect(contours2[i]);


            if (rect.height < 5 || rect.width > BinPlate.cols/2)
            {
                drawContours(BinPlateCopy, contours2, i, black, -1, 8, hierarchy2);
            }
            else
            {
                
                Char = plates[k](rect);
                c++;
                nsize = Size(Char.cols*10, Char.rows*10);

                /**/
                resize(Char, Char, nsize);
                Char = Grey2Binary(Char, OTSU(Char) + 20);
                Char = InverseGrey(Char);
                Char = padImage(Char, 30, 255);
                Char = Dilation(Char, 1);

                /*
                //resize(Char, Char, nsize);
                Char = Grey2Binary(Char, OTSU(Char)+20);
                //Char = InverseGrey(Char);
                resize(Char, Char, nsize);
                Char = padImage(Char, 30, 0);
                //resize(Char, Char, nsize);
                Char = Grey2Binary(Char, OTSU(Char)+20);
                Char = InverseGrey(Char);*/

                imshow("Character" + to_string(c), Char);

                
                api->SetImage(Char.data, Char.rows, Char.cols, 1, Char.step);
                string outtext = std::string(api->GetUTF8Text());

                cout << to_string(c) << ". " << outtext << endl;

                //waitKey();
            }
            
        }
        cout << endl;
        imshow("Segmented Plate2" + to_string(k + 1), BinPlateCopy);
        /*
        api->SetImage(BinPlateCopy.data, BinPlateCopy.rows, BinPlateCopy.cols, 1, BinPlateCopy.step);
        string outtext = std::string(api->GetUTF8Text());

        cout << outtext;*/
        


        if (waitKey() == 'q') { break; }
    }

    //cv::waitKey();


    api->End();


    std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
