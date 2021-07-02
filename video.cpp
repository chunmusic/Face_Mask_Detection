#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <ctime>

#include <string.h>

cv::dnn::Net net;
const char* classNames[] = {"with_mask","without_mask","mask_weared_incorrect"};

void initialize(std::string pbtxt = "graph.pbtxt",
                std::string model = "output/export_saved_model/frozen_inference_graph.pb")
{
  
  net = cv::dnn::readNetFromTensorflow(model,pbtxt);

  // Uncomment in case of having cuda
  // net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  // net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

}

cv::Mat detect_object(cv::Mat img){

  cv::Mat blob = cv::dnn::blobFromImage(img,1.0,cv::Size(300,300),cv::Scalar(0,0,0),true,false);
  net.setInput(blob);

  cv::Mat outputs = net.forward();
  cv::Mat detection(outputs.size[2], outputs.size[3], CV_32F, outputs.ptr<float>());

	for (int i = 0; i < detection.rows; i++)
	{
		float confidence = detection.at<float>(i, 2);

		if (confidence > 0.5)
		{	
			size_t objectClass = (size_t)(detection.at<float>(i, 1));
			int xLeftBottom = static_cast<int>(detection.at<float>(i, 3) * img.cols);
			int yLeftBottom = static_cast<int>(detection.at<float>(i, 4) * img.rows);
			int xRightTop = static_cast<int>(detection.at<float>(i, 5) * img.cols);
			int yRightTop = static_cast<int>(detection.at<float>(i, 6) * img.rows);

			std::ostringstream ss;
			ss << round(confidence*100);
			std::string conf(ss.str());

			cv::Rect object((int)xLeftBottom, (int)yLeftBottom,(int)(xRightTop - xLeftBottom),(int)(yRightTop - yLeftBottom));

			cv::rectangle(img, object, cv::Scalar(0, 255, 0), 2);
			std::string label = std::string(classNames[objectClass-1]) + ": " + conf + " %";
			int baseLine = 0;

			cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		  cv::rectangle(img, cv::Rect(cv::Point(xLeftBottom, yLeftBottom - labelSize.height),cv::Size(labelSize.width, labelSize.height + baseLine)),cv::Scalar(0, 255, 0), cv::FILLED);
			cv::putText(img, label, cv::Point(xLeftBottom, yLeftBottom),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}

  return img;
}


int main( int argc, char** argv ) {

  long frameCounter = 0;
  long fps_buffer = 0;

  std::time_t timeBegin = std::time(0);
  int tick = 0;

  // Initial setting for camera
  cv::Mat frame;
  cv::VideoCapture cap;
  
  int deviceID = 0;
  int apiID = cv::CAP_ANY;

  cap.open(deviceID,apiID);
  cap.set(cv::CAP_PROP_FPS,30);
  
  if(!cap.isOpened()){
    std::cerr << "ERROR! Unable to open camera \n" << std::endl;
    return -1;
  }

  std::cout << "Stat grabbing " << std::endl;

  // Initialize classes and import model
  initialize();


  // Loop for collecting camera frame
  for (;;)
  {

    cap.read(frame);

    if (frame.empty()){
      break;
    }

    frame = detect_object(frame);

    // Calculate FPS
    frameCounter++;
    std::time_t timeNow = std::time(0) - timeBegin;

    if (timeNow - tick >= 1)
    {
      tick++;
      fps_buffer = frameCounter;
      frameCounter = 0;
    }

    cv::putText(frame,"FPS: " + std::to_string(fps_buffer),cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);


    // Show result image
    cv::imshow("img",frame);

    if (cv::waitKey(1) >= 0){
      break;
    }
  }

  return 0;
}

//TODO: add NMS

//compile
//  g++ video.cpp -o video `pkg-config --cflags --libs opencv4`

//run
//  ./output