// Importing Libraries
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/correlation_tracker.h>
#include <cmath>
#include <iostream>

// Classifier File
cv::CascadeClassifier carCascade("vech.xml");

// Video file capture
cv::VideoCapture video("carsVideo.mp4");

// Constant Declaration
const int WIDTH = 1280;
const int HEIGHT = 720;

// Estimate speed function
double estimateSpeed(std::vector < int > location1, std::vector < int > location2) 
{
  double d_pixels = std::sqrt(std::pow(location2[0] - location1[0], 2) + std::pow(location2[1] - location1[1], 2));
  double ppm = 8.8;
  double d_meters = d_pixels / ppm;
  int fps = 18;
  double speed = d_meters * fps * 3.6;
  return speed;
}

// Tracking multiple objects
void trackMultipleObjects() 
{
  cv::Scalar rectangleColor(0, 255, 255);
  int frameCounter = 0;
  int currentCarID = 0;
  double fps = 0;

  std::map < int, dlib::correlation_tracker > carTracker;
  std::map < int, std::vector < int >> carLocation1;
  std::map < int, std::vector < int >> carLocation2;
  std::vector < double > speed(1000, 0.0);

  cv::VideoWriter out("outTraffic.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(WIDTH, HEIGHT));

  while (true) 
  {
    double start_time = static_cast < double > (cv::getTickCount());
    cv::Mat image;
    video >> image;
    if (image.empty()) 
    {
      break;
    }

    cv::resize(image, image, cv::Size(WIDTH, HEIGHT));
    cv::Mat resultImage = image.clone();

    frameCounter++;
    std::vector < int > carIDtoDelete;

    for (auto & pair: carTracker) 
    {
      int carID = pair.first;
      dlib::correlation_tracker & tracker = pair.second;
      dlib::rectangle rect = tracker.get_position();
      cv::Rect trackedRect(rect.left(), rect.top(), rect.width(), rect.height());
      tracker.update(image, rect);

      if (rect.area() < 7) {
        carIDtoDelete.push_back(carID);
      }
    }

    for (int carID: carIDtoDelete) 
    {
      std::cout << "Removing carID " << carID << " from list of trackers." << std::endl;
      std::cout << "Removing carID " << carID << " previous location." << std::endl;
      std::cout << "Removing carID " << carID << " current location." << std::endl;
      carTracker.erase(carID);
      carLocation1.erase(carID);
      carLocation2.erase(carID);
    }

    if (frameCounter % 10 == 0) 
    {
      cv::Mat gray;
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
      std::vector < cv::Rect > cars;
      carCascade.detectMultiScale(gray, cars, 1.1, 13, 18, cv::Size(24, 24));

      for (const cv::Rect & rect: cars) 
      {
        int x = rect.x;
        int y = rect.y;
        int w = rect.width;
        int h = rect.height;

        double x_bar = x + 0.5 * w;
        double y_bar = y + 0.5 * h;

        int matchCarID = -1;

        for (auto & pair: carTracker) 
        {
          int carID = pair.first;
          dlib::correlation_tracker & tracker = pair.second;
          dlib::rectangle trackedRect = tracker.get_position();

          int t_x = trackedRect.left();
          int t_y = trackedRect.top();
          int t_w = trackedRect.width();
          int t_h = trackedRect.height();

          double t_x_bar = t_x + 0.5 * t_w;
          double t_y_bar = t_y + 0.5 * t_h;

          if ((t_x <= x_bar && x_bar <= (t_x + t_w)) && (t_y <= y_bar && y_bar <= (t_y + t_h)) &&
            (x <= t_x_bar && t_x_bar <= (x + w)) && (y <= t_y_bar && t_y_bar <= (y + h))) {
            matchCarID = carID;
            break;
          }
        }

        if (matchCarID == -1) 
        {
          std::cout << "Creating new tracker " << currentCarID << std::endl;

          dlib::correlation_tracker tracker = dlib::correlation_tracker();
          dlib::rectangle dlib_rect(x, y, x + w, y + h);
          tracker.start_track(image, dlib_rect);

          carTracker[currentCarID] = tracker;
          carLocation1[currentCarID] = {
            x,
            y,
            w,
            h
          };

          currentCarID++;
        }
      }
    }

    for (auto & pair: carTracker) 
    {
      int carID = pair.first;
      dlib::correlation_tracker & tracker = pair.second;
      dlib::rectangle trackedRect = tracker.get_position();

      int t_x = trackedRect.left();
      int t_y = trackedRect.top();
      int t_w = trackedRect.width();
      int t_h = trackedRect.height();

      cv::rectangle(resultImage, cv::Rect(t_x, t_y, t_w, t_h), rectangleColor, 4);

      carLocation2[carID] = {
        t_x,
        t_y,
        t_w,
        t_h
      };
    }

    double end_time = static_cast < double > (cv::getTickCount());
    fps = cv::getTickFrequency() / (end_time - start_time);

    for (auto & pair: carLocation1) 
    {
      int carID = pair.first;
      if (frameCounter % 1 == 0) 
      {
        std::vector < int > & x1y1w1h1 = carLocation1[carID];
        std::vector < int > & x2y2w2h2 = carLocation2[carID];

        carLocation1[carID] = x2y2w2h2;

        if (x1y1w1h1 != x2y2w2h2) 
        {
          if ((speed[carID] == 0) && y1 >= 275 && y1 <= 285) {
            speed[carID] = estimateSpeed(x1y1w1h1, x2y2w2h2);
          }

          if (speed[carID] != 0 && y1 >= 180) 
          {
            cv::putText(resultImage, std::to_string(static_cast < int > (speed[carID])) + "km/h",
              cv::Point(x1 + w1 / 2, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 100), 2);
          }
        }
      }
    }

    cv::imshow("result", resultImage);
    out.write(resultImage);

    if (cv::waitKey(1) == 27) 
    {

      break;
    }
  }

  cv::destroyAllWindows();
  out.release();
}

int main() {
  trackMultipleObjects();
  return 0;
}
