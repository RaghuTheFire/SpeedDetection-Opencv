#include <opencv2/opencv.hpp>
#include <ctime>

const std::string cascade_src = "cars1.xml";
const std::string video_src = "video3.MP4";

int ax1 = 70;
int ay = 90;
int ax2 = 230;
int bx1 = 15;
int by = 125;
int bx2 = 225;

double Speed_Cal(double time) 
{
  // Here i converted m to Km and second to hour then divison to reach Speed in this form (KM/H)
  // the 9.144 is distance of free space between two lines # found in https://news.osu.edu/slow-down----those-lines-on-the-road-are-longer-than-you-think/
  // i know that the 9.144 is an standard and my video may not be that but its like guess and its need Field research
  try 
  {
    double Speed = (9.144 * 3600) / (time * 1000);
    return Speed;
  } 
  catch (const std::exception & e) 
  {
    std::cout << 5 << std::endl;
    return 0.0;
  }
}

int main() 
{
  // car num
  int i = 1;
  double start_time = static_cast < double > (std::clock()) / CLOCKS_PER_SEC;

  // video ....
  cv::VideoCapture cap(video_src);
  cv::CascadeClassifier car_cascade(cascade_src);

  while (true) 
  {
    cv::Mat img;
    cap.read(img);
    if (img.empty()) 
    {
      break;
    }

    // bluring to have exacter detection
    cv::Mat blurred;
    cv::blur(img, blurred, cv::Size(15, 15));
    cv::Mat gray;
    cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);
    std::vector < cv::Rect > cars;
    car_cascade.detectMultiScale(gray, cars, 1.1, 2);

    // line a #i know road has got
    cv::line(img, cv::Point(ax1, ay), cv::Point(ax2, ay), cv::Scalar(255, 0, 0), 2);
    // line b
    cv::line(img, cv::Point(bx1, by), cv::Point(bx2, by), cv::Scalar(255, 0, 0), 2);

    for (const auto & car: cars) 
    {
      cv::rectangle(img, car, cv::Scalar(0, 0, 255), 2);
      cv::circle(img, cv::Point((car.x + car.x + car.width) / 2, (car.y + car.y + car.height) / 2), 1, cv::Scalar(0, 255, 0), -1);

      if (ay == (car.y + car.y + car.height) / 2) {
        start_time = static_cast < double > (std::clock()) / CLOCKS_PER_SEC;
        break;
      }

      if (ay <= (car.y + car.y + car.height) / 2) {
        if (by <= (car.y + car.y + car.height) / 2 && by + 10 >= (car.y + car.y + car.height) / 2) {
          cv::line(img, cv::Point(bx1, by), cv::Point(bx2, by), cv::Scalar(0, 255, 0), 2);
          double Speed = Speed_Cal(static_cast < double > (std::clock()) / CLOCKS_PER_SEC - start_time);
          std::cout << "Car Number " << i << " Speed: " << Speed << std::endl;
          i++;
          cv::putText(img, "Speed: " + std::to_string(Speed) + "KM/H", cv::Point(car.x, car.y - 15), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 3);
          break;
        } 
        else 
        {
          cv::putText(img, "Calcuting", cv::Point(100, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 3);
          break;
        }
      }
    }

    cv::imshow("video", img);

    if (cv::waitKey(33) == 27) 
    {
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();

  return 0;
}
