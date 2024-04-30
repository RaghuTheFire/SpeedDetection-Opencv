#include <opencv2/opencv.hpp>
#include <cmath>
#include <ctime>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>

const int limit = 80; // km/hr

const std::string traffic_record_folder_name = "datasave";

void createFolders() 
{
  const std::string exceeded_folder = traffic_record_folder_name + "/exceeded";
  std::filesystem::create_directory(traffic_record_folder_name);
  std::filesystem::create_directory(exceeded_folder);
}

class EuclideanDistTracker 
{
  private: std::unordered_map < int,
  std::pair < int,
  int >> center_points;
  int id_count = 0;
  std::vector < double > s1(1000, 0.0);
  std::vector < double > s2(1000, 0.0);
  std::vector < double > s(1000, 0.0);
  std::vector < int > f(1000, 0);
  std::vector < int > capf(1000, 0);
  int count = 0;
  int exceeded = 0;

  public: EuclideanDistTracker() 
  {
      
  }

  std::vector < std::vector < int >> update(const std::vector < cv::Rect > & objects_rect) 
  {
    std::vector < std::vector < int >> objects_bbs_ids;

    for (const auto & rect: objects_rect) 
    {
      int x = rect.x, y = rect.y, w = rect.width, h = rect.height;
      int cx = (x + x + w) / 2;
      int cy = (y + y + h) / 2;

      bool same_object_detected = false;
      for (const auto & [id, pt]: center_points) {
        double dist = std::hypot(cx - pt.first, cy - pt.second);
        if (dist < 70) {
          center_points[id] = std::make_pair(cx, cy);
          objects_bbs_ids.emplace_back(std::vector < int > {
            x,
            y,
            w,
            h,
            id
          });
          same_object_detected = true;

          if (y >= 410 && y <= 430) 
          {
            s1[id] = static_cast < double > (cv::getTickCount());
          }

          if (y >= 235 && y <= 255) 
          {
            s2[id] = static_cast < double > (cv::getTickCount());
            s[id] = (s2[id] - s1[id]) / cv::getTickFrequency();
          }

          if (y < 235) 
          {
            f[id] = 1;
          }
        }
      }

      if (!same_object_detected) 
      {
        center_points[id_count] = std::make_pair(cx, cy);
        objects_bbs_ids.emplace_back(std::vector < int > {
          x,
          y,
          w,
          h,
          id_count
        });
        id_count++;
        s[id_count] = 0.0;
        s1[id_count] = 0.0;
        s2[id_count] = 0.0;
      }
    }

    std::unordered_map < int, std::pair < int, int >> new_center_points;
    for (const auto & obj_bb_id: objects_bbs_ids) {
      int object_id = obj_bb_id.back();
      new_center_points[object_id] = center_points[object_id];
    }
    center_points = std::move(new_center_points);

    return objects_bbs_ids;
  }

  int getsp(int id) 
  {
    if (s[id] != 0.0) 
    {
      return static_cast < int > (214.15 / s[id]);
    }
    return 0;
  }

  void capture(const cv::Mat & img, int x, int y, int h, int w, int sp, int id) 
  {
    if (capf[id] == 0) 
    {
      capf[id] = 1;
      f[id] = 0;
      cv::Rect roi(x - 5, y - 5, w + 10, h + 10);
      cv::Mat crop_img = img(roi);
      std::string filename = traffic_record_folder_name + "/" + std::to_string(id) + "_speed_" + std::to_string(sp) + ".jpg";
      cv::imwrite(filename, crop_img);
      count++;

      std::ofstream file(speed_record_file_location, std::ios_base::app);
      if (sp > limit) {
        std::string exceeded_filename = traffic_record_folder_name + "/exceeded/" + std::to_string(id) + "_speed_" + std::to_string(sp) + ".jpg";
        cv::imwrite(exceeded_filename, crop_img);
        file << id << " \t " << sp << "<---exceeded\n";
        exceeded++;
      } 
      else 
      {
        file << id << " \t " << sp << "\n";
      }
    }
  }

  int getLimit() const 
  {
    return limit;
  }

  void end() 
  {
    std::ofstream file(speed_record_file_location, std::ios_base::app);
    file << "\n-------------\n";
    file << "-------------\n";
    file << "SUMMARY\n";
    file << "-------------\n";
    file << "Total Vehicles :\t" << count << "\n";
    file << "Exceeded speed limit :\t" << exceeded << "\n";
  }
};


int main() 
{
  VideoCapture cap("demo.mp4");
  if (!cap.isOpened()) 
  {
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  EuclideanDistTracker tracker;

  Ptr < BackgroundSubtractorMOG2 > object_detector = createBackgroundSubtractorMOG2();
  Ptr < BackgroundSubtractorMOG2 > fgbg = createBackgroundSubtractorMOG2(true);

  Mat kernel_op = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
  Mat kernel_op2 = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
  Mat kernel_cl = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
  Mat kernel_e = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));

  int f = 25;
  int w = int(1000 / (f - 1));

  while (true) 
  {
    Mat frame;
    cap >> frame;
    if (frame.empty()) 
    {
      break;
    }

    resize(frame, frame, Size(), 0.5, 0.5);
    int height = frame.rows;
    int width = frame.cols;

    Mat roi = frame(Rect(200, 50, 760, 490));

    Mat mask;
    object_detector -> apply(roi, mask);
    threshold(mask, mask, 250, 255, THRESH_BINARY);

    Mat fgmask;
    fgbg -> apply(roi, fgmask);
    Mat imBin;
    threshold(fgmask, imBin, 200, 255, THRESH_BINARY);
    Mat mask1;
    morphologyEx(imBin, mask1, MORPH_OPEN, kernel_op);
    Mat mask2;
    morphologyEx(mask1, mask2, MORPH_CLOSE, kernel_cl);
    Mat e_img;
    erode(mask2, e_img, kernel_e);

    vector < vector < float >> detections;
    vector < vector < Point >> contours;
    findContours(e_img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) 
    {
      double area = contourArea(contours[i]);
      if (area > 1000) 
      {
        Rect rect = boundingRect(contours[i]);
        detections.push_back({
          static_cast < float > (rect.x),
          static_cast < float > (rect.y),
          static_cast < float > (rect.width),
          static_cast < float > (rect.height)
        });
        rectangle(roi, rect, Scalar(0, 255, 0), 3);
      }
    }

    vector < vector < float >> boxes_ids = tracker.update(detections);
    for (size_t i = 0; i < boxes_ids.size(); i++) 
    {
      int x = static_cast < int > (boxes_ids[i][0]);
      int y = static_cast < int > (boxes_ids[i][1]);
      int w = static_cast < int > (boxes_ids[i][2]);
      int h = static_cast < int > (boxes_ids[i][3]);
      int id = static_cast < int > (boxes_ids[i][4]);

      if (tracker.getsp(id) < tracker.limit()) 
      {
        putText(roi, to_string(id) + " " + to_string(tracker.getsp(id)), Point(x, y - 15), FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 0), 2);
        rectangle(roi, Rect(x, y, w, h), Scalar(0, 255, 0), 3);
      } 
      else 
      {
        putText(roi, to_string(id) + " " + to_string(tracker.getsp(id)), Point(x, y - 15), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255), 2);
        rectangle(roi, Rect(x, y, w, h), Scalar(0, 165, 255), 3);
      }

      int s = tracker.getsp(id);
      if (s != 0) 
      {
        tracker.capture(roi, x, y, h, w, s, id);
      }
    }

    line(roi, Point(0, 410), Point(960, 410), Scalar(0, 0, 255), 2);
    line(roi, Point(0, 430), Point(960, 430), Scalar(0, 0, 255), 2);
    line(roi, Point(0, 235), Point(960, 235), Scalar(0, 0, 255), 2);
    line(roi, Point(0, 255), Point(960, 255), Scalar(0, 0, 255), 2);

    imshow("Mask", mask2);
    imshow("Erode", e_img);
    imshow("ROI", roi);

    int key = waitKey(w - 10);
    if (key == 27) 
    {
      tracker.end();
      break;
    }
  }

  cap.release();
  destroyAllWindows();

  return 0;
}
