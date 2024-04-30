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
