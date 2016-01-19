#include <iostream>
#include <cstdlib>
#include <random>
#include <limits>
#include <thread>
#include <future>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

struct Circle {
  Circle(const cv::Point& c, int r, const cv::Scalar& col) : center(c), radius(r), color(col) {}
  cv::Point center;
  int radius;
  cv::Scalar color;
};

using Circles = std::vector<Circle>;

void print_help() {
  std::cerr << "Usage: gc [input image] [output image]\n";
}

template<class T>
T clamp(const T&n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

void generate_end_evaluate_solution(
    Circles& circles,
    cv::Mat& image,
    cv::Mat& image_lab,
    double& distance,
    std::mt19937& rnd_engine,
    std::uniform_int_distribution<int>& generate_random_center_x,
    std::uniform_int_distribution<int>& generate_random_center_y,
    std::uniform_int_distribution<int>& generate_random_radius,
    std::uniform_int_distribution<>& generate_random_color_channel_value,
    std::discrete_distribution<>& generate_random_mutation_operation,
    std::uniform_int_distribution<>& generate_random_circle_mutation_operation,
    std::normal_distribution<>& generate_random_mutation_center_x,
    std::normal_distribution<>& generate_random_mutation_center_y,
    std::normal_distribution<>& generate_random_mutation_radius,
    std::normal_distribution<>& generate_random_mutation_color_channel_value,
    const cv::Mat& original_image_lab) {

  image = cv::Scalar();

  /** Mutate candidate */
  const int kMutationOperation = generate_random_mutation_operation(rnd_engine);

  if (circles.empty() || 2 == kMutationOperation) {
    /** Add new circle */
    circles.emplace_back(
      cv::Point(generate_random_center_x(rnd_engine), generate_random_center_y(rnd_engine)),
      generate_random_radius(rnd_engine),
      cv::Scalar(
        generate_random_color_channel_value(rnd_engine),
        generate_random_color_channel_value(rnd_engine),
        generate_random_color_channel_value(rnd_engine)
      ));
  } else if (1 == kMutationOperation) {
    /** Remove existing circle */
    std::uniform_int_distribution<int> generate_random_circle_index(0, circles.size() - 1);
    const int kRandomCircleIndex = generate_random_circle_index(rnd_engine);

    circles.erase(circles.begin() + kRandomCircleIndex);
  } else {
    /** Modify existing circle */
    std::uniform_int_distribution<int> generate_random_circle_index(0, circles.size() - 1);
    const int kRandomCircleIndex = generate_random_circle_index(rnd_engine);

    Circle& circle = circles[kRandomCircleIndex];

    const int kCircleMutationOperation = generate_random_circle_mutation_operation(rnd_engine);

    if (0 == kCircleMutationOperation) {
      /** Modify position */
      circle.center.x =
        clamp(circle.center.x + static_cast<int>(generate_random_mutation_center_x(rnd_engine)), 0, image.cols - 1);
      circle.center.y =
        clamp(circle.center.y + static_cast<int>(generate_random_mutation_center_y(rnd_engine)), 0, image.rows - 1);
    } else if (1 == kCircleMutationOperation) {
      /** Modify radius */
      circle.radius = std::max(1, circle.radius + static_cast<int>(generate_random_mutation_radius(rnd_engine)));
    } else {
      /** Modify color */
      for (int i = 0; i < 3; ++i) {
        circle.color.val[i] =
          clamp(static_cast<int>(circle.color.val[i] +
            generate_random_mutation_color_channel_value(rnd_engine)), 0, 255);
      }
    }
  }

  for (auto& circle : circles) {
    cv::Mat temp = image.clone();
    temp = cv::Scalar();
    cv::circle(temp, circle.center, circle.radius, circle.color, CV_FILLED);

    cv::addWeighted(temp, 0.5, image, 1, 0, image);
  }

  cv::cvtColor(image, image_lab, CV_BGR2Lab);

  distance = cv::norm(original_image_lab, image_lab);
}

int main(int argc, char* argv[]) {
  if (3 != argc) {
    print_help();
    return EXIT_FAILURE;
  }

  const std::string kInputImageFilename = argv[1];
  const std::string kOutputImageFilename = argv[2];

  const cv::Mat original_image = cv::imread(kInputImageFilename, CV_LOAD_IMAGE_COLOR);
  const cv::Mat original_image_lab = [&original_image]() {
    cv::Mat temp;
    cv::cvtColor(original_image, temp, CV_BGR2Lab);
    return temp;
  }();

  if (original_image.empty()) {
    std::cerr << "Could not read image \"" << kInputImageFilename << "\"\n";
    return EXIT_FAILURE;
  }

  std::cout << "Press ESC to save output image and exit ..." << std::endl;

  const std::string kOriginalImageWindowName = "Original image - Press ESC to save output image and exit ...";
  cv::namedWindow(kOriginalImageWindowName, CV_GUI_EXPANDED);
  cv::imshow(kOriginalImageWindowName, original_image);

  {
    cv::Mat output_image = original_image.clone();
    output_image = cv::Scalar();

    const std::string kOutputImageWindowName = "Output image - Press ESC to save output image and exit ...";
    cv::namedWindow(kOutputImageWindowName, CV_GUI_EXPANDED);
    cv::imshow(kOutputImageWindowName, output_image);

    Circles best_circles;
    double best_distance = -1;

    const unsigned kNumWorkers = std::max(static_cast<unsigned>(1), std::thread::hardware_concurrency());

    std::vector<std::future<void>> results(kNumWorkers);
    std::vector<Circles> circles(kNumWorkers);
    std::vector<cv::Mat> images(kNumWorkers);
    std::vector<cv::Mat> images_lab(kNumWorkers);
    std::vector<double> distances(kNumWorkers, -1);
    std::vector<std::mt19937> rnd_engines(kNumWorkers);
    std::vector<std::uniform_int_distribution<int>>
      generate_random_center_x(kNumWorkers, std::uniform_int_distribution<int>(0, output_image.cols - 1));
    std::vector<std::uniform_int_distribution<int>>
      generate_random_center_y(kNumWorkers, std::uniform_int_distribution<int>(0, output_image.rows - 1));
    const int kMaxDimension = std::max(output_image.cols, output_image.rows);
    std::vector<std::uniform_int_distribution<int>>
      generate_random_radius(kNumWorkers, std::uniform_int_distribution<int>(1, kMaxDimension));
    std::vector<std::uniform_int_distribution<int>>
      generate_random_color_channel_values(kNumWorkers, std::uniform_int_distribution<int>(0, 255));
    std::vector<std::discrete_distribution<>>
      generate_random_mutation_operations(kNumWorkers, std::discrete_distribution<>({40, 30, 30}));
    std::vector<std::uniform_int_distribution<>>
      generate_random_circle_mutation_operations(kNumWorkers, std::uniform_int_distribution<>(0, 2));
    std::vector<std::normal_distribution<>>
      generate_random_mutation_center_x(kNumWorkers,
        std::normal_distribution<>(1, std::max(1, output_image.cols / 5)));
    std::vector<std::normal_distribution<>>
      generate_random_mutation_center_y(kNumWorkers,
        std::normal_distribution<>(1, std::max(1, output_image.rows / 5)));
    std::vector<std::normal_distribution<>>
      generate_random_mutation_radius(kNumWorkers, std::normal_distribution<>(1, std::max(1, kMaxDimension / 5)));
    std::vector<std::normal_distribution<>>
      generate_random_mutation_color_channel_values(kNumWorkers, std::normal_distribution<>(0, 32));
    for (unsigned i = 0; i < kNumWorkers; ++i) {
      images[i] = output_image.clone();
      images_lab[i] = output_image.clone();
      rnd_engines[i] = std::mt19937(std::random_device{}());
    }

    std::chrono::system_clock::time_point last_time_gui_update;
    const int kGuiUpdateIntervalMs = 200;

    while(true) {
      for (unsigned i = 0; i < kNumWorkers; ++i) {
        circles[i] = best_circles;

        results[i] =
          std::async(
            std::launch::async,
            generate_end_evaluate_solution,
            std::ref(circles[i]),
            std::ref(images[i]),
            std::ref(images_lab[i]),
            std::ref(distances[i]),
            std::ref(rnd_engines[i]),
            std::ref(generate_random_center_x[i]),
            std::ref(generate_random_center_y[i]),
            std::ref(generate_random_radius[i]),
            std::ref(generate_random_color_channel_values[i]),
            std::ref(generate_random_mutation_operations[i]),
            std::ref(generate_random_circle_mutation_operations[i]),
            std::ref(generate_random_mutation_center_x[i]),
            std::ref(generate_random_mutation_center_y[i]),
            std::ref(generate_random_mutation_radius[i]),
            std::ref(generate_random_mutation_color_channel_values[i]),
            std::cref(original_image_lab));
      }

      /** Wait for results and also get best candidate */
      double best_candidate_distance = -1;
      unsigned best_candidate_index = 0;
      for (unsigned i = 0; i < kNumWorkers; ++i) {
        results[i].get();
        if (-1 == best_candidate_distance || distances[i] < best_candidate_distance) {
          best_candidate_distance = distances[i];
          best_candidate_index = i;
        }
      }

      if (-1 == best_distance || best_candidate_distance < best_distance) {
        best_distance = best_candidate_distance;
        output_image = images[best_candidate_index].clone();
        best_circles = circles[best_candidate_index];
      }

      {
        auto now = std::chrono::system_clock::now();
        const auto& gui_update_time_diff_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time_gui_update).count();
        if (gui_update_time_diff_ms >= kGuiUpdateIntervalMs) {
          cv::imshow(kOriginalImageWindowName, original_image);
          cv::imshow(kOutputImageWindowName, output_image);
          last_time_gui_update = now;
          /** Wait for ESC key */
          if (27 == static_cast<char>(cv::waitKey(1))) {
            break;
          }
        }
      }
    }

    cv::imwrite(kOutputImageFilename, output_image);
  }

  cv::destroyAllWindows();
};
