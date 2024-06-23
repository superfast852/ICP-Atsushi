//
// Created by gg on 17/06/24.
//

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>

using namespace std::chrono;
using namespace std;
using namespace Eigen;

// Function to generate linearly spaced vector
VectorXf linspace(float start, float end, int num) {
    VectorXf linspaced(num);
    float delta = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        linspaced[i] = start + delta * i;
    }
    return linspaced;
}

// Function to generate sine wave and its noisy, shifted copy
std::vector<std::vector<Vector2f>> gen_sines(int n_pts, float dev, float x_shift, float rot, Vector2f trans) {
    // Generate x values and pure sine wave y values
    VectorXf x = linspace(0, 6.28, n_pts);
    VectorXf y = x.array().sin();

    // Generate sine wave y values with x_shift
    VectorXf y_shifted = (x.array() + x_shift).sin();

    // Generate noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, dev);
    VectorXf noise(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        noise[i] = d(gen);
    }

    // Create noisy sine wave y values
    VectorXf y_noised = y_shifted + noise;

    // Create point vectors
    std::vector<Vector2f> pure_points(n_pts);
    std::vector<Vector2f> noised_points(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        pure_points[i] = Vector2f(x[i], y[i]);
        noised_points[i] = Vector2f(x[i], y_noised[i]);
    }

    // Function to morph graph (translate and rotate points)
    auto morph_graph = [](std::vector<Vector2f>& points, Vector2f translation, float theta) {
        // Create rotation matrix
        Matrix2f rotation_matrix;
        rotation_matrix << std::cos(theta), -std::sin(theta),
                std::sin(theta),  std::cos(theta);
        // Apply translation and rotation
        for (auto& point : points) {
            point += translation;
            point = rotation_matrix * point;
        }
    };

    // Morph the noisy sine wave points
    morph_graph(noised_points, trans, rot);

    return {pure_points, noised_points};
}


// NNA
static MatrixXd tiled_matrix(const MatrixXd& mat, int n) {
    int rows = mat.rows();
    int cols = mat.cols();

    MatrixXd tiled_matrix(rows*n, cols);

    for (int i = 0; i < n; ++i) {
        tiled_matrix.block(i*rows, 0, rows, cols) = mat;
    }
    return tiled_matrix;
}

static MatrixXd custom_repeat(const MatrixXd& mat, int n) {
    int rows = mat.rows();
    int cols = mat.cols();

    MatrixXd repeated_matrix(rows*n, cols);
    for (int i = 0; i < n; ++i) {
        repeated_matrix.block(i*rows, 0, rows, cols) = mat.row(i).replicate(rows, 1);;
    }
    return repeated_matrix;
}

VectorXd argmins(const MatrixXd& ve) {
    VectorXd indices = VectorXd::Zero(ve.rows());
    #pragma omp parallel for
    for (int i = 0; i < ve.rows(); ++i) {
        double min = ve.row(i).minCoeff();
        for (int j = 0; j < ve.cols(); ++j) {
            if (ve(i, j) == min) {
                indices(i) = j;
                break;
            }
        }
    }
    return indices;
}

VectorXd calculate_distances(const MatrixXd& current_points, const MatrixXd& previous_points) {
    int current_rows = current_points.rows();
    int previous_rows = previous_points.rows();

    MatrixXd current_points_repeated = custom_repeat(current_points, previous_rows);
    MatrixXd previous_points_tiled = tiled_matrix(previous_points, current_rows);

    MatrixXd diff = current_points_repeated - previous_points_tiled;
    VectorXd distances(diff.rows());

    #pragma omp parallel for
    for (int i = 0; i < diff.rows(); ++i) {
        distances(i) = diff.row(i).norm();
    }
    return distances;
}

VectorXd getNearestNeighbors(const MatrixXd& current_points, const MatrixXd& previous_points) {
    VectorXd NN = argmins(calculate_distances(current_points, previous_points).reshaped(current_points.rows(), previous_points.rows()));
    return NN;
}

MatrixXd vectorToMatrix(const std::vector<Vector2f>& vec) {
    MatrixXd mat(vec.size(), 2);
    for (int i = 0; i < vec.size(); i++) {
        mat(i, 0) = vec[i](0);
        mat(i, 1) = vec[i](1);
    }
    return mat;
}


int main() {
    // Example usage
    int n_pts = 1000;
    float dev = 0.1;
    float x_shift = 0;
    float rot = M_PI / 2;  // 45 degrees in radians
    Vector2f trans(2.0, 2.0);

    auto result = gen_sines(n_pts, dev, x_shift, rot, trans);

    MatrixXd goal = vectorToMatrix(result[0]);
    MatrixXd src_mat = vectorToMatrix(result[1]);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    VectorXd NN = getNearestNeighbors(goal, src_mat);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    cout << "Time taken: " << duration/1000 << "ms" << endl;
}