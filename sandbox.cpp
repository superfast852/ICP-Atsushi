#include <eigen3/Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>

using namespace std::chrono;
using namespace std;
using namespace Eigen;

class ICP {
public:
    ICP() = default;
    ~ICP() = default;

    std::vector<Matrix3f> H_trace;

    Matrix3f run(const std::vector<Vector2f>& target, const std::vector<Vector2f>& src, int max_iter, float tol, bool trace = false) {
        Matrix3f H;
        bool initialized = false;
        if (trace){
            H_trace.clear();
        }
        auto start = high_resolution_clock::now();
        H.setConstant(std::numeric_limits<double>::quiet_NaN());
        MatrixXd goal = vectorToMatrix(target);
        MatrixXd src_mat = vectorToMatrix(src);
        MatrixXf src_sort;
        float dErr = MAXFLOAT;
        float preError = MAXFLOAT;
        cout << "Initialization: " << duration_cast<milliseconds>(high_resolution_clock::now()-start).count() << endl;
        for (int i = 0; i<max_iter; i++) {
            if (abs(dErr) < tol) {
                break;
            }
            start = high_resolution_clock::now();
            float error = point_error(goal, src_mat);
            MatrixXi indexes = getNearestNeighbors(goal, src_mat).cast<int>();
            src_sort = sortPoints(src_mat.cast<float>(), indexes);
            cout << "NNA: " << duration_cast<milliseconds>(high_resolution_clock::now()-start).count() << endl;


            // TODO: this is autogenerated. If it does not work, look at this first. Eq: svd_n2
            JacobiSVD<MatrixXd> homogenous = getHomogenous(goal, src_sort.cast<double>());
            Matrix2d R = getRotation(homogenous);
            Vector2d t = getTranslation(getCenterOfMass(goal), getCenterOfMass(src_sort.cast<double>()), R);
            src_mat = src_mat * R.transpose();
            src_mat = src_mat.rowwise() + t.transpose();
            dErr = preError - error;

            if (dErr < 0 && calc_dist_means(goal, src_mat) > 10) {
                break;
            }

            preError = error;
            if (!initialized){
                H = update_homogenous(R.cast<float>(), t.cast<float>());
                initialized = true;
            } else {
                H = update_homogenous(H, R.cast<float>(), t.cast<float>());
            }

            if (trace){
                H_trace.push_back(H);
            }

            if (abs(dErr) < tol) {
                break;
            }
        }
        return H;
    }
private:
    static Matrix3f update_homogenous(const Matrix3f& Hin, const Matrix2f& R, const Vector2f& t) {
        Matrix3f H = Matrix3f::Zero(3, 3);
        H.block(0, 0, 2, 2) = R;
        H.block(0, 2, 2, 1) = t;
        H(2, 2) = 1;

        return H * Hin;
    }

    static Matrix3f update_homogenous(const Matrix2f& R, const Vector2f& t) {
        Matrix3f H = Matrix3f::Zero(3, 3);
        H.block(0, 0, 2, 2) = R;
        H.block(0, 2, 2, 1) = t;
        H(2, 2) = 1;

        return H;
    }

    // Optimize this.
    static double point_error(const MatrixXd& goal, const MatrixXd& src) {
        double sum = 0;
        for (int i = 0; i < goal.rows(); i++) {
            sum += (goal.row(i) - src.row(i)).norm();
        }
        return sum;
    }


    // Maybe optimize this?
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

    VectorXd calculate_distances(const MatrixXd& current_points, const MatrixXd& previous_points) {
        VectorXd out = (custom_repeat(current_points, previous_points.rows()) - tiled_matrix(previous_points, current_points.rows())).rowwise().norm();
        return out;
    }

    VectorXd argmins(const MatrixXd& ve) {
        VectorXd indices = VectorXd::Zero(ve.rows());
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

    VectorXd getNearestNeighbors(const MatrixXd& current_points, const MatrixXd& previous_points) {
        return argmins(calculate_distances(current_points, previous_points).reshaped(current_points.rows(), previous_points.rows()));
    }

    MatrixXd vectorToMatrix(const std::vector<Vector2f>& vec) {
        MatrixXd mat(vec.size(), 2);
        for (int i = 0; i < vec.size(); i++) {
            mat(i, 0) = vec[i](0);
            mat(i, 1) = vec[i](1);
        }
        return mat;
    }

    Matrix2d getRotation(JacobiSVD<MatrixXd> homogenous){
        MatrixXd U = homogenous.matrixU();
        MatrixXd V = homogenous.matrixV();
        Matrix2d R = V*U.transpose();
        if (R.determinant() < 0) {
            V.col(1) *= -1;
            R = V * U.transpose();
        }
        return R;
    }

    Vector2d getTranslation(Vector2d com_prev, Vector2d com_curr, Matrix2d R){
        return com_prev - R * com_curr;
    }

    JacobiSVD<MatrixXd> getHomogenous(MatrixXd prev, MatrixXd curr){
        Vector2d com_prev = getCenterOfMass(prev);
        Vector2d com_curr = getCenterOfMass(curr);
        MatrixXd prev_centered = prev.rowwise() - com_prev.transpose();
        MatrixXd curr_centered = curr.rowwise() - com_curr.transpose();

        MatrixXd W = curr_centered.transpose() * prev_centered;
        // Get Singular Value Decomposition
        JacobiSVD<MatrixXd> svd(W, ComputeThinU | ComputeThinV);
        return svd;
    }

    MatrixXf sortPoints(const MatrixXf& src, const VectorXi& indexes) {
        MatrixXf sorted(src.rows(), src.cols());
        for (int i = 0; i < src.rows(); i++) {
            sorted.row(i) = src.row(indexes(i));
        }
        return sorted;
    }

    double calc_dist_means(const MatrixXd &p1, const MatrixXd &p2) {
        return (p1.colwise().mean() - p2.colwise().mean()).norm();
    }

    Vector2d getCenterOfMass(const MatrixXd& mat) {
        Vector2d com = Vector2d::Zero();
        for (int i = 0; i < mat.rows(); i++) {
            com += mat.row(i).transpose();
        }
        return com / mat.rows();
    }

};



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

int main() {
    // Example usage
    int n_pts = 2000;
    float dev = 0.1;
    float x_shift = 0;
    float rot = M_PI / 2;  // 45 degrees in radians
    Vector2f trans(2.0, 2.0);

    auto result = gen_sines(n_pts, dev, x_shift, rot, trans);

    ICP icp;
    Matrix3f H = icp.run(result[0], result[1], 100, 1e-5);
    std::cout << "Homogenous matrix:\n" << H << std::endl;

    return 0;
}

