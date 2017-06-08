#ifndef KERNEL_DATA_H_
#define KERNEL_DATA_H_
#include <vector>
#include <assert.h>
#include <iostream>
#include <limits>
#include <math.h>

namespace RKM
{

enum kernel_type {LINEAR, GAUSSIAN};

class kernel_data
{
    public:
        // Constructor/Destructor
        kernel_data(const size_t _n_sample, const size_t _n_feature);
        ~kernel_data();

        // Data related manipulation
        void clear_data();
        void x_append(const std::vector<double> _x);
        void t_append(const double _t);

        double get_data(const size_t i, const size_t j) const;
        double get_label(const size_t i) const;
        // Uses append to create data, and expect no further changes once created
        //void set_data(const int i, const int j, const double val);
        //void set_label(const int i, const double val);

        size_t get_n_sample() const;
        size_t get_n_feature() const;

        void scale_features();

        void print_data(std::ostream& os) const;

        // Kernel
        void set_kernel(const std::string& kernel_name);
        void set_gamma(double _gamma);
        void print_kernel(std::ostream& os) const;

        // atomized kernel, i-th and j-th vectors, k-th feature
        double kernel_one(size_t i, size_t j, size_t k) const;
        double K(size_t i, size_t j, const std::vector<double>& v) const;

    private:
        // Data
        std::vector<double> x;  // samples/features
        std::vector<double> t;  // lables
        size_t n_sample;
        size_t n_feature;

        // Feature scaling
        // Normalization: x' = (x-min)/(max-min)
        // Recover: x = x'*(max-min)+min
        std::vector<double> max_x;
        std::vector<double> min_x;

        // Kenrel related
        kernel_type kern_type;
        double gamma;

};

}   // namespace rkm
#endif  // KERNEL_DATA_H_
