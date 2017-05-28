#ifndef KERNEL_DATA_H_
#define KERNEL_DATA_H_
#include <vector>
#include <assert.h>

namespace RKM
{

enum kernel_type {LINEAR, GAUSSIAN};

class kernel_data
{
    public:
        // Constructor/Destructor
        kernel_data(const std::size_t _n_sample, const std::size_t _n_feature);
        ~kernel_data();

        // Data related manipulation
        void clear_data();
        void x_append(const std::vector<double> _x);
        void t_append(const double _t);

        double get_data(const std::size_t i, const std::size_t j) const;
        double get_label(const std::size_t i) const;
        // Uses append to create data, and expect no further changes once created
        //void set_data(const int i, const int j, const double val);
        //void set_label(const int i, const double val);

        std::size_t get_n_sample() const;
        std::size_t get_n_feature() const;

    private:
        // Data
        std::vector<double> x;  // samples/features
        std::vector<double> t;  // lables

        // Kenrel related
        kernel_type kern_type;
        double gamma;
        double tau;
};

}   // namespace rkm
#endif  // KERNEL_DATA_H_
