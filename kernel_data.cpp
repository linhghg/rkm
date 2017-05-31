#include "kernel_data.h"

namespace RKM
{

kernel_data::kernel_data(const size_t _n_sample, const size_t _n_feature)
{
    n_sample = _n_sample;
    n_feature = _n_feature;
    // Memory allocation
    x.reserve(_n_sample*_n_feature);
    t.reserve(_n_sample);
}

kernel_data::~kernel_data()
{
    clear_data();
}

void kernel_data::clear_data()
{
    x.clear();
    t.clear();
}

void kernel_data::x_append(const std::vector<double> _x)
{
    for (size_t i=0;i<_x.size();++i)
    {
        x.push_back(_x[i]);
    }
}

void kernel_data::t_append(const double _t)
{
    t.push_back(_t);
}

double kernel_data::get_data(const size_t i, const size_t j) const
{
    return x[i*n_feature+j];
}

double kernel_data::get_label(const size_t i) const
{
    return(t[i]);
}

//void kernel_data::set_data(const int i, const int j, const double val)
//{
//    x[i*n_feature+j] = val;
//}
//
//void kernel_data::set_label(const int i, const double val)
//{
//    t[i] = val;
//}

size_t kernel_data::get_n_sample() const
{
    //return t.size();
    return n_sample;
}

size_t kernel_data::get_n_feature() const
{
    //size_t n_feature = x.size()/t.size();
    //assert(t.size()*n_feature == x.size());
    return n_feature;
}

void kernel_data::scale_features()
{
    // Initialize max_x and min_x
    max_x.resize(n_feature, -1.0*std::numeric_limits<double>::infinity());
    min_x.resize(n_feature, +1.0*std::numeric_limits<double>::infinity());
    // Traversing the data to collect min and max
    for (size_t i=0;i<n_sample;++i)
    {
        for (size_t j=0;j<n_feature;++j)
        {
            size_t idx = i*n_feature+j;
            if (x[idx] < min_x[j])
            {
                min_x[j] = x[idx];
            }
            if (x[idx] > max_x[j])
            {
                max_x[j] = x[idx];
            }
        }
    }

    // check max_x - min_x > 0
    for (size_t j=0;j<n_feature;++j)
    {
        assert(max_x[j]>min_x[j]);
    }

    // Scale the data
    // Normalization: x' = (x-min)/(max-min)
    for (size_t i=0;i<n_sample;++i)
    {
        for (size_t j=0;j<n_feature;++j)
        {
            size_t idx = i*n_feature+j;
            x[idx] = (x[idx] - min_x[j]) / (max_x[j] - min_x[j]);
            assert(x[idx] == x[idx]); // NaN check
        }
    }
} //void kernel_data::scale_features()

void kernel_data::print_data(std::ostream& os) const
{
    for (size_t i=0;i<t.size();++i)
    {
        os<<get_label(i)<<" :";
        for (size_t j=0;j<n_feature;++j)
        {
            os<<" "<<get_data(i, j);
        }
        os<<"\n";
    }
}

} // namespace rkm
