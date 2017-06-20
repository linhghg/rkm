#ifndef RKM_H_
#define RKM_H_
#include <assert.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "kernel_data.h"

namespace RKM
{

class rkm
{
    public:
        rkm(const std::string& input_file_name);
        ~rkm();
        void solve();
        void solve_no_fs();
        void solve_iter();
        double predict(const std::vector<double>& x) const;
        void test() const;
        // number of relevance vectors/features
        int get_n_rv() const;
        int get_n_rf() const;
        // write/read alpha, beta
        void write_model_file() const;
        void read_model_file(const std::string& model_file_name);
        void read_model_file();
    private:
        // Data related
        kernel_data* kd;
        std::string train_file;
        std::string test_file;
        std::string model_file;

        // Model related
        // Cost coefficients
        double Cp;
        double Cn;
        double Cb;
        // Stoping Criteria
        double eps;
        double tau;

        // Solution
        std::vector<double> alpha;
        std::vector<double> beta;
        double rho; // -b as defined in LibSVM

        // Gradient related
        std::vector<double> Ga;
        std::vector<double> Gb;
        std::vector<double> Q_alpha;

        enum alpha_status_type { LOWER_BOUND, UPPER_BOUND, FREE };
        std::vector<alpha_status_type> alpha_status;
        bool is_upper_bound(size_t i) const { return alpha_status[i] == UPPER_BOUND; }
        bool is_lower_bound(size_t i) const { return alpha_status[i] == LOWER_BOUND; }
        bool is_free(size_t i) const { return alpha_status[i] == FREE; }
        bool is_in_I_up(size_t i) const;
        bool is_in_I_low(size_t i) const;

        // Program related
        int verbose;
        // Encapsultate for readability
        double K(size_t i, size_t j) const;
        double Y(size_t i) const;
        double Q(size_t i, size_t j, size_t k) const;

        // subroutins
        double get_C(size_t i) const;
        void update_alpha_status(size_t i);
        bool select_working_set(size_t& i, size_t& j) const;
        bool select_working_set(size_t& i, size_t& j, double& delta_obj) const;
        bool select_working_feature(size_t& k) const;
        double calculate_rho() const;
        double sum_of_beta() const;

}; //class rkm

struct delim
{
    delim(char _c) : c(_c) { }
    char c;
};

inline std::istream& operator>>(std::istream& is, delim x)
{
    char c;
    if (is >> c && c != x.c)
        is.setstate(std::iostream::failbit);
    return is;
}

} //namespace RKM

#endif // RKM_H_
