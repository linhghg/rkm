#include "rkm.h"

namespace RKM
{

    rkm::rkm(const std::string& input_file_name)
    {
        // all kinds of settings to be read
        std::string task_name;
        std::string kernel_name;
        size_t n_feature = 0;
        size_t n_sample = 0;
        double gamma;
        double tau;

        std::string str;
        // input file
        std::ifstream fin (input_file_name, std::ios_base::in);
        while (fin>>str)
        {
            fin>>delim(':');
            if (str.compare("train_file") == 0)
            {
                fin>>train_file;
            }
            else if (str.compare("test_file") == 0)
            {
                fin>>test_file;
            }
            else if (str.compare("verbose") == 0)
            {
                fin>>verbose;
            }
            else if (str.compare("task") == 0)
            {
                fin>>task_name;
            }
            else if (str.compare("Cp") == 0)
            {
                fin>>Cp;
            }
            else if (str.compare("Cn") == 0)
            {
                fin>>Cn;
            }
            else if (str.compare("eps") == 0)
            {
                fin>>eps;
            }
            else if (str.compare("kernel") == 0)
            {
                fin>>kernel_name;
            }
            else if (str.compare("gamma") == 0)
            {
                fin>>gamma;
            }
            else if (str.compare("tau") == 0)
            {
                fin>>tau;
            }
            else
            {
                fin>>str;
            }
        } // while (fin>>str)
        fin.close();

        if (verbose)
        {
            std::cout<<"========================\n";
            std::cout<<"Task: "<<task_name<<"...\n";
            std::cout<<"========================\n";
            std::cout<<"Reading training data file: "<<train_file<<"...\n";
        }

        // training data file
        fin.open(train_file, std::ios_base::in);
        // calc n_sample and n_feature
        while (std::getline(fin, str))
        {
            n_sample++;
            if (n_feature == 0)
            {
                std::stringstream ss(str);
                double temp;
                ss>>temp>>delim(':');
                while (ss>>temp)
                {
                    n_feature++;
                }
            }
        }
        fin.close();

        if (verbose)
        {
            std::cout<<"Found "<<n_sample<<" samples with "<<n_feature<<" features.\n";
            std::cout<<"Loading data to solver...\n";
        }

        // vector size allocation
        kd = new kernel_data(n_sample, n_feature);
        // set parameters
        kd -> set_kernel(kernel_name);
        kd -> set_gamma(gamma);
        kd -> set_tau(tau);

        // read data into the kernel_data
        fin.open(train_file, std::ios_base::in);
        std::vector<double> x(n_feature);
        double t;
        for (size_t i=0;i<n_sample;++i)
        {
            fin>>t>>delim(':');
            kd->t_append(t);
            for (size_t j=0;j<n_feature;++j)
            {
                fin>>x[j];
            }
            kd->x_append(x);
        }
        fin.close();

        if (verbose)
        {
            std::cout<<"Feature scaling...\n";
        }
        // Normalization
        kd->scale_features();

    } //rkm::rkm(const std::string& input_file_name)

    rkm::~rkm()
    {
        if (kd)
        {
            delete kd;
        }
    }

    double rkm::K(size_t i, size_t j) const
    {
        return kd->K(i, j, beta);
    }

    void rkm::solve()
    {
        size_t n_feature = kd->get_n_feature();
        size_t n_sample = kd->get_n_sample();

        // TBD
        // Initialization
        beta.resize(n_feature, 1.0);
        beta[1] = 0.0;
        beta[3] = 0.0;

        alpha.resize(n_sample, 0.0);
    }

} // namespace rkm
