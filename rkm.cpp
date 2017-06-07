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
        alpha_status.resize(n_sample);
        for (size_t i=0;i<n_sample;++i)
        {
            update_alpha_status(i);
        }

        // TBD: shrinking heuristic

        // Gradient init
        std::vector<double> G(n_sample);
        std::vector<double> G_bar(n_sample);
        for (size_t i=0;i<n_sample;++i)
        {
            G[i] = -1;
            G_bar[i] = 0;
        }
        //for (size_t i=0;i<n_sample;++i)
        //{
        //    if (!is_lower_bound(i))
        //    {
        //        for (size_t j=0;j<n_sample;++j)
        //        {
        //            G[i] += alpha[i]*K(i, j); // TBD: yi, yj
        //        }
        //        if (is_upper_bound(i))
        //        {
        //            for (size_t j=0;j<n_sample;++j)
        //            {
        //                G_bar[i] += get_C(i)*K(i, j);
        //            }
        //        }
        //    }
        //}

        // optimization step

        int iter = 0;
        int max_iter = n_sample*100;
        if (n_sample>std::numeric_limits<int>::max()/100)
        {
            max_iter = std::numeric_limits<int>::max();
        }
        if (max_iter < 10000000) max_iter = 10000000;
        int counter = min(n_sample,1000)+1;

        while(iter < max_iter)
        {
            // show progress, shrinking TBD
            if(--counter == 0)
            {
                counter = min(n_sample,1000);
                //if(shrinking) do_shrinking();
                std::cout<<".";
            }
        } // while(iter < max_iter)

    } // void rkm::solve()

    int rkm::select_working_set(size_t& i, size_t& j) const
    {
        // return i,j such that
        // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        // j: minimizes the decrease of obj value
        //    (if quadratic coefficeint <= 0, replace it with tau)
        //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

        double Gmax = -INF;
        double Gmax2 = -INF;
        int Gmax_idx = -1;
        int Gmin_idx = -1;
        double obj_diff_min = INF;

        for(int t=0;t<active_size;t++)
            if(y[t]==+1)
            {
                if(!is_upper_bound(t))
                    if(-G[t] >= Gmax)
                    {
                        Gmax = -G[t];
                        Gmax_idx = t;
                    }
            }
            else
            {
                if(!is_lower_bound(t))
                    if(G[t] >= Gmax)
                    {
                        Gmax = G[t];
                        Gmax_idx = t;
                    }
            }

        int i = Gmax_idx;
        const Qfloat *Q_i = NULL;
        if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
            Q_i = Q->get_Q(i,active_size);

        for(int j=0;j<active_size;j++)
        {
            if(y[j]==+1)
            {
                if (!is_lower_bound(j))
                {
                    double grad_diff=Gmax+G[j];
                    if (G[j] >= Gmax2)
                        Gmax2 = G[j];
                    if (grad_diff > 0)
                    {
                        double obj_diff;
                        double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
                        if (quad_coef > 0)
                            obj_diff = -(grad_diff*grad_diff)/quad_coef;
                        else
                            obj_diff = -(grad_diff*grad_diff)/TAU;

                        if (obj_diff <= obj_diff_min)
                        {
                            Gmin_idx=j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            }
            else
            {
                if (!is_upper_bound(j))
                {
                    double grad_diff= Gmax-G[j];
                    if (-G[j] >= Gmax2)
                        Gmax2 = -G[j];
                    if (grad_diff > 0)
                    {
                        double obj_diff;
                        double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
                        if (quad_coef > 0)
                            obj_diff = -(grad_diff*grad_diff)/quad_coef;
                        else
                            obj_diff = -(grad_diff*grad_diff)/TAU;

                        if (obj_diff <= obj_diff_min)
                        {
                            Gmin_idx=j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            }
        }

        if(Gmax+Gmax2 < eps || Gmin_idx == -1)
        return 1;

        out_i = Gmax_idx;
        out_j = Gmin_idx;
        return 0;
    } // int rkm::select_working_set(size_t& i, size_t& j) const

    double rkm::get_C(size_t i) const
    {
        return ((kd->get_label(i)>0)? Cp : Cn);
    }

    void rkm::update_alpha_status(size_t i)
    {
        if (alpha[i] >= get_C(i))
            alpha_status[i] = UPPER_BOUND;
        else if(alpha[i] <= 0)
            alpha_status[i] = LOWER_BOUND;
        else alpha_status[i] = FREE;
    }

} // namespace rkm
