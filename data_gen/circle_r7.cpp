#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

int main(){
    srand(0);   // to reproduce
    ofstream fout1("train.dat");
    for (int i=0;i<400;++i)
    {
        int label = 1;
        double x1 = 1.2*rand()/RAND_MAX;
        double x2 = 1.2*rand()/RAND_MAX;
        double x3 = 1.2*rand()/RAND_MAX;
        double x4 = 1.2*rand()/RAND_MAX;
        double x5 = 1.2*rand()/RAND_MAX;
        double x6 = 1.2*rand()/RAND_MAX;
        double x7 = 1.2*rand()/RAND_MAX;
        if (x2*x2+x5*x5>1.0)
        {
            label = -1;
        }
        fout1<<label<<": ";
        fout1<<x1<<" ";
        fout1<<x2<<" ";
        fout1<<x3<<" ";
        fout1<<x4<<" ";
        fout1<<x5<<" ";
        fout1<<x6<<" ";
        fout1<<x7<<"\n";
    }

    ofstream fout2("test.dat");
    for (int i=0;i<400;++i)
    {
        int label = 1;
        double x1 = 1.2*rand()/RAND_MAX;
        double x2 = 1.2*rand()/RAND_MAX;
        double x3 = 1.2*rand()/RAND_MAX;
        double x4 = 1.2*rand()/RAND_MAX;
        double x5 = 1.2*rand()/RAND_MAX;
        double x6 = 1.2*rand()/RAND_MAX;
        double x7 = 1.2*rand()/RAND_MAX;
        if (x2*x2+x5*x5>1.0)
        {
            label = -1;
        }
        fout2<<label<<": ";
        fout2<<x1<<" ";
        fout2<<x2<<" ";
        fout2<<x3<<" ";
        fout2<<x4<<" ";
        fout2<<x5<<" ";
        fout2<<x6<<" ";
        fout2<<x7<<"\n";
    }
    fout2.close();
}
