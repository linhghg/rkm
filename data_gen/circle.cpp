#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

int main(){
    srand(0);   // to reproduce
    ofstream fout1("train.dat");
    for (int i=0;i<=12;++i)
    {
        for (int j=0;j<=12;++j)
        {
            int label = 1;
            if (i*i+j*j>100)
            {
                label = -1;
            }
            fout1<<label<<" : ";
            double val = i/10.0;
            fout1<<val<<" ";
            val = 1.2*rand()/RAND_MAX;
            fout1<<val<<" ";
            val = j/10.0;
            fout1<<val<<" ";
            val = 1.2*rand()/RAND_MAX;
            fout1<<val<<endl;
        }
    }
    fout1.close();

    ofstream fout2("test.dat");
    for (int i=0;i<100;++i)
    {
        int label = 1;
        double x1 = 1.2*rand()/RAND_MAX;
        double x2 = 1.2*rand()/RAND_MAX;
        double x3 = 1.2*rand()/RAND_MAX;
        double x4 = 1.2*rand()/RAND_MAX;
        if (x1*x1+x3*x3>1.0)
        {
            label = -1;
        }
        fout2<<label<<": ";
        fout2<<x1<<" ";
        fout2<<x2<<" ";
        fout2<<x3<<" ";
        fout2<<x4<<endl;
    }
    fout2.close();
}
