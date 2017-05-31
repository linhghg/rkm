#ifndef RKM_H_
#define RKM_H_
#include <assert.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "kernel_data.h"

namespace RKM
{

class rkm
{
    public:
        rkm(const std::string& input_file_name);
        ~rkm();
    private:
        // Data related
        kernel_data* kd;
        std::string train_file;
        std::string test_file;

        // Model related

        // Program related
        int verbose;

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
