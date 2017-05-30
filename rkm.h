#ifndef RKM_H_
#define RKM_H_
#include <assert.h>
#include <string>
#include <iostream>
#include <fstream>
#include "kernel_data.h"

namespace RKM
{

class rkm
{
    public:
        rkm(const std::string& input_file_name);
        ~rkm();
    private:
        kernel_data* kd;
}; //class kernel_data

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
