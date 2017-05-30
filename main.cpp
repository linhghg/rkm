#include <iostream>
#include "rkm.h"

int main(int argc, char* argv[]){
    std::string str(argv[1]);
    std::cout<<str<<std::endl;
    RKM::rkm solver(str);
}
