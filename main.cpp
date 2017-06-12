#include <iostream>
#include "rkm.h"

int main(int argc, char* argv[]){
    std::string str(argv[1]);
    RKM::rkm solver(str);
    solver.solve();
    solver.write_model_file();
    solver.test();
}
