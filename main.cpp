#include <iostream>
#include "rkm.h"
#include "timer.h"

int main(int argc, char* argv[]){
    std::string str(argv[1]);
    RKM::timer run_time;

    run_time.start();
    RKM::rkm solver(str);
    run_time.stop("## Loading data time");

    run_time.start();
    solver.solve();
    //solver.solve_no_fs();
    //solver.solve_iter();
    run_time.stop("## Training time");

    solver.write_model_file();
    //solver.read_model_file();
    run_time.start();
    solver.test();
    run_time.stop("## Testing time");
}
