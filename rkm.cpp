#include "rkm.h"

namespace RKM
{

    rkm::rkm(const std::string& input_file_name)
    {
        std::ifstream fin (input_file_name, std::ios_base::in);
        int temp;
        fin>>temp>>delim(':');
        std::cout<<temp<<std::endl;
        fin>>temp;
        std::cout<<temp<<std::endl;
        fin.close();
    }

    rkm::~rkm()
    {
        delete kd;
    }

} // namespace rkm
