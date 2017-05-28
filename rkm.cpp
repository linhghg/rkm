#include "rkm.h"

namespace RKM
{

    rkm::rkm(const std::size_t i, const std::size_t j)
    {
        kd = new kernel_date(i, j);
    }

    rkm::~rkm()
    {
        if (kd)
        {
            delete kd;
        }
    }

} // namespace rkm
