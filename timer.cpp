#include "timer.h"

namespace RKM
{
    void timer::start()
    {
        t1 = std::chrono::system_clock::now();
    }
    void timer::stop()
    {
        t2 = std::chrono::system_clock::now();
    }
    void timer::stop(const std::string& message)
    {
        stop();
        report(message);
    }
    void timer::report(const std::string& message) const
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        double elapsed_in_secs = elapsed.count()*1e-3;
        std::cout<<message<<": "<<elapsed_in_secs<<" s\n";
    }
} // namespace RKM
