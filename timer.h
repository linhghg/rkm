#ifndef TIMER_H_
#define TIMER_H_
#include <string>
#include <iostream>
#include <chrono>

namespace RKM
{

class timer
{
    public:
        void start();
        void stop();
        void stop(const std::string& message);
        void report(const std::string& message) const;
    private:
        std::chrono::system_clock::time_point t1;
        std::chrono::system_clock::time_point t2;

}; //class timer

} //namespace RKM

#endif // TIMER_H_
