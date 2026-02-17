#ifndef COMMANDQUEUE_HH
#define COMMANDQUEUE_HH

#include <queue>
#include <mutex>
#include <string>
namespace B4 {
    struct ParticleCommand {
        std::string particleType;
        double energy;
        int count;
    };

    // Global Shared Variables
    extern std::queue<ParticleCommand> g_commandQueue;
    extern std::mutex g_queueMutex;
}
#endif