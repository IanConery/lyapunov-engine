#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <mutex>

namespace lyapunov {
namespace logging {

enum class Level {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

inline const char* level_to_string(Level level) {
    switch (level) {
        case Level::DEBUG:   return "DEBUG";
        case Level::INFO:    return "INFO";
        case Level::WARNING: return "WARN";
        case Level::ERROR:   return "ERROR";
        default:             return "UNKNOWN";
    }
}

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(Level level) {
        current_level_ = level;
    }

    Level get_level() const {
        return current_level_;
    }

    void log(Level level, const std::string& message) {
        if (level < current_level_) {
            return;
        }

        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::lock_guard<std::mutex> lock(mutex_);
        std::cerr << "[" << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S") << "] "
                  << "[" << level_to_string(level) << "] "
                  << message << std::endl;
    }

private:
    Logger() : current_level_(Level::INFO) {}
    Level current_level_;
    std::mutex mutex_;
};

#define LYAPUNOV_LOG(level, msg)                                                    \
    do {                                                                        \
        std::ostringstream ss;                                                  \
        ss << msg;                                                              \
        ::lyapunov::logging::Logger::instance().log(::lyapunov::logging::Level::level, ss.str()); \
    } while (0)

#define LYAPUNOV_LOG_INFO(msg)  LYAPUNOV_LOG(INFO, msg)
#define LYAPUNOV_LOG_WARN(msg)  LYAPUNOV_LOG(WARNING, msg)
#define LYAPUNOV_LOG_ERROR(msg) LYAPUNOV_LOG(ERROR, msg)
#define LYAPUNOV_LOG_DEBUG(msg) LYAPUNOV_LOG(DEBUG, msg)

} // namespace logging
} // namespace lyapunov
