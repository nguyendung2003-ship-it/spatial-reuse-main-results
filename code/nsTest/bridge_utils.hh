#ifndef __BRIDGE_UTILS_HH
#define __BRIDGE_UTILS_HH

#include <string>

namespace BridgeUtils {
	bool sendAll(int fd, const std::string& payload);
	bool receiveLine(int fd, std::string& line, unsigned int maxBytes = 1024 * 1024);
	bool connectTcp(const std::string& host, const std::string& port, int& fd, std::string& error);
}

#endif
