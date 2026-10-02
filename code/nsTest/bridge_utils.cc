#include "bridge_utils.hh"

#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace BridgeUtils {
	bool sendAll(int fd, const std::string& payload) {
		const char* cursor = payload.c_str();
		size_t remaining = payload.size();
		while (remaining > 0) {
			ssize_t sent = send(fd, cursor, remaining, 0);
			if (sent < 0 && errno == EINTR) continue;
			if (sent <= 0) return false;
			cursor += sent;
			remaining -= sent;
		}

		return true;
	}

	bool receiveLine(int fd, std::string& line, unsigned int maxBytes) {
		line.clear();
		char c = '\0';
		while (true) {
			ssize_t received = recv(fd, &c, 1, 0);
			if (received < 0 && errno == EINTR) continue;
			if (received <= 0) return false;
			if (c == '\n') return true;
			if (c != '\r') line.push_back(c);
			if (line.size() > maxBytes) return false;
		}
	}

	bool connectTcp(const std::string& host, const std::string& port, int& fd, std::string& error) {
		fd = -1;
		error.clear();

		struct addrinfo hints;
		std::memset(&hints, 0, sizeof(hints));
		hints.ai_family = AF_UNSPEC;
		hints.ai_socktype = SOCK_STREAM;

		struct addrinfo* result = nullptr;
		int status = getaddrinfo(host.c_str(), port.c_str(), &hints, &result);
		if (status != 0) {
			error = gai_strerror(status);
			return false;
		}

		for (struct addrinfo* candidate = result; candidate != nullptr; candidate = candidate->ai_next) {
			int candidateFd = socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
			if (candidateFd < 0) continue;
			if (connect(candidateFd, candidate->ai_addr, candidate->ai_addrlen) == 0) {
				int noDelay = 1;
				setsockopt(candidateFd, IPPROTO_TCP, TCP_NODELAY, &noDelay, sizeof(noDelay));
				fd = candidateFd;
				break;
			}
			close(candidateFd);
		}
		freeaddrinfo(result);

		if (fd < 0) {
			std::ostringstream stream;
			stream << "connect failed";
			if (errno != 0) stream << ": " << std::strerror(errno);
			error = stream.str();
		}

		return fd >= 0;
	}
}
