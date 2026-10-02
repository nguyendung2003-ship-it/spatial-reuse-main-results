#ifndef __RUNTIME_TIMING_HH
#define __RUNTIME_TIMING_HH

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace RuntimeTiming {

inline std::string envOrEmpty(const char* name) {
	const char* value = std::getenv(name);
	return value == nullptr ? std::string() : std::string(value);
}

struct Record {
	std::string source;
	std::string topology;
	std::string method;
	std::string optimizer;
	std::string sampler;
	std::string hcmProfile;
	std::string seed;
	std::string surrogate;
	std::string backend;
	std::string architecture;
	std::string operation;
	std::uint64_t callIndex;
	std::uint64_t nObservations;
	std::uint64_t nCandidates;
	std::uint64_t elapsedNs;
};

class Recorder {
	public:
		void add(
			const std::string& source,
			const std::string& operation,
			std::uint64_t nObservations,
			std::uint64_t nCandidates,
			std::uint64_t elapsedNs) {
			if (envOrEmpty("NSTEST_CPP_TIMING_FILE").empty()) return;
			std::uint64_t callIndex = this->_counts[operation]++;
			this->_records.push_back({
				source,
				envOrEmpty("NSTEST_TOPO"),
				envOrEmpty("NSTEST_TIMING_METHOD"),
				envOrEmpty("NSTEST_OPTIMIZER"),
				envOrEmpty("NSTEST_SAMPLER"),
				envOrEmpty("NSTEST_HCM_PROFILE"),
				envOrEmpty("NSTEST_AGENT_SEED"),
				"",
				"cpp",
				"",
				operation,
				callIndex,
				nObservations,
				nCandidates,
				elapsedNs,
			});
		}

		~Recorder() {
			const std::string path = envOrEmpty("NSTEST_CPP_TIMING_FILE");
			if (path.empty() || this->_records.empty()) return;
			std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
			if (!output.is_open()) return;
			output << "source\ttopology\tmethod\toptimizer\tsampler\thcm_profile\tseed"
			       << "\tsurrogate\tbackend\tarchitecture\toperation\tcall_index"
			       << "\tn_obs\tenv_step\tn_candidates\telapsed_ns\n";
			for (const Record& record: this->_records) {
				output
					<< record.source << '\t'
					<< record.topology << '\t'
					<< record.method << '\t'
					<< record.optimizer << '\t'
					<< record.sampler << '\t'
					<< record.hcmProfile << '\t'
					<< record.seed << '\t'
					<< record.surrogate << '\t'
					<< record.backend << '\t'
					<< record.architecture << '\t'
					<< record.operation << '\t'
					<< record.callIndex << '\t'
					<< record.nObservations << '\t'
					<< record.nObservations << '\t'
					<< record.nCandidates << '\t'
					<< record.elapsedNs << '\n';
			}
		}

	private:
		std::vector<Record> _records;
		std::map<std::string, std::uint64_t> _counts;
};

inline Recorder& recorder() {
	static Recorder instance;
	return instance;
}

inline void record(
	const std::string& source,
	const std::string& operation,
	std::uint64_t nObservations,
	std::uint64_t nCandidates,
	std::uint64_t elapsedNs) {
	recorder().add(source, operation, nObservations, nCandidates, elapsedNs);
}

class Scoped {
	public:
		Scoped(
			const std::string& operation,
			std::uint64_t nObservations = 0,
			std::uint64_t nCandidates = 0,
			const std::string& source = "cpp")
			: _source(source),
			  _operation(operation),
			  _nObservations(nObservations),
			  _nCandidates(nCandidates),
			  _started(std::chrono::steady_clock::now()) { }

		~Scoped() {
			const std::chrono::steady_clock::time_point stopped =
				std::chrono::steady_clock::now();
			const std::uint64_t elapsed = static_cast<std::uint64_t>(
				std::chrono::duration_cast<std::chrono::nanoseconds>(
					stopped - this->_started).count());
			record(
				this->_source,
				this->_operation,
				this->_nObservations,
				this->_nCandidates,
				elapsed);
		}

	private:
		std::string _source;
		std::string _operation;
		std::uint64_t _nObservations;
		std::uint64_t _nCandidates;
		std::chrono::steady_clock::time_point _started;
};

}  // namespace RuntimeTiming

#endif
