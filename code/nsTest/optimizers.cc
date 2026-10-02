#include "optimizers.hh"
#include "bridge_utils.hh"
#include "runtime_timing.hh"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <unistd.h>

namespace {
	std::string optimizerEnvOrDefault(const char* name, const std::string& fallback) {
		const char* value = std::getenv(name);
		return value == nullptr || std::strlen(value) == 0 ? fallback : std::string(value);
	}

	bool optimizerEnvFlag(const char* name) {
		const char* value = std::getenv(name);
		if (value == nullptr || std::strlen(value) == 0) return false;
		std::string text(value);
		std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return text != "0" && text != "false" && text != "no";
	}

	unsigned int optimizerEnvUInt(const char* name, unsigned int fallback) {
		const char* value = std::getenv(name);
		if (value == nullptr || std::strlen(value) == 0) return fallback;
		return std::max(1, std::atoi(value));
	}

	double optimizerEnvDouble(const char* name, double fallback) {
		const char* value = std::getenv(name);
		if (value == nullptr || std::strlen(value) == 0) return fallback;
		char* end = nullptr;
		double parsed = std::strtod(value, &end);
		if (end == value || *end != '\0' || !std::isfinite(parsed)) return fallback;
		return parsed;
	}

	unsigned int optimizerSeed() {
		const char* value = std::getenv("NSTEST_AGENT_SEED");
		if (value != nullptr && std::strlen(value) > 0) {
			return std::max(1, std::atoi(value));
		}
		value = std::getenv("NSTEST_SEED");
		if (value != nullptr && std::strlen(value) > 0) {
			return std::max(1, std::atoi(value));
		}
		return static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
	}

	double clampUnitOptimizer(double value) {
		return std::max(0.0, std::min(1.0, value));
	}
}

/**
 * Build an Optimizer
 * 
 * @param sampler Sampler* the sampler to use to get new actions
 */
Optimizer::Optimizer(Sampler* sampler, unsigned int testPeriod): _sampler(sampler), _generator(std::default_random_engine(optimizerSeed())), _distribution(0.0, 1.0), _testPeriod(testPeriod) {  };

/**
 * Virtual destructor for Optimizer
 */
Optimizer::~Optimizer() {  }

void Optimizer::setSeed(unsigned int seed) {
	this->_generator.seed(seed);
}

/**
 * Add a configuration and its associated reward to the history.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated
 * @param forward bool whether to forward to sampler (optional, default: true)
 */
void Optimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	this->_history.push_back(std::make_tuple(configuration, reward));
	if (forward)
		this->_sampler->addToBase(configuration, reward);
}

bool Optimizer::readyForAnother() const { return true; }

unsigned int Optimizer::getTestPeriod() const { return this->_testPeriod; }

/**
 * Show the average reward and the number of times each configuration is played
 */
void Optimizer::showDecisions() const {
	std::map<NetworkConfiguration, double> rewards;
	std::map<NetworkConfiguration, double> counters;
	std::vector<NetworkConfiguration> keys;
	// Compute statistics
	for (std::tuple<NetworkConfiguration, double> t: this->_history) {
		NetworkConfiguration conf = std::get<0>(t);
		double r = std::get<1>(t);

		if (rewards.find(conf) != rewards.end()) {
			rewards[conf] = (counters[conf] * rewards[conf] + r) / (counters[conf] + 1);
			counters[conf]++;
		} else {
			rewards[conf] = r;
			counters[conf] = 1;
			keys.push_back(conf);
		}
	}

	// Log the stats
	std::cout << "Rewards: [";
	for (NetworkConfiguration conf: keys) {
		std::cout << " " << (round(1000.0*rewards[conf])/1000.0) << ",";
	}
	std::cout << " ]" << std::endl << "Counter: [";
	for (NetworkConfiguration conf: keys) {
		std::cout << " " << counters[conf] << ",";
	}
	std::cout << " ]" << std::endl;
}

RandomNeighborOptimizer::RandomNeighborOptimizer(Sampler* sampler) : Optimizer(sampler) {  }

void RandomNeighborOptimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	Optimizer::addToBase(configuration, reward, false);
	this->_chosen = configuration;
}

NetworkConfiguration RandomNeighborOptimizer::optimize() {
	if (this->_firstCollection) {
		if (this->_counter == 0) {
			this->_chosen = (*this->_sampler)();
		}
	} else {
		if (this->_counter == 0) {
			int idx = std::uniform_int_distribution<int>(0, this->_chosen.size() - 1)(this->_generator);
			const unsigned int idx2 = std::uniform_int_distribution<int>(0, 1)(this->_generator);
			int sign = std::uniform_int_distribution<int>(0, 1)(this->_generator) == 0 ? -1 : 1;
			if (idx2 == 0) {
				std::get<0>(this->_chosen[idx]) += sign;
			} else {
				std::get<1>(this->_chosen[idx]) += sign;
			}
		}
	}

	this->_counter += 1;
	if (this->_counter == this->_n) {
		this->_secondCollection = !this->_secondCollection;
		this->_firstCollection = !this->_firstCollection;
		this->_counter = 0;
	}

	return this->_chosen;
}

/**
 * Do nothing optimizer. Useful to study reward drifts on the simulator.
 */
IdleOptimizer::IdleOptimizer() : Optimizer(nullptr) {  }

/**
 * Add the configuration to the history but does not forward to an unused
 * sampler.
 * 
 * @param configuration the configuration tested
 * @param reward the reward obtained
 * @param forward whether to forward the data to the sampler
 */
void IdleOptimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	Optimizer::addToBase(configuration, reward, false);
	this->_chosen = configuration;
}

/**
 * As an idle optimizer, do nothing except chosing the previously chosen
 * configuration.
 * 
 * @return the previously chosen configuration
 */
NetworkConfiguration IdleOptimizer::optimize() {
	return this->_chosen;
}

/**
 * Build an EpsilonGreedyOptimizer
 * 
 * @param sampler Sampler* the sampler to use to get new actions
 * @param epsilon double the exploration parameter
 */
EpsilonGreedyOptimizer::EpsilonGreedyOptimizer(Sampler* sampler, double epsilon): Optimizer(sampler), _epsilon(epsilon) {  };


/**
 * Add an observation to the optimizer and its associated sampler.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated to the configuration 
 */
void EpsilonGreedyOptimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	Optimizer::addToBase(configuration, reward, forward);

	if (this->_results.find(configuration) != this->_results.end())
		this->_results[configuration] = 0.8 * this->_results[configuration] + 0.2 * reward;
	else
		this->_results[configuration] = reward;
}

/**
 * Find the optimal configuration according to e-greedy strategy.
 * 
 * @return the optimal configuration according to e-greedy strategy
 */
NetworkConfiguration EpsilonGreedyOptimizer::optimize() {
	// this->showDecisions();

	// Explore or exploit
	bool explore = this->_distribution(this->_generator) < this->_epsilon;
	if (explore) {
		NetworkConfiguration sampled = (*this->_sampler)();
		if (std::get<0>(sampled[0]) != 0)
			return sampled;
	}

	// Exploitation
	double max = 0;
	NetworkConfiguration confMax;
	for (std::map<NetworkConfiguration, double>::iterator it = this->_results.begin(); it != this->_results.end(); ++it) {
		double challenger = it->second;
		if (max < challenger) {
			max = challenger;
			confMax = it->first;
		}
	}

	return confMax;
}

/**
 * Build a ThompsonGammaNormalOptimizer
 * 
 * @param sampler Sampler* the sampler to draw new configurations from
 * @param sampleSize unsigned int the sample size to use for update
 * @param add double the exploration parameter
 */
ThompsonGammaNormalOptimizer::ThompsonGammaNormalOptimizer(Sampler* sampler, unsigned int sampleSize, double eps, unsigned int to_ig, bool chain)
	: Optimizer(sampler, sampleSize),
	  _testLeft(sampleSize),
	  _epsilon(eps),
	  _to_ig(to_ig),
	  _chain(chain),
	  _acquisitionEpsilon(optimizerEnvDouble("NSTEST_TGNORM_ACQUISITION_EPS", 0.0)),
	  _acquisitionPoolSize(optimizerEnvUInt("NSTEST_TGNORM_CANDIDATES", 512)),
	  _acquisitionNovelDraws(optimizerEnvUInt("NSTEST_TGNORM_NOVEL_CANDIDATES", 128)),
	  _acquisitionMaxAttempts(optimizerEnvUInt("NSTEST_TGNORM_CANDIDATE_ATTEMPTS", 50000)),
	  _acquisitionColdStart(optimizerEnvUInt("NSTEST_TGNORM_COLD_START", 20)),
	  _acquisitionDecisionCount(0) {
	if (!std::isfinite(this->_acquisitionEpsilon) || this->_acquisitionEpsilon < 0.0 || this->_acquisitionEpsilon > 1.0) {
		this->_acquisitionEpsilon = 0.0;
	}
}

bool ThompsonGammaNormalOptimizer::readyForAnother() const {
	return this->_testLeft == 0;
}


/**
 * Add an observation to the optimizer and its associated sampler.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated to the configuration 
 */
void ThompsonGammaNormalOptimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	if (this->_testLeft == 0) {
		this->_testLeft = this->_testPeriod;
		this->_chosen = configuration;
	}

	this->_testLeft--;
	bool ignore = this->_testLeft >= this->_testPeriod - this->_to_ig;
	if (!ignore)
		Optimizer::addToBase(configuration, reward, forward);

	if (this->_chosen.size() == 0)
		this->_chosen = configuration;

	// Search in attribute for a preexisting sample
	if (!ignore && this->_gammaNormals.find(configuration) != this->_gammaNormals.end()) {
		GammaNormalSample& lns = this->_gammaNormals[configuration]; 
		std::vector<double>& sample = std::get<4>(lns);
		sample.push_back(reward);
		// Update the gamma normal if we reach sample size
		unsigned int n = this->_testPeriod - this->_to_ig;
		if (sample.size() == n) {
			double mean = std::accumulate(sample.begin(), sample.end(), 0.0) / n;
			double var = n * (std::inner_product(sample.begin(), sample.end(), sample.begin(), 0.0) / n - mean * mean) / (n - 1);
			if (var <= 0) {
				var = 1e-9;
			}
			double &mu = std::get<0>(lns),
						 &lambda = std::get<1>(lns),
						 &alpha = std::get<2>(lns),
						 &beta = std::get<3>(lns),
						 newMu, newLambda, newAlpha, newBeta;
			bool& explore = std::get<5>(lns);
			if (explore) {
				newMu = mean;
				newLambda = n;
				newAlpha = n / 2.0;
				newBeta = n * var / 2.0;
				explore = false;
			} else {
				newMu = (lambda * mu + n * mean) / (lambda + n);
				newLambda = lambda + n;
				newAlpha = alpha + n / 2.0;
				newBeta = beta + (n * var + lambda * n * pow(mean - mu, 2) / (lambda + n)) / 2.0;
			}

			mu = newMu;
			lambda = newLambda;
			alpha = newAlpha;
			beta = newBeta;
			sample.clear();
		}
 	} else {
		 // Create a new instance in logNormals
		this->_gammaNormals[configuration] = std::make_tuple(0.5, 1.0, 0.5, 0.025, std::vector<double>({reward}), true);
	}
}

/**
 * Find the best configuration according to ThompsonGammaNormal strategy
 * 
 * @return the best configuration according to ThompsonGammaNormal strategy 
 */
NetworkConfiguration ThompsonGammaNormalOptimizer::optimize() {
	// this->showDecisions();

	// unsigned _seed = std::chrono::system_clock::now().time_since_epoch().count();
  // std::default_random_engine generator(_seed);
  // std::uniform_real_distribution<double> distribution(0.0, 1.0);

	if (this->_chain && this->_testLeft > 0) {
		return this->_chosen;
	}

	if (this->_sampler->hasForcedAction()) {
		std::vector<NetworkConfiguration> forbidden;
		for (const auto& item: this->_gammaNormals) forbidden.push_back(item.first);
		NetworkConfiguration forced = this->_sampler->takeForcedAction(forbidden);
		if (!forced.empty() && std::get<0>(forced[0]) != 0) {
			this->_chosen = forced;
			this->_testLeft = this->_testPeriod;
			this->_acquisitionDecisionCount++;
			return this->_chosen;
		}
	}

	// Optional candidate-pool exploration floor matching the Neural/QNN
	// controller semantics. This is separate from per-arm TS epsilon and the
	// HCM sampler's global proposal probability.
	if (this->_acquisitionEpsilon > 0.0 &&
			this->_acquisitionDecisionCount >= this->_acquisitionColdStart &&
			this->_distribution(this->_generator) < this->_acquisitionEpsilon) {
		std::vector<NetworkConfiguration> forbidden;
		for (const auto& item: this->_gammaNormals) forbidden.push_back(item.first);
		unsigned int historicalCount = std::min<unsigned int>(
			static_cast<unsigned int>(forbidden.size()), this->_acquisitionPoolSize);
		unsigned int room = this->_acquisitionPoolSize - historicalCount;
		unsigned int novelTarget = std::min(room, this->_acquisitionNovelDraws);
		if (novelTarget > 0) {
			std::vector<NetworkConfiguration> unseen = this->_sampler->randomActionsFromSample(
				novelTarget, this->_acquisitionMaxAttempts, forbidden);
			if (!unseen.empty()) {
				std::uniform_int_distribution<std::size_t> pick(0, unseen.size() - 1);
				this->_chosen = unseen[pick(this->_generator)];
				this->_testLeft = this->_testPeriod;
				this->_sampler->notifyDecision(this->_chosen);
				std::cout << "tgnorm-acquisition-floor selected decision="
							<< this->_acquisitionDecisionCount + 1
							<< " unseen_candidates=" << unseen.size()
							<< " epsilon=" << this->_acquisitionEpsilon << std::endl;
				this->_acquisitionDecisionCount++;
				return this->_chosen;
			}
		}
	}

	// Explore or exploit
	NetworkConfiguration newConf;
	bool confChosen = false;
	bool explore = this->_distribution(this->_generator) < (this->_chain ? 1 - pow(1 - this->_epsilon, this->_testPeriod) : this->_epsilon);
	if (explore) {
		// Request a new configuration to the sampler
		std::vector<NetworkConfiguration> forbidden;
		for (std::map<NetworkConfiguration, GammaNormalSample>::iterator it = this->_gammaNormals.begin(); it != this->_gammaNormals.end(); ++it)
			forbidden.push_back(it->first);
		NetworkConfiguration sampled = (*this->_sampler)(forbidden);
		if (std::get<0>(sampled[0]) != 0) {
			confChosen = true;
			newConf = sampled;
		}

		// std::cout << "Configuration: ";
		// for (std::tuple<double, double> t: newConf) {
		// 	std::cout << '(' << std::get<0>(t) << "," << std::get<1>(t) << "),"; 
		// }
		// std::cout << std::endl;
	}

	// Look for not enough explored configurations
	if (!confChosen) {
		std::vector<NetworkConfiguration> toExplore;
		for (std::map<NetworkConfiguration, GammaNormalSample>::iterator it = this->_gammaNormals.begin(); it != this->_gammaNormals.end(); ++it)
			if (std::get<5>(it->second))
				toExplore.push_back(it->first);
		if (!toExplore.empty()) {
			// Test a not enough explored configuration
			std::uniform_int_distribution<> d(0, toExplore.size()-1);
			newConf = toExplore[d(this->_generator)];
		} else {
			// Sample taus for normal distributions
			std::vector<double> mus(this->_gammaNormals.size());
			int i = 0;
			for (std::map<NetworkConfiguration, GammaNormalSample>::iterator it = this->_gammaNormals.begin(); it != this->_gammaNormals.end(); ++it) {
				std::gamma_distribution<> gamma(std::get<2>(it->second), 1.0 / std::get<3>(it->second));
				double tau = gamma(this->_generator);
				std::normal_distribution<> normal(std::get<0>(it->second), 1.0 / sqrt(std::get<1>(it->second) * tau));
				mus[i] = normal(this->_generator);
				i++;
			}
			int maxElementIndex = std::max_element(mus.begin(), mus.end()) - mus.begin();
			std::map<NetworkConfiguration, GammaNormalSample>::iterator maxConf = this->_gammaNormals.begin();
			for (int i = 0; i < maxElementIndex; i++) maxConf++;

			newConf = maxConf->first;
		}
	}

	this->_chosen = newConf;
	this->_testLeft = this->_testPeriod;
	this->_sampler->notifyDecision(this->_chosen);
	this->_acquisitionDecisionCount++;

	// for (std::tuple<double, double> t: this->_chosen) {
	// 	std::cout << '(' << std::get<0>(t) << "," << std::get<1>(t) << "),"; 
	// }
	// std::cout << std::endl;

	return this->_chosen;
}

NeuralBanditOptimizer::NeuralBanditOptimizer(Sampler* sampler, unsigned int sampleSize, double eps, unsigned int to_ig, bool chain, unsigned int seed)
	: Optimizer(sampler, sampleSize),
		_fallback(sampler, sampleSize, eps, to_ig, chain),
		_testLeft(sampleSize),
		_epsilon(eps),
		_to_ig(to_ig),
		_chain(chain),
				_bridgeWarned(false),
				_host(optimizerEnvOrDefault("NSTEST_NB_HOST", "127.0.0.1")),
				_port(optimizerEnvOrDefault("NSTEST_NB_PORT", "9877")),
				_candidatePoolSize(optimizerEnvUInt("NSTEST_NB_CANDIDATES", 2048)),
				_candidateMaxAttempts(optimizerEnvUInt("NSTEST_NB_CANDIDATE_ATTEMPTS", 50000)),
				_candidateNovelDraws(optimizerEnvUInt("NSTEST_NB_NOVEL_CANDIDATES", 32)),
				_guideWithFallback(optimizerEnvFlag("NSTEST_NB_GUIDE_FALLBACK") ||
												 optimizerEnvOrDefault("NSTEST_NB_GUIDE_FALLBACK", "1") == "1" ||
												 optimizerEnvOrDefault("NSTEST_NB_GUIDE_FALLBACK", "1") == "true"),
				_fallbackColdStartObservations(optimizerEnvUInt("NSTEST_NB_COLD_START", 20)),
				_directConfigMode(optimizerEnvFlag("NSTEST_NB_DIRECT") ||
											 optimizerEnvOrDefault("NSTEST_NB_MODE", "CHOICE") == "CONFIG" ||
											 optimizerEnvOrDefault("NSTEST_NB_MODE", "CHOICE") == "config"),
				_rawObserveMode(optimizerEnvFlag("NSTEST_NB_RAW_OBSERVE")),
				_nbObservationCount(0),
			_hasBestConfiguration(false),
			_bestReward(-1.0e300),
			_seed(seed),
			_bridgeConnectCount(0),
			_runtimeStep(0),
			_bridgeFd(-1),
			_traceEnabled(optimizerEnvFlag("NSTEST_NB_TRACE")),
			_tracePath(optimizerEnvOrDefault("NSTEST_NB_TRACE_FILE", "")),
			_traceStep(0) {
			if (this->_rawObserveMode) {
				this->_testPeriod = optimizerEnvUInt("NSTEST_NB_TEST_PERIOD", sampleSize);
				this->_testLeft = this->_testPeriod;
				if (std::getenv("NSTEST_NB_CHAIN") != nullptr)
					this->_chain = optimizerEnvFlag("NSTEST_NB_CHAIN");
			}
			this->setSeed(seed + 104729u);
			this->_fallback.setSeed(seed + 130363u);
			RandomSampler* randomSampler = dynamic_cast<RandomSampler*>(sampler);
			if (randomSampler != nullptr) randomSampler->setSeed(seed + 155921u);
			if (this->_traceEnabled) {
				if (this->_tracePath.empty()) {
					std::ostringstream path;
					path << "./scratch/nsTest/data/nb_trace_seed" << this->_seed
							 << "_pid" << getpid() << ".tsv";
				this->_tracePath = path.str();
			}
				this->_traceFile.open(this->_tracePath, std::ios::out | std::ios::app);
				if (!this->_traceFile.is_open()) {
					std::cerr << "ns3-nb: could not open trace file '" << this->_tracePath << "'" << std::endl;
					this->_traceEnabled = false;
				} else {
					this->_traceFile << "# seed=" << this->_seed << "\tpid=" << getpid()
														 << "\tprotocol=" << (this->_directConfigMode ? "ASK_CONFIG" : "ASK_CHOICE") << std::endl;
					this->_traceFile << "event\tdecision\tcandidate_index\tchosen\treward\tconfiguration" << std::endl;
					this->_traceFile.flush();
				}
			}
			if (this->nbBridgeEnabled()) this->ensureBridgeConnected();
		}

	NeuralBanditOptimizer::~NeuralBanditOptimizer() {
		if (this->_traceFile.is_open()) this->_traceFile.close();
		this->closeBridge();
	}

bool NeuralBanditOptimizer::nbBridgeEnabled() const {
	return optimizerEnvFlag("NSTEST_NB_BRIDGE");
}

bool NeuralBanditOptimizer::ensureBridgeConnected() {
	if (!this->nbBridgeEnabled()) return false;
	if (this->_bridgeFd >= 0) return true;

	std::string error;
	if (!BridgeUtils::connectTcp(this->_host, this->_port, this->_bridgeFd, error)) {
		if (!this->_bridgeWarned) {
			std::cerr << "ns3-nb: cannot connect to " << this->_host << ":" << this->_port
								<< " (" << error << "); falling back to GM-TS" << std::endl;
			this->_bridgeWarned = true;
		}
		return false;
	}

	if (!this->sendHello()) {
		this->closeBridge();
		if (!this->_bridgeWarned) {
			std::cerr << "ns3-nb: connected to " << this->_host << ":" << this->_port
								<< " but HELLO failed; falling back to GM-TS" << std::endl;
			this->_bridgeWarned = true;
		}
		return false;
	}

	if (this->_bridgeConnectCount > 0) {
		std::cerr << "ns3-nb: reconnected to neural bandit bridge; Python bandit state was reset with seed "
							<< this->_seed << " while local GM-TS fallback kept its history" << std::endl;
	}
	this->_bridgeConnectCount++;
	this->_bridgeWarned = false;
	std::cerr << "ns3-nb: connected to neural bandit bridge at "
						<< this->_host << ":" << this->_port << " (seed " << this->_seed << ")" << std::endl;
	return true;
}

void NeuralBanditOptimizer::closeBridge() {
	if (this->_bridgeFd >= 0) {
		close(this->_bridgeFd);
		this->_bridgeFd = -1;
	}
}

bool NeuralBanditOptimizer::sendHello() {
	std::ostringstream payload;
	payload << "HELLO " << this->_seed << " "
				<< 2 * this->_sampler->_parameters.size() << "\n";
	return BridgeUtils::sendAll(this->_bridgeFd, payload.str());
}

bool NeuralBanditOptimizer::readyForAnother() const {
	return this->_testLeft == 0;
}

std::vector<double> NeuralBanditOptimizer::normalizedConfiguration(const NetworkConfiguration& configuration) const {
	std::vector<double> values;
	values.reserve(2 * configuration.size());
	for (unsigned int i = 0; i < configuration.size(); i++) {
		if (i < this->_sampler->_parameters.size()) {
			std::tuple<double, double> normalized = this->_sampler->_parameters[i].normalize(configuration[i]);
			values.push_back(clampUnitOptimizer(std::get<0>(normalized)));
			values.push_back(clampUnitOptimizer(std::get<1>(normalized)));
		} else {
			values.push_back(clampUnitOptimizer((std::get<0>(configuration[i]) + 82.0) / 20.0));
			values.push_back(clampUnitOptimizer((std::get<1>(configuration[i]) - 1.0) / 20.0));
		}
	}

	return values;
}

NetworkConfiguration NeuralBanditOptimizer::configurationFromNormalized(const std::vector<double>& values) const {
	NetworkConfiguration configuration;
	if (values.size() % 2 != 0) return configuration;

	configuration.reserve(values.size() / 2);
	for (unsigned int i = 0; i < values.size(); i += 2) {
		double sensitivityNorm = clampUnitOptimizer(values[i]);
		double txPowerNorm = clampUnitOptimizer(values[i + 1]);
		if (sensitivityNorm + txPowerNorm > 0.95) {
			txPowerNorm = std::max(0.0, 0.95 - sensitivityNorm);
		}

		std::tuple<double, double> decoded;
		unsigned int paramIndex = i / 2;
		if (paramIndex < this->_sampler->_parameters.size()) {
			decoded = this->_sampler->_parameters[paramIndex].fromNormalized(std::make_tuple(sensitivityNorm, txPowerNorm));
		} else {
			decoded = std::make_tuple(-82.0 + 20.0 * sensitivityNorm, 1.0 + 20.0 * txPowerNorm);
		}

		double sensitivity = std::get<0>(decoded);
		double txPower = std::get<1>(decoded);
		double maxSensitivity = std::max(-82.0, std::min(-62.0, -82.0 + (20.0 - txPower)));
		if (sensitivity > maxSensitivity) sensitivity = maxSensitivity;
		configuration.push_back(std::make_tuple(sensitivity, txPower));
	}

	if (!confConstraint(configuration)) configuration.clear();
	return configuration;
}

std::string NeuralBanditOptimizer::traceConfiguration(const NetworkConfiguration& configuration) const {
	std::ostringstream stream;
	stream << std::setprecision(12);
	for (unsigned int i = 0; i < configuration.size(); i++) {
		if (i > 0) stream << ";";
		stream << std::get<0>(configuration[i]) << "," << std::get<1>(configuration[i]);
	}
	return stream.str();
}

void NeuralBanditOptimizer::traceDecision(const NetworkConfiguration& configuration) {
	if (!this->_traceEnabled || !this->_traceFile.is_open()) return;
	unsigned int decision = this->_traceStep++;
	this->_traceFile << std::setprecision(12);
	this->_traceFile << "CONFIG\t" << decision << "\t-1\t1\t\t"
									 << this->traceConfiguration(configuration) << std::endl;
	this->_traceFile.flush();
}

void NeuralBanditOptimizer::traceObservation(const NetworkConfiguration& configuration, double reward) {
	if (!this->_traceEnabled || !this->_traceFile.is_open()) return;
	this->_traceFile << std::setprecision(12)
									 << "OBSERVE\t" << this->_traceStep << "\t-1\t1\t" << reward
									 << "\t" << this->traceConfiguration(configuration) << std::endl;
	this->_traceFile.flush();
}

void NeuralBanditOptimizer::traceFallback(const NetworkConfiguration& configuration) {
	if (!this->_traceEnabled || !this->_traceFile.is_open()) return;
	this->_traceFile << "FALLBACK\t" << this->_traceStep++ << "\t-1\t1\t\t"
									 << this->traceConfiguration(configuration) << std::endl;
	this->_traceFile.flush();
}

	NetworkConfiguration NeuralBanditOptimizer::referenceConfiguration() const {
		if (this->_hasBestConfiguration) return this->_bestConfiguration;
		return this->_chosen;
	}

	bool NeuralBanditOptimizer::addCandidate(const NetworkConfiguration& candidate, std::vector<NetworkConfiguration>& candidates) const {
		if (candidate.size() != this->_sampler->_parameters.size()) return false;
		if (!confConstraint(candidate)) return false;
		if (std::find(candidates.begin(), candidates.end(), candidate) != candidates.end()) return false;
		candidates.push_back(candidate);
		return true;
	}

	std::vector<NetworkConfiguration> NeuralBanditOptimizer::buildCandidatePool() {
		std::vector<NetworkConfiguration> candidates;
		if (this->_hasBestConfiguration) this->addCandidate(this->_bestConfiguration, candidates);
		if (!this->_chosen.empty()) this->addCandidate(this->_chosen, candidates);

		for (std::map<NetworkConfiguration, std::vector<double>>::const_iterator it = this->_pendingRewards.begin();
				 it != this->_pendingRewards.end() && candidates.size() < this->_candidatePoolSize;
				 ++it) {
			this->addCandidate(it->first, candidates);
		}

		for (History::const_iterator it = this->_history.begin();
				 it != this->_history.end() && candidates.size() < this->_candidatePoolSize;
				 ++it) {
			this->addCandidate(std::get<0>(*it), candidates);
		}

		unsigned int room = this->_candidatePoolSize > candidates.size() ? this->_candidatePoolSize - candidates.size() : 0;
		unsigned int novelTarget = std::min(room, this->_candidateNovelDraws);
		if (novelTarget > 0) {
			std::vector<NetworkConfiguration> novel = this->_sampler->randomActionsFromSample(novelTarget, this->_candidateMaxAttempts, candidates);
			for (const NetworkConfiguration& sampled: novel) {
				this->addCandidate(sampled, candidates);
			}
		}

		return candidates;
	}

bool NeuralBanditOptimizer::sendObserve(const NetworkConfiguration& configuration, double reward) {
	if (!this->ensureBridgeConnected()) return false;

	std::vector<double> normalized = this->normalizedConfiguration(configuration);
	std::ostringstream payload;
	payload << std::setprecision(12);
	payload << "STEP " << this->_runtimeStep << "\n";
	payload << "OBSERVE " << normalized.size() << " " << reward;
	for (double value: normalized) payload << " " << value;
	payload << "\n";

	if (!BridgeUtils::sendAll(this->_bridgeFd, payload.str())) {
		std::cerr << "ns3-nb: bridge write failed during OBSERVE; closing connection and falling back to GM-TS" << std::endl;
		this->closeBridge();
		return false;
	}

	return true;
}

	bool NeuralBanditOptimizer::askConfigFromBridge(NetworkConfiguration& chosen) {
		if (!this->ensureBridgeConnected()) return false;

		unsigned int dim = 2 * this->_sampler->_parameters.size();
		std::ostringstream payload;
		payload << std::setprecision(12);
		payload << "STEP " << this->_runtimeStep << "\n";
		payload << "ASK_CONFIG " << dim << "\n";

		if (!this->_chosen.empty()) {
			std::vector<double> current = this->normalizedConfiguration(this->_chosen);
			payload << "CURRENT";
			for (double value: current) payload << " " << value;
			payload << "\n";
		}

		if (this->_hasBestConfiguration) {
			std::vector<double> best = this->normalizedConfiguration(this->_bestConfiguration);
			payload << "BEST";
			for (double value: best) payload << " " << value;
			payload << "\n";
		}
		payload << "END\n";

		if (!BridgeUtils::sendAll(this->_bridgeFd, payload.str())) {
			std::cerr << "ns3-nb: bridge write failed during ASK_CONFIG; closing connection and falling back to GM-TS" << std::endl;
			this->closeBridge();
			return false;
		}

		std::string response;
		if (!BridgeUtils::receiveLine(this->_bridgeFd, response)) {
			std::cerr << "ns3-nb: bridge read failed during ASK_CONFIG; closing connection and falling back to GM-TS" << std::endl;
			this->closeBridge();
			return false;
		}

		std::istringstream parser(response);
		std::string token;
		unsigned int responseDim = 0;
		parser >> token >> responseDim;
		if (token != "CONFIG" || responseDim != dim) {
			std::cerr << "ns3-nb: invalid bridge response '" << response
								<< "'; falling back to GM-TS" << std::endl;
			return false;
		}

		std::vector<double> values;
		values.reserve(dim);
		for (unsigned int i = 0; i < dim; i++) {
			double value = 0.0;
			if (!(parser >> value)) {
				std::cerr << "ns3-nb: CONFIG response had too few values; falling back to GM-TS" << std::endl;
				return false;
			}
			values.push_back(value);
		}

		chosen = this->configurationFromNormalized(values);
		if (chosen.empty()) {
			std::cerr << "ns3-nb: CONFIG response decoded to an invalid configuration; falling back to GM-TS" << std::endl;
			return false;
		}
		return true;
	}

	bool NeuralBanditOptimizer::askChoiceFromBridge(NetworkConfiguration& chosen) {
		if (!this->ensureBridgeConnected()) return false;

		unsigned int dim = 2 * this->_sampler->_parameters.size();
		std::chrono::steady_clock::time_point candidateStarted = std::chrono::steady_clock::now();
		std::vector<NetworkConfiguration> candidates = this->buildCandidatePool();
		RuntimeTiming::record(
			"cpp",
			"candidate_generation",
			this->_runtimeStep,
			candidates.size(),
			static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now() - candidateStarted).count()));
		if (candidates.empty()) {
			std::cerr << "ns3-nb: could not build a valid sampler candidate pool; falling back to GM-TS" << std::endl;
			return false;
		}

		std::ostringstream payload;
		payload << std::setprecision(12);
		payload << "STEP " << this->_runtimeStep << "\n";
		payload << "ASK_CHOICE " << dim << " " << candidates.size() << "\n";

		if (!this->_chosen.empty()) {
			std::vector<double> current = this->normalizedConfiguration(this->_chosen);
		payload << "CURRENT";
		for (double value: current) payload << " " << value;
		payload << "\n";
	}

	if (this->_hasBestConfiguration) {
		std::vector<double> best = this->normalizedConfiguration(this->_bestConfiguration);
		payload << "BEST";
			for (double value: best) payload << " " << value;
			payload << "\n";
		}
		for (const NetworkConfiguration& candidate: candidates) {
			std::vector<double> normalized = this->normalizedConfiguration(candidate);
			payload << "CANDIDATE";
			for (double value: normalized) payload << " " << value;
			payload << "\n";
		}
		payload << "END\n";

		if (!BridgeUtils::sendAll(this->_bridgeFd, payload.str())) {
			std::cerr << "ns3-nb: bridge write failed during ASK_CHOICE; closing connection and falling back to GM-TS" << std::endl;
			this->closeBridge();
			return false;
		}

		std::string response;
		if (!BridgeUtils::receiveLine(this->_bridgeFd, response)) {
			std::cerr << "ns3-nb: bridge read failed during ASK_CHOICE; closing connection and falling back to GM-TS" << std::endl;
			this->closeBridge();
			return false;
		}

		std::istringstream parser(response);
		std::string token;
		unsigned int chosenIndex = 0;
		parser >> token >> chosenIndex;
		if (token != "CHOICE") {
			std::cerr << "ns3-nb: invalid bridge response '" << response
								<< "'; falling back to GM-TS" << std::endl;
			return false;
		}
		if (chosenIndex >= candidates.size()) {
			std::cerr << "ns3-nb: bridge returned CHOICE " << chosenIndex
								<< " outside candidate pool of size " << candidates.size()
								<< "; falling back to GM-TS" << std::endl;
			return false;
		}

		chosen = candidates[chosenIndex];
		return true;
	}

void NeuralBanditOptimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	this->_runtimeStep++;
	std::chrono::steady_clock::time_point localStarted = std::chrono::steady_clock::now();
	auto recordLocalUpdate = [&]() {
		RuntimeTiming::record(
			"cpp",
			"algorithm_update_local",
			this->_runtimeStep,
			0,
			static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now() - localStarted).count()));
	};
	if (this->_testLeft == 0) {
		this->_testLeft = this->_testPeriod;
		this->_chosen = configuration;
	}

	this->_testLeft--;
	bool ignore = this->_testLeft >= this->_testPeriod - this->_to_ig;
	if (!this->_rawObserveMode)
		this->_fallback.addToBase(configuration, reward, forward, individual_rewards);

	if (this->_chosen.size() == 0)
		this->_chosen = configuration;

	if (this->_rawObserveMode) {
		recordLocalUpdate();
		this->sendObserve(configuration, reward);
		this->_nbObservationCount++;
		return;
	}

	if (ignore) {
		recordLocalUpdate();
		return;
	}

	std::vector<double>& sample = this->_pendingRewards[configuration];
	sample.push_back(reward);
	unsigned int n = this->_testPeriod - this->_to_ig;
	if (sample.size() >= n) {
		double mean = std::accumulate(sample.begin(), sample.end(), 0.0) / sample.size();
		this->_history.push_back(std::make_tuple(configuration, mean));
		std::pair<double, unsigned int>& aggregate = this->_configurationRewardStats[configuration];
		aggregate.first += mean;
		aggregate.second++;
		this->_hasBestConfiguration = false;
		this->_bestReward = -1.0e300;
		for (const auto& item: this->_configurationRewardStats) {
			double averageReward = item.second.first / item.second.second;
			if (!this->_hasBestConfiguration || averageReward > this->_bestReward) {
				this->_bestReward = averageReward;
				this->_bestConfiguration = item.first;
				this->_hasBestConfiguration = true;
			}
		}
		recordLocalUpdate();
		this->traceObservation(configuration, mean);
		this->sendObserve(configuration, mean);
		this->_nbObservationCount++;
		sample.clear();
		return;
	}
	recordLocalUpdate();
}

NetworkConfiguration NeuralBanditOptimizer::fallbackOptimize() {
	std::chrono::steady_clock::time_point localStarted = std::chrono::steady_clock::now();
	this->_chosen = this->_fallback.optimize();
	this->_testLeft = this->_testPeriod;
	RuntimeTiming::record(
		"cpp",
		"algorithm_decision_local",
		this->_runtimeStep,
		0,
		static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now() - localStarted).count()));
	this->traceFallback(this->_chosen);
	return this->_chosen;
}

NetworkConfiguration NeuralBanditOptimizer::optimize() {
	std::chrono::steady_clock::time_point localStarted = std::chrono::steady_clock::now();
	auto recordLocalDecision = [&](std::size_t candidates = 0) {
		RuntimeTiming::record(
			"cpp",
			"algorithm_decision_local",
			this->_runtimeStep,
			candidates,
			static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now() - localStarted).count()));
	};
	if (this->_chain && this->_testLeft > 0) {
		recordLocalDecision();
		return this->_chosen;
	}
	if (this->_sampler->hasForcedAction()) {
		std::vector<NetworkConfiguration> forbidden;
		for (const auto& item: this->_configurationRewardStats) forbidden.push_back(item.first);
		NetworkConfiguration forced = this->_sampler->takeForcedAction(forbidden);
		if (!forced.empty() && std::get<0>(forced[0]) != 0) {
			this->_chosen = forced;
			this->_testLeft = this->_testPeriod;
			recordLocalDecision();
			this->traceDecision(forced);
			return this->_chosen;
		}
	}
	recordLocalDecision();
	if (this->_guideWithFallback && this->_nbObservationCount < this->_fallbackColdStartObservations) {
		return this->fallbackOptimize();
	}

	if (!this->ensureBridgeConnected()) {
		return this->fallbackOptimize();
	}

	NetworkConfiguration chosen;
	bool ok = this->_directConfigMode ? this->askConfigFromBridge(chosen) : this->askChoiceFromBridge(chosen);
	if (!ok) {
		return this->fallbackOptimize();
	}

	localStarted = std::chrono::steady_clock::now();
	this->_sampler->selectedAction(chosen);
	this->_sampler->notifyDecision(chosen);
	this->_chosen = chosen;
	this->_testLeft = this->_testPeriod;
	recordLocalDecision();
	this->traceDecision(chosen);
	return this->_chosen;
}

InspireOptimizer::InspireOptimizer(
	Sampler* sampler,
	const std::vector<std::vector<unsigned int>>& neighborhoods,
	unsigned int seed)
	: Optimizer(sampler, 1),
		_neighborhoods(neighborhoods),
		_localHistories(neighborhoods.size()),
		_rng(seed == 0 ? 1 : seed),
		_window(optimizerEnvUInt("NSTEST_INSPIRE_WINDOW", 32)),
		_restarts(optimizerEnvUInt("NSTEST_INSPIRE_RESTARTS", 6)),
		_sweeps(optimizerEnvUInt("NSTEST_INSPIRE_SWEEPS", 3)),
		_hyperPeriod(optimizerEnvUInt("NSTEST_INSPIRE_HYPER_PERIOD", 10)),
		_lengthScales(neighborhoods.size(), 0.35),
		_step(0),
		_traceEnabled(optimizerEnvFlag("NSTEST_INSPIRE_TRACE")),
		_tracePath(optimizerEnvOrDefault("NSTEST_INSPIRE_TRACE_FILE", "")) {
	this->openTrace();
}

InspireOptimizer::~InspireOptimizer() {
	if (this->_traceFile.is_open()) this->_traceFile.close();
}

void InspireOptimizer::openTrace() {
	if (!this->_traceEnabled) return;
	if (this->_tracePath.empty()) {
		std::ostringstream path;
		path << "/tmp/ns3_inspire_trace_" << getpid() << ".tsv";
		this->_tracePath = path.str();
	}
	this->_traceFile.open(this->_tracePath.c_str(), std::ios::out | std::ios::trunc);
	if (this->_traceFile.is_open()) {
		this->_traceFile << "step\tagent\tneighborhood\tprescription\tconsensus\n";
	} else {
		std::cerr << "ns3-inspire: cannot open trace file " << this->_tracePath << std::endl;
		this->_traceEnabled = false;
	}
}

std::vector<double> InspireOptimizer::localConfiguration(
	unsigned int agent,
	const NetworkConfiguration& configuration) const {
	std::vector<double> result;
	if (agent >= this->_neighborhoods.size()) return result;
	for (unsigned int member: this->_neighborhoods[agent]) {
		if (member >= configuration.size()) continue;
		result.push_back(clampUnitOptimizer((std::get<0>(configuration[member]) + 82.0) / 20.0));
		result.push_back(clampUnitOptimizer((std::get<1>(configuration[member]) - 1.0) / 20.0));
	}
	return result;
}

bool InspireOptimizer::validPair(double sensitivity, double power) const {
	return sensitivity >= -82.0 && sensitivity <= -62.0 &&
		power >= 1.0 && power <= 21.0 &&
		sensitivity + power <= -62.0;
}

NetworkConfiguration InspireOptimizer::randomLocalConfiguration(unsigned int agent) {
	NetworkConfiguration result;
	if (agent >= this->_neighborhoods.size()) return result;
	std::uniform_int_distribution<int> sensitivityDistribution(-82, -62);
	for (unsigned int ignored: this->_neighborhoods[agent]) {
		(void) ignored;
		int sensitivity = sensitivityDistribution(this->_rng);
		int maximumPower = std::max(1, std::min(21, -62 - sensitivity));
		std::uniform_int_distribution<int> powerDistribution(1, maximumPower);
		result.push_back(std::make_tuple(sensitivity, powerDistribution(this->_rng)));
	}
	return result;
}

double InspireOptimizer::matern32(
	const std::vector<double>& lhs,
	const std::vector<double>& rhs,
	double lengthScale) const {
	double squaredDistance = 0.0;
	for (unsigned int i = 0; i < lhs.size() && i < rhs.size(); i++) {
		double difference = lhs[i] - rhs[i];
		squaredDistance += difference * difference;
	}
	double scaledDistance = std::sqrt(3.0 * squaredDistance) / std::max(1e-9, lengthScale);
	return (1.0 + scaledDistance) * std::exp(-scaledDistance);
}

bool InspireOptimizer::choleskyDecompose(std::vector<std::vector<double>>& matrix) const {
	const unsigned int n = matrix.size();
	for (unsigned int row = 0; row < n; row++) {
		for (unsigned int column = 0; column <= row; column++) {
			double value = matrix[row][column];
			for (unsigned int k = 0; k < column; k++) value -= matrix[row][k] * matrix[column][k];
			if (row == column) {
				if (!std::isfinite(value) || value <= 1e-12) return false;
				matrix[row][column] = std::sqrt(value);
			} else {
				matrix[row][column] = value / matrix[column][column];
			}
		}
		for (unsigned int column = row + 1; column < n; column++) matrix[row][column] = 0.0;
	}
	return true;
}

std::vector<double> InspireOptimizer::choleskySolve(
	const std::vector<std::vector<double>>& lower,
	const std::vector<double>& rhs) const {
	const unsigned int n = lower.size();
	std::vector<double> intermediate(n, 0.0), result(n, 0.0);
	for (unsigned int row = 0; row < n; row++) {
		double value = row < rhs.size() ? rhs[row] : 0.0;
		for (unsigned int column = 0; column < row; column++) value -= lower[row][column] * intermediate[column];
		intermediate[row] = value / lower[row][row];
	}
	for (int row = static_cast<int>(n) - 1; row >= 0; row--) {
		double value = intermediate[row];
		for (unsigned int column = row + 1; column < n; column++) value -= lower[column][row] * result[column];
		result[row] = value / lower[row][row];
	}
	return result;
}

InspireOptimizer::GaussianProcessModel InspireOptimizer::fitModel(unsigned int agent) {
	GaussianProcessModel model;
	if (agent >= this->_localHistories.size() || this->_localHistories[agent].empty()) return model;
	const std::vector<LocalObservation>& history = this->_localHistories[agent];
	unsigned int begin = history.size() > this->_window ? history.size() - this->_window : 0;
	for (unsigned int index = begin; index < history.size(); index++) {
		model.features.push_back(history[index].configuration);
		model.labels.push_back(history[index].reward);
	}

	const bool tuneHyperparameters = model.features.size() <= 2 || this->_step % this->_hyperPeriod == 0;
	const std::vector<double> candidates = tuneHyperparameters
		? std::vector<double>({0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00})
		: std::vector<double>({this->_lengthScales[agent]});
	double bestLikelihood = -std::numeric_limits<double>::infinity();
	for (double lengthScale: candidates) {
		const unsigned int n = model.features.size();
		std::vector<std::vector<double>> lower(n, std::vector<double>(n, 0.0));
		for (unsigned int row = 0; row < n; row++) {
			for (unsigned int column = 0; column <= row; column++) {
				lower[row][column] = this->matern32(model.features[row], model.features[column], lengthScale);
				if (row == column) lower[row][column] += 1e-6;
				lower[column][row] = lower[row][column];
			}
		}
		if (!this->choleskyDecompose(lower)) continue;
		std::vector<double> alpha = this->choleskySolve(lower, model.labels);
		double quadratic = std::inner_product(model.labels.begin(), model.labels.end(), alpha.begin(), 0.0);
		double signalVariance = std::max(1e-9, quadratic / std::max<unsigned int>(1, n));
		double logDeterminantCorrelation = 0.0;
		for (unsigned int index = 0; index < n; index++) logDeterminantCorrelation += 2.0 * std::log(lower[index][index]);
		double likelihood = -0.5 * (n * std::log(signalVariance) + logDeterminantCorrelation + n);
		if (likelihood > bestLikelihood) {
			bestLikelihood = likelihood;
			model.lengthScale = lengthScale;
			model.signalVariance = signalVariance;
			model.cholesky = lower;
			model.alpha = alpha;
			model.valid = true;
		}
	}
	if (model.valid) this->_lengthScales[agent] = model.lengthScale;
	return model;
}

std::pair<double, double> InspireOptimizer::predict(
	const GaussianProcessModel& model,
	const std::vector<double>& configuration) const {
	if (!model.valid) return std::make_pair(0.0, 1.0);
	std::vector<double> correlations(model.features.size(), 0.0);
	for (unsigned int index = 0; index < model.features.size(); index++) {
		correlations[index] = this->matern32(configuration, model.features[index], model.lengthScale);
	}
	double mean = std::inner_product(correlations.begin(), correlations.end(), model.alpha.begin(), 0.0);
	std::vector<double> solved = this->choleskySolve(model.cholesky, correlations);
	double explained = std::inner_product(correlations.begin(), correlations.end(), solved.begin(), 0.0);
	double variance = model.signalVariance * std::max(1e-12, 1.0 + 1e-6 - explained);
	return std::make_pair(mean, variance);
}

double InspireOptimizer::expectedImprovement(
	const GaussianProcessModel& model,
	const std::vector<double>& configuration) const {
	if (!model.valid || model.labels.empty()) return 1.0;
	std::pair<double, double> posterior = this->predict(model, configuration);
	double bestReward = *std::max_element(model.labels.begin(), model.labels.end());
	double standardDeviation = std::sqrt(std::max(0.0, posterior.second));
	if (standardDeviation <= 1e-12) return std::max(0.0, posterior.first - bestReward);
	double z = (posterior.first - bestReward) / standardDeviation;
	double cdf = 0.5 * std::erfc(-z / std::sqrt(2.0));
	double pdf = std::exp(-0.5 * z * z) / std::sqrt(2.0 * std::acos(-1.0));
	return (posterior.first - bestReward) * cdf + standardDeviation * pdf;
}

std::vector<double> InspireOptimizer::prescribe(unsigned int agent, const GaussianProcessModel& model) {
	std::vector<std::vector<double>> starts;
	if (!this->_current.empty()) starts.push_back(this->localConfiguration(agent, this->_current));
	if (agent < this->_localHistories.size() && !this->_localHistories[agent].empty()) {
		const LocalObservation* best = &this->_localHistories[agent][0];
		for (const LocalObservation& observation: this->_localHistories[agent]) {
			if (observation.reward > best->reward) best = &observation;
		}
		starts.push_back(best->configuration);
	}
	for (unsigned int restart = 0; restart < this->_restarts; restart++) {
		NetworkConfiguration randomConfiguration = this->randomLocalConfiguration(agent);
		std::vector<double> normalized;
		for (const std::tuple<double, double>& pair: randomConfiguration) {
			normalized.push_back(clampUnitOptimizer((std::get<0>(pair) + 82.0) / 20.0));
			normalized.push_back(clampUnitOptimizer((std::get<1>(pair) - 1.0) / 20.0));
		}
		starts.push_back(normalized);
	}
	if (starts.empty()) return {};

	std::vector<double> best = starts[0];
	double bestValue = this->expectedImprovement(model, best);
	for (std::vector<double> candidate: starts) {
		double candidateValue = this->expectedImprovement(model, candidate);
		for (unsigned int sweep = 0; sweep < this->_sweeps; sweep++) {
			bool improved = false;
			for (unsigned int dimension = 0; dimension < candidate.size(); dimension++) {
				for (int direction: {-1, 1}) {
					std::vector<double> neighbor = candidate;
					neighbor[dimension] = clampUnitOptimizer(neighbor[dimension] + direction / 20.0);
					unsigned int pair = dimension / 2;
					double sensitivity = -82.0 + 20.0 * neighbor[2 * pair];
					double power = 1.0 + 20.0 * neighbor[2 * pair + 1];
					if (!this->validPair(std::round(sensitivity), std::round(power))) continue;
					double value = this->expectedImprovement(model, neighbor);
					if (value > candidateValue + 1e-15) {
						candidate = neighbor;
						candidateValue = value;
						improved = true;
					}
				}
			}
			if (!improved) break;
		}
		if (candidateValue > bestValue) {
			best = candidate;
			bestValue = candidateValue;
		}
	}
	return best;
}

double InspireOptimizer::marginalMedian(std::vector<double> values) const {
	if (values.empty()) return 0.0;
	std::sort(values.begin(), values.end());
	// Select an actual prescription, not the average of the two central values.
	return values[(values.size() - 1) / 2];
}

void InspireOptimizer::traceDecision(
	const std::vector<NetworkConfiguration>& prescriptions,
	const NetworkConfiguration& consensus) {
	if (!this->_traceEnabled || !this->_traceFile.is_open()) return;
	for (unsigned int agent = 0; agent < prescriptions.size(); agent++) {
		this->_traceFile << this->_step << "\t" << agent << "\t";
		for (unsigned int index = 0; index < this->_neighborhoods[agent].size(); index++) {
			if (index > 0) this->_traceFile << ",";
			this->_traceFile << this->_neighborhoods[agent][index];
		}
		this->_traceFile << "\t";
		for (unsigned int index = 0; index < prescriptions[agent].size(); index++) {
			if (index > 0) this->_traceFile << ",";
			this->_traceFile << "(" << std::get<0>(prescriptions[agent][index]) << "," << std::get<1>(prescriptions[agent][index]) << ")";
		}
		this->_traceFile << "\t";
		for (unsigned int index = 0; index < consensus.size(); index++) {
			if (index > 0) this->_traceFile << ",";
			this->_traceFile << "(" << std::get<0>(consensus[index]) << "," << std::get<1>(consensus[index]) << ")";
		}
		this->_traceFile << "\n";
	}
	this->_traceFile.flush();
}

void InspireOptimizer::addToBase(
	NetworkConfiguration configuration,
	double reward,
	bool forward,
	std::vector<std::tuple<double, unsigned int>> individualRewards) {
	(void) forward;
	this->_current = configuration;
	this->_history.push_back(std::make_tuple(configuration, reward));
	if (individualRewards.size() < this->_neighborhoods.size()) {
		std::cerr << "ns3-inspire: expected " << this->_neighborhoods.size()
			<< " selfish rewards, received " << individualRewards.size() << std::endl;
		return;
	}
	for (unsigned int agent = 0; agent < this->_neighborhoods.size(); agent++) {
		double localReward = 0.0;
		for (unsigned int neighbor: this->_neighborhoods[agent]) {
			if (neighbor >= individualRewards.size() || this->_neighborhoods[neighbor].empty()) continue;
			localReward += std::get<0>(individualRewards[neighbor]) / this->_neighborhoods[neighbor].size();
		}
		this->_localHistories[agent].push_back({this->localConfiguration(agent, configuration), localReward});
	}
}

NetworkConfiguration InspireOptimizer::optimize() {
	if (this->_neighborhoods.empty()) return this->_current;
	std::uint64_t fitElapsedNs = 0;
	std::uint64_t inferenceElapsedNs = 0;
	std::vector<NetworkConfiguration> prescriptions(this->_neighborhoods.size());
	for (unsigned int agent = 0; agent < this->_neighborhoods.size(); agent++) {
		std::chrono::steady_clock::time_point fitStarted = std::chrono::steady_clock::now();
		GaussianProcessModel model = this->fitModel(agent);
		fitElapsedNs += static_cast<std::uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now() - fitStarted).count());
		std::chrono::steady_clock::time_point inferenceStarted = std::chrono::steady_clock::now();
		std::vector<double> normalized = this->prescribe(agent, model);
		for (unsigned int index = 0; index + 1 < normalized.size(); index += 2) {
			double sensitivity = std::round(-82.0 + 20.0 * normalized[index]);
			double power = std::round(1.0 + 20.0 * normalized[index + 1]);
			prescriptions[agent].push_back(std::make_tuple(sensitivity, power));
		}
		inferenceElapsedNs += static_cast<std::uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now() - inferenceStarted).count());
	}

	std::chrono::steady_clock::time_point consensusStarted = std::chrono::steady_clock::now();
	NetworkConfiguration consensus(this->_neighborhoods.size(), std::make_tuple(-82.0, 20.0));
	for (unsigned int target = 0; target < this->_neighborhoods.size(); target++) {
		std::vector<double> sensitivities, powers;
		for (unsigned int agent = 0; agent < this->_neighborhoods.size(); agent++) {
			std::vector<unsigned int>::const_iterator location = std::find(
				this->_neighborhoods[agent].begin(), this->_neighborhoods[agent].end(), target);
			if (location == this->_neighborhoods[agent].end()) continue;
			unsigned int index = location - this->_neighborhoods[agent].begin();
			if (index >= prescriptions[agent].size()) continue;
			sensitivities.push_back(std::get<0>(prescriptions[agent][index]));
			powers.push_back(std::get<1>(prescriptions[agent][index]));
		}
		double sensitivity = this->marginalMedian(sensitivities);
		double power = this->marginalMedian(powers);
		if (!this->validPair(sensitivity, power)) power = std::max(1.0, std::min(power, -62.0 - sensitivity));
		consensus[target] = std::make_tuple(sensitivity, power);
	}
	inferenceElapsedNs += static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now() - consensusStarted).count());
	RuntimeTiming::record("cpp", "training_fit", this->_step, 0, fitElapsedNs);
	RuntimeTiming::record("cpp", "inference_model", this->_step, 0, inferenceElapsedNs);
	this->_step++;
	this->traceDecision(prescriptions, consensus);
	this->_current = consensus;
	return consensus;
}

/**
 * Build a ThompsonNormalOptimizer
 * 
 * @param sampler Sampler* the sampler to draw new configurations from
 * @param add double the exploration parameter
 */
ThompsonNormalOptimizer::ThompsonNormalOptimizer(Sampler* sampler, double eps): Optimizer(sampler), _epsilon(eps) {  }

/**
 * Add an observation to the optimizer and its associated sampler.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated to the configuration 
 */
void ThompsonNormalOptimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	Optimizer::addToBase(configuration, reward, forward);

	// Search in attribute for a preexisting sample
	if (this->_normals.find(configuration) != this->_normals.end()) {
		NormalParameters& nps = this->_normals[configuration];
		double &mean = std::get<0>(nps),
					 &var = std::get<1>(nps);
		unsigned int &n = std::get<2>(nps);
		// Update the normal
		mean = (n * mean + reward) / (n + 1);
		var = 1.0 / (n + 1);
		n = n + 1;
 	} else {
		 // Create a new instance in normals
		this->_normals[configuration] = std::make_tuple(0, 1, 1);
	}
}

/**
 * Find the best configuration according to ThompsonNormal strategy
 * 
 * @return the best configuration according to ThompsonNormal strategy 
 */
NetworkConfiguration ThompsonNormalOptimizer::optimize() {
	// this->showDecisions();

	// Explore or exploit
	bool explore = this->_distribution(this->_generator) < this->_epsilon;
	if (explore) {
		// Request a new configuration to the sampler
		std::vector<NetworkConfiguration> forbidden;
		for (std::map<NetworkConfiguration, NormalParameters>::iterator it = this->_normals.begin(); it != this->_normals.end(); ++it)
			forbidden.push_back(it->first);
		NetworkConfiguration sampled = (*this->_sampler)(forbidden);
		if (std::get<0>(sampled[0]) != 0)
			return sampled;
	}

	// Sample mus for normal distributions
	std::vector<double> mus(this->_normals.size());
	int i = 0;
	for (std::map<NetworkConfiguration, NormalParameters>::iterator it = this->_normals.begin(); it != this->_normals.end(); ++it) {
		std::normal_distribution<> normal(std::get<0>(it->second), sqrt(std::get<1>(it->second)));
		mus[i] = normal(this->_generator);
		i++;
	}
	int maxElementIndex = std::max_element(mus.begin(), mus.end()) - mus.begin();
	std::map<NetworkConfiguration, NormalParameters>::iterator maxConf = this->_normals.begin();
	for (int i = 0; i < maxElementIndex; i++) maxConf++;

	return maxConf->first;
}
