#include "samplers.hh"
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include "unistd.h"

namespace {
	unsigned int samplerSeed() {
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

	bool paperHcmProfile() {
		const char* value = std::getenv("NSTEST_HCM_PROFILE");
		if (value == nullptr) return false;
		std::string profile(value);
		std::transform(profile.begin(), profile.end(), profile.begin(), [](unsigned char c) {
			return std::tolower(c);
		});
		return profile == "paper" || profile == "paper-batch" || profile == "paper_batch";
	}

	bool authorHcmProfile() {
		const char* value = std::getenv("NSTEST_HCM_PROFILE");
		if (value == nullptr) return false;
		std::string profile(value);
		std::transform(profile.begin(), profile.end(), profile.begin(), [](unsigned char c) {
			return std::tolower(c);
		});
		return profile == "author" || profile == "original" || profile == "upstream" ||
			profile == "author-restart" || profile == "author_restart" ||
			profile == "author-fixed" || profile == "author_fixed" ||
			profile == "author-fixed-restart" || profile == "author_fixed_restart";
	}

	bool fixedAuthorHcmProfile() {
		const char* value = std::getenv("NSTEST_HCM_PROFILE");
		if (value == nullptr) return false;
		std::string profile(value);
		std::transform(profile.begin(), profile.end(), profile.begin(), [](unsigned char c) {
			return std::tolower(c);
		});
		return profile == "author-fixed" || profile == "author_fixed" ||
			profile == "author-fixed-restart" || profile == "author_fixed_restart";
	}

	bool restartHcmProfile() {
		const char* value = std::getenv("NSTEST_HCM_PROFILE");
		if (value == nullptr) return false;
		std::string profile(value);
		std::transform(profile.begin(), profile.end(), profile.begin(), [](unsigned char c) {
			return std::tolower(c);
		});
		return profile == "author-restart" || profile == "author_restart" ||
			profile == "author-fixed-restart" || profile == "author_fixed_restart";
	}

	double hcmGlobalEpsilon() {
		const char* value = std::getenv("NSTEST_HCM_GLOBAL_EPS");
		if (value == nullptr || std::strlen(value) == 0) return 0.0;
		char* end = nullptr;
		double parsed = std::strtod(value, &end);
		if (end == value || !std::isfinite(parsed)) return 0.0;
		return std::max(0.0, std::min(1.0, parsed));
	}

	double hcmEnvDouble(const char* name, double fallback, double minimum, double maximum) {
		const char* value = std::getenv(name);
		if (value == nullptr || std::strlen(value) == 0) return fallback;
		char* end = nullptr;
		double parsed = std::strtod(value, &end);
		if (end == value || !std::isfinite(parsed)) return fallback;
		return std::max(minimum, std::min(maximum, parsed));
	}

	unsigned int hcmEnvUInt(const char* name, unsigned int fallback, unsigned int minimum, unsigned int maximum) {
		const char* value = std::getenv(name);
		if (value == nullptr || std::strlen(value) == 0) return fallback;
		char* end = nullptr;
		unsigned long parsed = std::strtoul(value, &end, 10);
		if (end == value) return fallback;
		return std::max(minimum, std::min(maximum, static_cast<unsigned int>(parsed)));
	}

	unsigned int primeForDimension(unsigned int dimension) {
		unsigned int found = 0;
		for (unsigned int candidate = 2; ; candidate++) {
			bool prime = true;
			for (unsigned int divisor = 2; divisor * divisor <= candidate; divisor++) {
				if (candidate % divisor == 0) {
					prime = false;
					break;
				}
			}
			if (prime && found++ == dimension) return candidate;
		}
	}

	double radicalInverse(unsigned int index, unsigned int base) {
		double value = 0.0;
		double factor = 1.0 / static_cast<double>(base);
		while (index > 0) {
			value += factor * static_cast<double>(index % base);
			index /= base;
			factor /= static_cast<double>(base);
		}
		return value;
	}
}

/**
 * Constraint on the parameters
 * 
 * @param txPower double the transmission power
 * @param obssPd double the sensibility
 * 
 * @return `true` if the constraint is satisfied, `false` otherwise
 */
bool constraint(double txPower, double obssPd) {
	return obssPd <= std::max(-82.0, std::min(-62.0, -82.0 + (20.0 - txPower)));
}

/**
 * Apply the parameter constraint on a whole network configuration
 * 
 * @param conf Container& containing the network configuration
 * 
 * @return `true` if the configuration respects the constraint, `false`
 * otherwise
 */
bool confConstraint(const NetworkConfiguration& conf) {
	for (std::tuple<double, double> t: conf) {
		if (!constraint(std::get<1>(t), std::get<0>(t))) {
			return false;
		}
	}

	return true;
}

/**
 * Initialize a sampler object
 * 
 * @param parameters Container the parameters, defining the space to sample on
 */
Sampler::Sampler(const std::vector<ConstrainedCouple>& parameters) : _parameters(parameters) {  }

/**
 * Virtual destructor for Sampler
 */
Sampler::~Sampler() {  }

/**
 * Build a random action from the specified parameters
 * 
 * @param forbidden Container the actions that are forbidden to be returned
 * 
 * @return a random action not in `forbidden` 
 */
NetworkConfiguration Sampler::randomAction(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) const {
	NetworkConfiguration conf;
	unsigned int i = 0;
	do {
		conf.clear();
		if (i > 10000) {
			return {std::tuple<double, double>(0, 0)};
		}

		for (ConstrainedCouple c: this->_parameters)
			conf.push_back(c.randomValue());
		i++;
	} while (std::find(forbidden.begin(), forbidden.end(), conf) != forbidden.end());

	return conf;
}

/**
 * Add a NetworkConfiguration and its associated reward to the base of the
 * agent. If the NetworkConfiguration is already present in base, the reward is
 * averaged according to an exponential average (alpha = 0.1).
 * 
 * @param features NetworkConfiguration the network configuration
 * @param reward double the associated reward
 */
void Sampler::addToBase(NetworkConfiguration features, double reward) {
	std::vector<NetworkConfiguration>::iterator it = std::find(this->_features.begin(), this->_features.end(), features);
	if (it != this->_features.end()) {
		int index = it - this->_features.begin();
		this->_rewards[index] = 0.5 * this->_rewards[index] + 0.5 * reward;
	} else {
		this->_features.push_back(features);
		this->_rewards.push_back(reward);
	}
}

std::vector<NetworkConfiguration> Sampler::randomActionsFromSample(unsigned int count, unsigned int maxAttempts, const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) {
	std::vector<NetworkConfiguration> candidates;
	std::vector<NetworkConfiguration> blocked = forbidden;
	unsigned int attempts = 0;
	while (candidates.size() < count && attempts < maxAttempts) {
		NetworkConfiguration sampled = this->randomActionFromSample(blocked);
		attempts++;
		if (sampled.empty() || !confConstraint(sampled)) continue;
		if (std::find(blocked.begin(), blocked.end(), sampled) != blocked.end()) continue;
		candidates.push_back(sampled);
		blocked.push_back(sampled);
	}

	return candidates;
}

void Sampler::selectedAction(const NetworkConfiguration& configuration) { }

bool Sampler::hasForcedAction() { return false; }

NetworkConfiguration Sampler::takeForcedAction(const std::vector<NetworkConfiguration>& forbidden) { return {}; }

void Sampler::notifyDecision(const NetworkConfiguration& configuration) { }

/**
 * Initialize a random sampler
 * 
 * @param parameters Container the parameters, defining the space to sample on
 */ 
RandomSampler::RandomSampler(const std::vector<ConstrainedCouple>& parameters) : Sampler(parameters), _generator(samplerSeed()) {  }

void RandomSampler::setSeed(unsigned int seed) {
	this->_generator.seed(seed);
}

/**
 * Initialize a uniform sampler
 * 
 * @param parameters Container the parameters, defining the space to sample on
 */
UniformSampler::UniformSampler(const std::vector<ConstrainedCouple>& parameters) : RandomSampler(parameters) {  }

/**
 * Return a random action based on collected rewards. Same as randomAction().
 * 
 * @param forbidden Container the network configuration we don't want.
 * 
 * @return a network configuration not in forbidden
 */
inline NetworkConfiguration UniformSampler::randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) { return this->randomAction(forbidden); }

/**
 * Operator () overloaded to be used
 * 
 * @param forbidden Container& containing the already explored configurations
 * 
 * @return a configuration not in `forbidden`
 */
inline NetworkConfiguration UniformSampler::operator()(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) { return this->randomAction(forbidden); }

/**
 * Build an HGM sampler.
 * 
 * @param parameters Container the parameters that define the sampling space
 * @param def NetworkConfiguration the default configuration of the network
 * @param nMax unsigned int the max number of gaussians in the mixture
 * @param eps double the exploration parameter
 * @param dist double the distance of the new target
 * @param nTests FunctionPointer a way to quantify how many tests must be done 
 */
HGMTSampler::HGMTSampler(const std::vector<ConstrainedCouple>& parameters, std::vector<NetworkConfiguration> bases, unsigned int nMax, double eps, double dist, unsigned int (*nTests)(std::vector<GaussianT>)) : RandomSampler(parameters), _nTests(nTests), _eps(eps), _dist(dist), _nMax(nMax), _testCounter(1) {
	for (NetworkConfiguration n: bases)
		this->addGaussian(n, 1.0, -1);
}

/**
 * Add a gaussian to the gaussian mixture. If the mixture is already full,
 * delete the oldest gaussian.
 * 
 * @param center NetworkConfiguration the center of the gaussian
 * @param dist double the distance to put the target at
 * @param reward double the reward max attained by the gaussian
 */ 
void HGMTSampler::addGaussian(const NetworkConfiguration& center, double dist, double reward) {
	// Turn the NetworkConfiguration into a list of normalized double
	std::vector<double> c(2 * center.size());
	unsigned int i = 0;
	for (std::tuple<double, double> t: center) {
		// Normalize the tuple
		std::tuple<double, double> normalized = this->_parameters[(int) (i / 2)].normalize(t);
		// Add it to the list
		c[i] = std::get<0>(normalized);
		c[i + 1] = std::get<1>(normalized);
		i += 2;
	}

	// Add the gaussian to the mixture
	if (this->_gaussians.size() == this->_nMax)
		this->_gaussians.erase(this->_gaussians.begin());

	this->_gaussians.push_back(std::make_tuple(c, dist / (21.0 * c.size()), reward));
}

/**
 * Add a NetworkConfiguration and its associated reward to the base of the
 * agent. If the NetworkConfiguration is already present in base, the reward is
 * averaged according to an exponential average (alpha = 0.1). Update also the
 * average reward of the gaussian used.
 * 
 * @param features NetworkConfiguration the network configuration
 * @param reward double the associated reward
 */
void HGMTSampler::addToBase(NetworkConfiguration features, double reward) {
	Sampler::addToBase(features, reward);

	// Update a center ?
	for (GaussianT gt: this->_gaussians) {
		NetworkConfiguration ct = this->rebuildConfiguration(std::get<0>(gt));
		if (features == ct) {
			double& maxRew = std::get<2>(gt);
			if (maxRew == -1) maxRew = reward;
			else maxRew = 0.9 * maxRew + 0.1 * reward;
		}
	}

	// Enough tests to change the mixture
	if (this->_testCounter >= this->_nTests(this->_gaussians)) {
		// Find the n greatest rewards
		std::vector<double> vec_to_sort = this->_rewards;
		std::sort(vec_to_sort.begin(), vec_to_sort.end(), std::greater<double>());
		int n = std::min(this->_nMax, (unsigned int) vec_to_sort.size());
		std::vector<double> dest(n);
		for (int i = 0; i < n; i++) {
			dest[i] = vec_to_sort[i];
		}

		// Add new gaussians best on the n best
		double target = *std::max_element(dest.begin(), dest.end()) + this->_dist * this->_eps;
		std::vector<GaussianT> gaussians_copy = this->_gaussians;
		this->_gaussians.clear();
		for (double d: dest) {
			if (d != -1) {
				NetworkConfiguration c = this->_features[std::find(this->_rewards.begin(), this->_rewards.end(), d) - this->_rewards.begin()];
				double gtarget = (target - d) / this->_eps;
				for (GaussianT g: gaussians_copy) {
					if (c == this->rebuildConfiguration(std::get<0>(g))) {
						gtarget = std::max(gtarget, 21.0 * std::get<1>(g) * 2.0 * c.size() + 1.0);
						break;
					}
				}
				this->addGaussian(c, gtarget, d);
			}
		}

		this->_testCounter = 0;
	}
}

/**
 * Sample the parameter space according to the gaussian mixture.
 * 
 * @param forbidden Container the NetworkConfiguration we don't want
 * 
 * @return a new NetworkConfiguration, not in forbidden
 */
NetworkConfiguration HGMTSampler::randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) {
	std::vector<double> probs;
	for (Gaussian g: this->_gaussians) {
		double weight = std::get<2>(g);
		if (weight < 0) weight = 0.5;

		probs.push_back(weight);
	}

	std::discrete_distribution<int> discreteDist(probs.begin(), probs.end());
	NetworkConfiguration selectedConf;
	unsigned int i = 0;
	do {
		i++;
		if (i > 10000) {
			return {std::tuple<double, double>(0, 0)};
		}

		if (i % 1000 == 0)
			this->increaseStds();
		selectedConf = this->rebuildConfiguration(this->normedSample(discreteDist));
	} while (std::find(forbidden.begin(), forbidden.end(), selectedConf) != forbidden.end() || !confConstraint(selectedConf));

	return selectedConf;
}

std::vector<NetworkConfiguration> HGMTSampler::randomActionsFromSample(unsigned int count, unsigned int maxAttempts, const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) {
	std::vector<NetworkConfiguration> candidates;
	std::vector<NetworkConfiguration> blocked = forbidden;
	if (count == 0 || this->_gaussians.empty()) return candidates;

	std::vector<GaussianT> gaussians = this->_gaussians;
	std::vector<double> probs;
	for (GaussianT g: gaussians) {
		double weight = std::get<2>(g);
		if (weight < 0) weight = 0.5;
		probs.push_back(weight);
	}
	std::discrete_distribution<int> discreteDist(probs.begin(), probs.end());

	unsigned int attempts = 0;
	unsigned int rejected = 0;
	while (candidates.size() < count && attempts < maxAttempts) {
		if (rejected > 0 && rejected % 1000 == 0) {
			for (unsigned int i = 0; i < gaussians.size(); i++) {
				double& std = std::get<1>(gaussians[i]);
				std += 1.0 / std::get<0>(gaussians[i]).size();
			}
		}

		int toUse = discreteDist(this->_generator);
		std::vector<double> normed(2 * this->_parameters.size());
		unsigned int i = 0;
		for (double ci: std::get<0>(gaussians[toUse])) {
			std::normal_distribution<double> normalDist(ci, std::get<1>(gaussians[toUse]));
			do {
				normed[i] = normalDist(this->_generator);
			} while (normed[i] < 0 || normed[i] > 1);
			i++;
		}

		NetworkConfiguration selectedConf = this->rebuildConfiguration(normed);
		attempts++;
		if (std::find(blocked.begin(), blocked.end(), selectedConf) != blocked.end() || !confConstraint(selectedConf)) {
			rejected++;
			continue;
		}

		candidates.push_back(selectedConf);
		blocked.push_back(selectedConf);
		rejected = 0;
	}

	return candidates;
}

/**
 * Increase the standard deviations of all gaussians, when possible. 
 */
void HGMTSampler::increaseStds() {
	for (unsigned int i = 0; i < this->_gaussians.size(); i++) {
		double& std = std::get<1>(this->_gaussians[i]);
		std += 1.0 / std::get<0>(this->_gaussians[i]).size();
	}
}

/**
 * Rebuild configuration given a sample coming from a multidimensional normal
 * distribution.
 * 
 * @param sampled Container the sample from the multidim normal distribution
 * 
 * @return the network configuration
 */
NetworkConfiguration HGMTSampler::rebuildConfiguration(const std::vector<double>& sampled) const {
	unsigned int i = 0;
	NetworkConfiguration conf(sampled.size() / 2);
	for (const ConstrainedCouple& c: this->_parameters) {
		conf[(int) (i / 2)] = c.fromNormalized(std::make_tuple(sampled[i], sampled[i + 1]));
		i += 2;
	}

	return conf;
} 

/**
 * Sample from one of the gaussians, chosen with discreteDist
 * 
 * @param discreteDist discrete_distribution a discrete distribution to choose
 * one of the gaussians
 * 
 * @return the vector issued from the normal distribution sampling 
 */ 
std::vector<double> HGMTSampler::normedSample(std::discrete_distribution<int>& discreteDist) {
	int toUse = discreteDist(this->_generator);
	std::vector<double> normed(2 * this->_parameters.size());
	unsigned int i = 0;
	for (double ci: std::get<0>(this->_gaussians[toUse])) {
		std::normal_distribution<double> normalDist(ci, std::get<1>(this->_gaussians[toUse]));
		do {
			normed[i] = normalDist(this->_generator);
		} while (normed[i] < 0 || normed[i] > 1);
		i++;
	}

	return normed;
}

/**
 * Overloading of () operator, use the gaussian mixture to get a new element
 * 
 * @param forbidden Container the configurations we don't want
 * 
 * @return a network configuration not in forbidden
 */
NetworkConfiguration HGMTSampler::operator()(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) {
	this->_testCounter++;
	return this->randomActionFromSample(forbidden);
}

/**
 * Build an HGM sampler.
 * 
 * @param parameters Container the parameters that define the sampling space
 * @param def NetworkConfiguration the default configuration of the network
 * @param nMax unsigned int the max number of gaussians in the mixture
 * @param eps double the exploration parameter
 * @param dist double the distance of the new target
 * @param nTests FunctionPointer a way to quantify how many tests must be done 
 */
HCMSampler::HCMSampler(const std::vector<ConstrainedCouple>& parameters, std::vector<NetworkConfiguration> bases, unsigned int nMax, double eps, unsigned int (*nTests)(std::vector<Ring>), DistanceMode dmode) : RandomSampler(parameters), _nTests(nTests), _eps(eps), _dist(paperHcmProfile() ? 1.0 : 5.0), _nMax(nMax), _testCounter(authorHcmProfile() ? 1 : 0), _dmode(dmode), _paperProfile(paperHcmProfile()), _authorProfile(authorHcmProfile()), _fixedAuthorProfile(fixedAuthorHcmProfile()), _restartProfile(restartHcmProfile()), _globalEps(hcmGlobalEpsilon()), _batchExploreFraction(hcmEnvDouble("NSTEST_HCM_BATCH_EXPLORE_FRACTION", 0.0, 0.0, 1.0)), _batchMinRadius(hcmEnvDouble("NSTEST_HCM_BATCH_MIN_RADIUS", 5.0, 1.0, 20.0)), _batchMaxRadius(hcmEnvDouble("NSTEST_HCM_BATCH_MAX_RADIUS", 5.0, 1.0, 20.0)), _restartPatience(hcmEnvUInt("NSTEST_HCM_RESTART_PATIENCE", 48, 1, 100000)), _restartBurst(hcmEnvUInt("NSTEST_HCM_RESTART_BURST", 16, 1, 100000)), _restartMaxBursts(hcmEnvUInt("NSTEST_HCM_RESTART_MAX_BURSTS", 4, 1, 100000)), _restartBurstsStarted(0), _restartMinDelta(hcmEnvDouble("NSTEST_HCM_RESTART_MIN_DELTA", 0.01, 0.0, 1.0)), _restartTargetReward(hcmEnvDouble("NSTEST_HCM_RESTART_TARGET", 0.89, 0.0, 1.0)), _restartStagnation(0), _restartRemaining(0), _haltonIndex(1 + samplerSeed() % 4093), _restartBestReward(-std::numeric_limits<double>::infinity()), _restartDecisionPending(false), _restartLastWasForced(false) {
	if (this->_batchMaxRadius < this->_batchMinRadius) this->_batchMaxRadius = this->_batchMinRadius;
	for (NetworkConfiguration n: bases)
		this->addRing(n, this->_dist, 0.02, -1);
	if (std::getenv("NSTEST_HCM_TRACE") != nullptr) {
		std::cout << "ns3-hcm: profile=" << (this->_fixedAuthorProfile ? (this->_restartProfile ? "author-fixed-restart" : "author-fixed") : (this->_restartProfile ? "author-restart" : (this->_authorProfile ? "author" : (this->_paperProfile ? "paper" : "batch"))))
						<< " bases=" << bases.size() << " initial_radius=" << this->_dist
						<< " eps=" << this->_eps << " global_eps=" << this->_globalEps
						<< " batch_explore_fraction=" << this->_batchExploreFraction
						<< " batch_min_radius=" << this->_batchMinRadius
						<< " batch_max_radius=" << this->_batchMaxRadius
						<< " restart_patience=" << this->_restartPatience
						<< " restart_burst=" << this->_restartBurst
						<< " restart_max_bursts=" << this->_restartMaxBursts
						<< " restart_min_delta=" << this->_restartMinDelta
						<< " restart_target=" << this->_restartTargetReward << std::endl;
	}
}

/**
 * Add a gaussian to the gaussian mixture. If the mixture is already full,
 * delete the oldest gaussian.
 * 
 * @param center NetworkConfiguration the center of the gaussian
 * @param dist double the distance to put the target at
 * @param reward double the reward max attained by the gaussian
 */ 
void HCMSampler::addRing(const NetworkConfiguration& center, double dist, double std, double reward) {
	// Turn the NetworkConfiguration into a list of normalized double
	unsigned int n = 2 * center.size();
	std::vector<double> c(n);
	unsigned int i = 0;
	for (std::tuple<double, double> t: center) {
		// Normalize the tuple
		std::tuple<double, double> normalized = this->_parameters[(int) (i / 2)].normalize(t);
		// Add it to the list
		c[i] = std::get<0>(normalized);
		c[i + 1] = std::get<1>(normalized);
		i += 2;
	}

	// Add the circular to the mixture
	if (this->_circulars.size() == this->_nMax)
		this->_circulars.erase(this->_circulars.begin());

	this->_circulars.push_back(std::make_tuple(c, dist / 21.0, std / (21.0 * n), reward));
}

/**
 * Add a NetworkConfiguration and its associated reward to the base of the
 * agent. If the NetworkConfiguration is already present in base, the reward is
 * averaged according to an exponential average (alpha = 0.1). Update also the
 * average reward of the gaussian used.
 * 
 * @param features NetworkConfiguration the network configuration
 * @param reward double the associated reward
 */
void HCMSampler::addToBase(NetworkConfiguration features, double reward) {
	if (this->_authorProfile) {
		// Literal HCM behavior of upstream commit 25591d5.  In particular, the
		// value-copy center update and first-match handling of tied rewards are
		// preserved because changing either measurably changes the paper baseline.
		Sampler::addToBase(features, reward);
		if (this->_fixedAuthorProfile) {
			// The upstream range-for copied each Ring, so its EMA update was lost.
			for (Ring& c: this->_circulars) {
				NetworkConfiguration ct = this->rebuildConfiguration(std::get<0>(c));
				if (features == ct) {
					double& maxRew = std::get<3>(c);
					if (maxRew == -1) maxRew = reward;
					else maxRew = 0.9 * maxRew + 0.1 * reward;
				}
			}
		} else {
			// Preserve the literal upstream value-copy behavior for reproducibility.
			for (Ring c: this->_circulars) {
				NetworkConfiguration ct = this->rebuildConfiguration(std::get<0>(c));
				if (features == ct) {
					double& maxRew = std::get<3>(c);
					if (maxRew == -1) maxRew = reward;
					else maxRew = 0.9 * maxRew + 0.1 * reward;
				}
			}
		}

		unsigned int threshold = this->_nTests(this->_circulars);
		if (this->_dmode == CYCLE || this->_testCounter >= threshold) {
			std::vector<unsigned int> ranked(this->_rewards.size());
			std::iota(ranked.begin(), ranked.end(), 0);
			std::stable_sort(ranked.begin(), ranked.end(), [this](unsigned int lhs, unsigned int rhs) {
				if (this->_rewards[lhs] != this->_rewards[rhs])
					return this->_rewards[lhs] > this->_rewards[rhs];
				// ADHOC is quantized, so exact ties become common in long runs.
				// The fixed profile must not keep selecting the oldest stale winner.
				if (this->_fixedAuthorProfile) return lhs > rhs;
				return lhs < rhs;
			});
			unsigned int n = std::min(this->_nMax, static_cast<unsigned int>(ranked.size()));
			double max_elem = this->_rewards[ranked[0]];
			double target = std::min(this->_dist * this->_eps + max_elem, 1.0);
			if (this->_dmode == CYCLE) {
				double min_v = 1.0;
				double max_v = 10.0;
				unsigned int cycle_size = 36;
				double t_pos = 2.0 * static_cast<double>(std::min(this->_testCounter % cycle_size, cycle_size - (this->_testCounter % cycle_size))) / static_cast<double>(cycle_size);
				double dist = t_pos * (max_v - min_v) + min_v;
				target = std::min(1.0, max_elem + dist * this->_eps);
			}

			std::vector<Ring> circulars_copy = this->_circulars;
			this->_circulars.clear();
			for (unsigned int i = 0; i < n; i++) {
				double d = this->_rewards[ranked[i]];
				if (d != -1) {
					// The fixed profile keeps tied reward/configuration pairs distinct;
					// the literal profiles retain upstream's first-match collapse.
					unsigned int index = this->_fixedAuthorProfile
						? ranked[i]
						: static_cast<unsigned int>(std::find(this->_rewards.begin(), this->_rewards.end(), d) - this->_rewards.begin());
					NetworkConfiguration c = this->_features[index];
					double ctarget = std::max((target - d) / this->_eps, 1.0);
					this->addRing(c, ctarget, 0.02, d);
				}
			}
			if (std::getenv("NSTEST_HCM_TRACE") != nullptr) {
				std::cout << "ns3-hcm: rebuild profile=author observed=" << this->_features.size()
								<< " threshold=" << threshold << " max_reward=" << max_elem
								<< " rings=" << this->_circulars.size() << std::endl;
			}
			if (this->_dmode != CYCLE) this->_testCounter = 0;
		}
		return;
	}

	bool isNewConfiguration = std::find(this->_features.begin(), this->_features.end(), features) == this->_features.end();
	Sampler::addToBase(features, reward);
	if (this->_paperProfile || isNewConfiguration) this->_testCounter++;

	// Update a center ?
	for (Ring& c: this->_circulars) {
		NetworkConfiguration ct = this->rebuildConfiguration(std::get<0>(c));
		if (features == ct) {
			double& maxRew = std::get<3>(c);
			if (maxRew == -1) maxRew = reward;
			else maxRew = 0.9 * maxRew + 0.1 * reward;
		}
	}

	// std::cout << this->_testCounter << " vs. " << this->_nTests(this->_circulars) << std::endl;

	// Enough tests to change the mixture
	unsigned int threshold = this->updateThreshold();
	if (this->_dmode == CYCLE || this->_testCounter >= threshold) {
		// Rank reward/configuration pairs together. Looking a tied reward up with
		// std::find() duplicated the first matching configuration and collapsed
		// several rings onto the same center for quantized ADHOC rewards.
		std::vector<unsigned int> ranked(this->_rewards.size());
		std::iota(ranked.begin(), ranked.end(), 0);
		std::stable_sort(ranked.begin(), ranked.end(), [this](unsigned int lhs, unsigned int rhs) {
			return this->_rewards[lhs] > this->_rewards[rhs];
		});
		unsigned int n = std::min(this->_nMax, static_cast<unsigned int>(ranked.size()));

		// Add new circulars best on the n best
		double max_elem = this->_rewards[ranked[0]];
		double target = std::min(this->_dist * this->_eps + max_elem, 1.0);

		if (this->_dmode == CYCLE) {
			double min_v = 1.0;
			double max_v = 10.0;
			unsigned int cycle_size = 36;
			double t_pos = 2.0 * (double) std::min(this->_testCounter % cycle_size, cycle_size - (this->_testCounter % cycle_size)) / (double) cycle_size;
			double dist = t_pos * (max_v - min_v) + min_v;
			target = std::min(1.0, max_elem + dist * this->_eps);
			// std::cout << this->_testCounter << " and " << t_pos << " and " << dist << " and " << target << std::endl;
		}

		this->_circulars.clear();
		for (unsigned int i = 0; i < n; i++) {
			unsigned int rewardIndex = ranked[i];
			double d = this->_rewards[rewardIndex];
			if (d != -1) {
				NetworkConfiguration c = this->_features[rewardIndex];
				double ctarget = std::max((target - d) / this->_eps, 1.0);
				// std::cout << ctarget << std::endl;
				this->addRing(c, ctarget, 0.02, d);
			}
		}
		if (std::getenv("NSTEST_HCM_TRACE") != nullptr) {
			std::cout << "ns3-hcm: rebuild observed=" << this->_features.size()
							<< " threshold=" << threshold << " max_reward=" << max_elem
							<< " rings=" << this->_circulars.size() << std::endl;
		}
		// std::cout << std::endl;

		if (this->_dmode != CYCLE)
			this->_testCounter = 0;
	}
}

unsigned int HCMSampler::updateThreshold() const {
	if (!this->_paperProfile) return this->_nTests(this->_circulars);

	// Algorithm 2 updates after sum_i d_i * dim(c_i) configuration proposals.
	// The optimizer evaluates every chosen configuration three times (Table 3),
	// so this observation-side counter uses the equivalent 3x threshold. This
	// also guarantees progress when an optimizer temporarily repeats its best
	// configuration instead of selecting a new proposal.
	double required = 0.0;
	for (const Ring& ring: this->_circulars) {
		double latticeRadius = std::max(1.0, 21.0 * std::get<1>(ring));
		required += latticeRadius * std::get<0>(ring).size();
	}
	return std::max(1u, 3u * static_cast<unsigned int>(std::ceil(required)));
}

/**
 * Sample the parameter space according to the gaussian mixture.
 * 
 * @param forbidden Container the NetworkConfiguration we don't want
 * 
 * @return a new NetworkConfiguration, not in forbidden
 */
NetworkConfiguration HCMSampler::randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) {
	// The author HCM remains the primary proposal distribution.  An optional,
	// explicit global proposal floor prevents all ring-based optimizers from
	// becoming permanently trapped after their local centers have converged.
	if (this->_authorProfile && this->_globalEps > 0.0 &&
			std::uniform_real_distribution<double>(0.0, 1.0)(this->_generator) < this->_globalEps) {
		return this->randomAction(forbidden);
	}
	std::vector<double> probs;
	for (Ring c: this->_circulars) {
		double weight = std::get<3>(c);
		if (weight < 0) weight = 0.5;

		probs.push_back(weight);
	}

	// this->printRings();

	std::discrete_distribution<int> discreteDist(probs.begin(), probs.end());
	NetworkConfiguration selectedConf;
	unsigned int i = 0;
	unsigned int maxTries = this->_paperProfile ? 512 : 10000;
	do {
		i++;
		if (i > maxTries) {
			if (this->_authorProfile) return {std::tuple<double, double>(0, 0)};
			return {};
		}

		std::vector<double> sampled = this->normedSample(discreteDist);
		if (sampled.empty()) {
			// In high dimensions an exact-radius sphere can have no practical
			// intersection with the constrained box.  Do not spin forever: keep
			// exploration alive with the sampler's bounded uniform fallback.
			NetworkConfiguration fallback = this->randomAction(forbidden);
			if (fallback.size() == this->_parameters.size() && confConstraint(fallback))
				return fallback;
			return {};
		}
		selectedConf = this->rebuildConfiguration(sampled);
		// for (std::tuple<double, double> t: selectedConf) { std::cout << "(" << std::get<0>(t) << "," << std::get<1>(t) << "),"; }
		// std::cout << confConstraint(selectedConf) << std::endl;
	} while (std::find(forbidden.begin(), forbidden.end(), selectedConf) != forbidden.end() || !confConstraint(selectedConf));

	return selectedConf;
}

std::vector<NetworkConfiguration> HCMSampler::randomActionsFromSample(unsigned int count, unsigned int maxAttempts, const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) {
	if (this->_batchExploreFraction <= 0.0 && !this->_paperProfile)
		return Sampler::randomActionsFromSample(count, maxAttempts, forbidden);

	std::vector<NetworkConfiguration> candidates;
	std::vector<NetworkConfiguration> blocked = forbidden;
	if (count == 0 || this->_circulars.empty()) return candidates;

	std::vector<double> probs;
	for (const Ring& ring: this->_circulars) {
		double weight = std::get<3>(ring);
		probs.push_back(weight < 0 ? 0.5 : weight);
	}
	std::discrete_distribution<int> discreteDist(probs.begin(), probs.end());
	unsigned int expandedTarget = static_cast<unsigned int>(std::round(count * this->_batchExploreFraction));
	unsigned int localTarget = count - expandedTarget;
	unsigned int localAccepted = 0;
	unsigned int expandedAccepted = 0;
	unsigned int attempts = 0;
	unsigned int boundedAttempts = std::min(maxAttempts, std::max(4096u, count * 64u));
	while (candidates.size() < count && attempts < boundedAttempts) {
		attempts++;
		bool expanded = localAccepted >= localTarget && expandedAccepted < expandedTarget;
		double expandedRadius = this->_batchMinRadius;
		if (expanded && this->_batchMaxRadius > this->_batchMinRadius) {
			expandedRadius = std::uniform_real_distribution<double>(
				this->_batchMinRadius,
				this->_batchMaxRadius
			)(this->_generator);
		}
		std::vector<double> sampled = this->normedSample(
			discreteDist,
			expanded ? expandedRadius : 0.0
		);
		if (sampled.empty()) continue;
		NetworkConfiguration configuration = this->rebuildConfiguration(sampled);
		if (!confConstraint(configuration)) continue;
		if (std::find(blocked.begin(), blocked.end(), configuration) != blocked.end()) continue;
		candidates.push_back(configuration);
		blocked.push_back(configuration);
		if (expanded) expandedAccepted++;
		else localAccepted++;
	}
	return candidates;
}

/**
 * Increase the standard deviations of all gaussians, when possible. 
 */
void HCMSampler::shiftCenter() {
	for (unsigned int i = 0; i < this->_circulars.size(); i++) {
		bool tx = std::uniform_int_distribution<unsigned int>(0, 99)(this->_generator) >= 25;
		std::vector<double>& center = std::get<0>(this->_circulars[i]);
		unsigned int j = std::uniform_int_distribution<unsigned int>(0, center.size() / 2 - 1)(this->_generator);
		unsigned int idx = 2 * j + tx;
		double delta = ((center[idx] < 0.25) - (0.25 < center[idx])) / 21.0;
		center[idx] += delta;

		if (!confConstraint(rebuildConfiguration({center[2 * j + 1], center[2 * j]}))) {
			center[idx] -= delta;
			i--;
		}
	}
}

void HCMSampler::printRings() const {
	for (Ring c: this->_circulars) {
		std::cout << "Ring: ";
		for (std::tuple<double, double> t: rebuildConfiguration(std::get<0>(c))) { std::cout << "(" << std::get<0>(t) << "," << std::get<1>(t) << "),"; }
		std::cout << " " << std::get<1>(c) << std::endl;
	}
}

/**
 * Rebuild configuration given a sample coming from a multidimensional normal
 * distribution.
 * 
 * @param sampled Container the sample from the multidim normal distribution
 * 
 * @return the network configuration
 */
NetworkConfiguration HCMSampler::rebuildConfiguration(const std::vector<double>& sampled, std::vector<unsigned int> location) const {
	unsigned int i = 0;
	NetworkConfiguration conf(sampled.size() / 2);
	if (location.size() == 0) {
		for (unsigned int l = 0; l < sampled.size() / 2; l++)
			location.push_back(l);
	}

	for (unsigned int l: location) {
		conf[(int) (i / 2)] = this->_parameters[l].fromNormalized(std::make_tuple(sampled[i], sampled[i + 1]));
		i += 2;
	}
	return conf;
} 

/**
 * Sample from one of the gaussians, chosen with discreteDist
 * 
 * @param discreteDist discrete_distribution a discrete distribution to choose
 * one of the gaussians
 * 
 * @return the vector issued from the normal distribution sampling 
 */ 
std::vector<double> HCMSampler::normedSample(std::discrete_distribution<int>& discreteDist, double minimumLatticeRadius) {
	int toUse = discreteDist(this->_generator);
	unsigned int n = 2 * this->_parameters.size();
	std::vector<double>& center = std::get<0>(this->_circulars[toUse]);
	if (this->_paperProfile) {
		// Equation 6 is expressed in discrete L1 configuration distance.  Draw
		// directly from that lattice shell so radius one produces an actual
		// one-parameter neighbor instead of repeatedly rounding back to center.
		const double latticeStep = 1.0 / 20.0;
		unsigned int steps = std::max(1u, static_cast<unsigned int>(std::round(std::max(minimumLatticeRadius, 21.0 * std::get<1>(this->_circulars[toUse])))));
		for (unsigned int attempt = 0; attempt < 256; attempt++) {
			std::vector<int> deltas(n, 0);
			bool possible = true;
			for (unsigned int step = 0; step < steps; step++) {
				std::vector<std::tuple<unsigned int, int>> moves;
				for (unsigned int coordinate = 0; coordinate < n; coordinate++) {
					double value = center[coordinate] + deltas[coordinate] * latticeStep;
					if (deltas[coordinate] >= 0 && value + latticeStep <= 1.0 + 1e-12)
						moves.push_back(std::make_tuple(coordinate, 1));
					if (deltas[coordinate] <= 0 && value - latticeStep >= -1e-12)
						moves.push_back(std::make_tuple(coordinate, -1));
				}
				if (moves.empty()) {
					possible = false;
					break;
				}
				const auto& move = moves[std::uniform_int_distribution<unsigned int>(0, moves.size() - 1)(this->_generator)];
				deltas[std::get<0>(move)] += std::get<1>(move);
			}
			if (!possible) continue;

			std::vector<double> sampled(center);
			for (unsigned int coordinate = 0; coordinate < n; coordinate++)
				sampled[coordinate] += deltas[coordinate] * latticeStep;
			if (confConstraint(this->rebuildConfiguration(sampled))) return sampled;
		}
		return {};
	}
	if (this->_authorProfile) {
		unsigned int i = 1;
		std::vector<double> hypersphere_sample(n);
		bool sample_not_ok = true;
		do {
			if (i % 500000 == 0) this->shiftCenter();
			std::normal_distribution<double> normalDist(0, 1);
			for (unsigned int j = 0; j < n; j++) hypersphere_sample[j] = normalDist(this->_generator);
			double norm = 0;
			for (double d: hypersphere_sample) norm += d * d;
			norm = sqrt(norm);
			for (unsigned int j = 0; j < n; j++) hypersphere_sample[j] /= norm;

			sample_not_ok = false;
			double dist = std::max(std::get<1>(this->_circulars[toUse]), minimumLatticeRadius / 21.0);
			for (unsigned int j = 0; j < n; j++) {
				hypersphere_sample[j] = hypersphere_sample[j] * dist + center[j];
				sample_not_ok = hypersphere_sample[j] < 0 || hypersphere_sample[j] > 1;
				if (j % 2 == 1) {
					std::vector<double> couple({hypersphere_sample[j-1], hypersphere_sample[j]});
					if (!confConstraint(rebuildConfiguration(couple))) sample_not_ok = true;
				}
				if (sample_not_ok) break;
			}
			i++;
		} while (sample_not_ok);
		return hypersphere_sample;
	}
	unsigned int i = 0;
	const unsigned int maxAttempts = 4096;
	std::vector<double> normed(n);
	bool sample_not_ok = true;
	std::vector<double> hypersphere_sample(n);

	do {
		std::normal_distribution<double> normalDist(0, 1);
		// Sample a direction in space
		for (unsigned int j = 0; j < n; j++) {
			hypersphere_sample[j] = normalDist(this->_generator);
		}
		
		// Normalize the vector
		double norm = 0;
		for (double d: hypersphere_sample) norm += d * d;
		norm = sqrt(norm);
		for (unsigned int j = 0; j < n; j++) hypersphere_sample[j] /= norm;

		sample_not_ok = false;
		// normalDist = std::normal_distribution<double>(std::get<1>(this->_circulars[toUse]), std::get<2>(this->_circulars[toUse]));
		double dist = std::max(std::get<1>(this->_circulars[toUse]), minimumLatticeRadius / 21.0); // normalDist(this->_generator);
		for (unsigned int j = 0; j < n; j++) {
			hypersphere_sample[j] = hypersphere_sample[j] * dist + center[j];
			sample_not_ok = hypersphere_sample[j] < 0 || hypersphere_sample[j] > 1;
			if (j % 2 == 1) {
				std::vector<double> couple({hypersphere_sample[j-1], hypersphere_sample[j]});
				if (!confConstraint(rebuildConfiguration(couple))) {
					sample_not_ok = true;
				}
			}
			if (sample_not_ok)
					break;
		}
		i++;
	} while (sample_not_ok && i < maxAttempts);

	if (sample_not_ok) return {};

	return hypersphere_sample;
}

/**
 * Overloading of () operator, use the gaussian mixture to get a new element
 * 
 * @param forbidden Container the configurations we don't want
 * 
 * @return a network configuration not in forbidden
 */
NetworkConfiguration HCMSampler::operator()(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) {
	if (this->_authorProfile) this->_testCounter++;
	return this->randomActionFromSample(forbidden);
}

bool HCMSampler::hasForcedAction() {
	if (!this->_restartProfile) return false;

	// This hook runs only between complete optimizer chains.  The reward stored
	// by the sampler therefore contains all three ADHOC observations for the
	// preceding decision; forced probes never interrupt a measurement chain.
	if (this->_restartDecisionPending) {
		auto found = std::find(this->_features.begin(), this->_features.end(), this->_restartLastConfiguration);
		if (found != this->_features.end()) {
			unsigned int index = found - this->_features.begin();
			double reward = this->_rewards[index];
			bool materialImprovement = !std::isfinite(this->_restartBestReward) ||
				reward > this->_restartBestReward + this->_restartMinDelta;
			if (materialImprovement) {
				this->_restartBestReward = reward;
				this->_restartBestConfiguration = this->_restartLastConfiguration;
				this->_restartStagnation = 0;
				if (this->_restartLastWasForced) {
					this->_restartRemaining = 0;
					this->_restartBurstsStarted = this->_restartMaxBursts;
				}
			} else {
				this->_restartBestReward = std::max(this->_restartBestReward, reward);
				// A probe burst is evidence collection, not another failed local
				// decision.  Count only ordinary HCM/optimizer decisions toward the
				// next restart so bursts cannot trigger back-to-back forever.
				if (!this->_restartLastWasForced) this->_restartStagnation++;
			}
		}
		this->_restartDecisionPending = false;
	}
	if (this->_restartBestReward >= this->_restartTargetReward) {
		this->_restartRemaining = 0;
		this->_restartBurstsStarted = this->_restartMaxBursts;
		return false;
	}

	if (this->_restartRemaining == 0 &&
			this->_restartBurstsStarted < this->_restartMaxBursts &&
			this->_restartStagnation >= this->_restartPatience) {
		this->_restartRemaining = this->_restartBurst;
		this->_restartBurstsStarted++;
		this->_restartStagnation = 0;
		if (std::getenv("NSTEST_HCM_TRACE") != nullptr) {
			std::cout << "ns3-hcm: restart-trigger best_reward=" << this->_restartBestReward
						<< " burst=" << this->_restartBurst
						<< " burst_number=" << this->_restartBurstsStarted
						<< " halton_index=" << this->_haltonIndex << std::endl;
		}
	}
	return this->_restartRemaining > 0;
}

NetworkConfiguration HCMSampler::takeForcedAction(const std::vector<NetworkConfiguration>& forbidden) {
	if (!this->hasForcedAction()) return {};
	for (unsigned int attempt = 0; attempt < 10000; attempt++) {
		unsigned int sequenceIndex = this->_haltonIndex++;
		NetworkConfiguration configuration;
		for (unsigned int parameter = 0; parameter < this->_parameters.size(); parameter++) {
			double coordinate = radicalInverse(sequenceIndex, primeForDimension(parameter));
			// Match ConstrainedCouple::randomValue(): select uniformly from all
			// 210 integer lattice pairs satisfying sensitivity + power <= -62.
			// A single low-discrepancy coordinate per AP also keeps the Halton
			// search dimension equal to the number of APs instead of twice that.
			unsigned int pairIndex = std::min(209u, static_cast<unsigned int>(std::floor(210.0 * coordinate)));
			unsigned int sensitivityIndex = 0;
			while (pairIndex >= 20u - sensitivityIndex) {
				pairIndex -= 20u - sensitivityIndex;
				sensitivityIndex++;
			}
			unsigned int powerIndex = pairIndex;
			configuration.push_back(this->_parameters[parameter].fromNormalized(
				std::make_tuple(sensitivityIndex / 20.0, powerIndex / 20.0)
			));
		}
		if (!confConstraint(configuration)) continue;
		if (std::find(this->_features.begin(), this->_features.end(), configuration) != this->_features.end()) continue;
		if (std::find(forbidden.begin(), forbidden.end(), configuration) != forbidden.end()) continue;

		this->_restartRemaining--;
		this->_restartLastConfiguration = configuration;
		this->_restartDecisionPending = true;
		this->_restartLastWasForced = true;
		this->_testCounter++;
		if (std::getenv("NSTEST_HCM_TRACE") != nullptr) {
			std::cout << "ns3-hcm: restart-probe sequence_index=" << sequenceIndex
						<< " remaining=" << this->_restartRemaining << std::endl;
		}
		return configuration;
	}

	this->_restartRemaining = 0;
	return {};
}

void HCMSampler::notifyDecision(const NetworkConfiguration& configuration) {
	if (!this->_restartProfile) return;
	this->_restartLastConfiguration = configuration;
	this->_restartDecisionPending = true;
	this->_restartLastWasForced = false;
}

void HCMSampler::selectedAction(const NetworkConfiguration& configuration) {
	if (!this->_authorProfile) return;
	// The upstream API generated one candidate per sampler call.  ASK_CHOICE
	// generates a virtual batch, so count only the novel candidate that is
	// actually selected; this is the sequential upstream semantics for NB/QNN.
	if (std::find(this->_features.begin(), this->_features.end(), configuration) == this->_features.end()) {
		this->_testCounter++;
	}
}
