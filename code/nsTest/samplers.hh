#ifndef __SAMPLERS_HH
#define __SAMPLERS_HH

#include <vector>
#include <tuple>
#include <algorithm>
#include <random>
#include <chrono>

#include "parameters.hh"

using NetworkConfiguration = std::vector<std::tuple<double, double>>;

bool confConstraint(const NetworkConfiguration& conf);

class Sampler {
	public:
		Sampler(const std::vector<ConstrainedCouple>& parameters);
		virtual ~Sampler();
		virtual NetworkConfiguration randomAction(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>()) const;
		virtual void addToBase(NetworkConfiguration features, double reward);
		virtual NetworkConfiguration randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>()) = 0;
		virtual std::vector<NetworkConfiguration> randomActionsFromSample(unsigned int count, unsigned int maxAttempts, const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
		virtual bool hasForcedAction();
		virtual NetworkConfiguration takeForcedAction(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
		virtual void notifyDecision(const NetworkConfiguration& configuration);
		virtual void selectedAction(const NetworkConfiguration& configuration);
		virtual NetworkConfiguration operator()(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>()) = 0;

	public:
		std::vector<ConstrainedCouple> _parameters;
		std::vector<NetworkConfiguration> _features;
		std::vector<double> _rewards;
};

class RandomSampler : public Sampler {
	public:
		RandomSampler(const std::vector<ConstrainedCouple>&);
		void setSeed(unsigned int seed);

	protected:
		std::default_random_engine _generator;
};

class UniformSampler : public RandomSampler {
	public:
		UniformSampler(const std::vector<ConstrainedCouple>& parameters);
		virtual NetworkConfiguration randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
		virtual NetworkConfiguration operator()(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
};


using Gaussian = std::tuple<std::vector<double>, unsigned int, double>;
using StandardDeviations = std::vector<double>;

using GaussianT = std::tuple<std::vector<double>, double, double>;

class HGMTSampler : public RandomSampler {
	public:
		HGMTSampler(const std::vector<ConstrainedCouple>& parameters, std::vector<NetworkConfiguration> def, unsigned int nMax, double eps, double dist, unsigned int (*nTests)(std::vector<GaussianT>));
		void addGaussian(const NetworkConfiguration& center, double dist, double reward);
		void increaseStds();
		NetworkConfiguration rebuildConfiguration(const std::vector<double>& sampled) const;
		std::vector<double> normedSample(std::discrete_distribution<int>& discreteDist);
		virtual void addToBase(NetworkConfiguration features, double reward);
		virtual NetworkConfiguration randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
		virtual std::vector<NetworkConfiguration> randomActionsFromSample(unsigned int count, unsigned int maxAttempts, const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
		virtual NetworkConfiguration operator()(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());

	protected:
		unsigned int (*_nTests)(std::vector<GaussianT>);
		std::vector<GaussianT> _gaussians;
		double _eps;
		double _dist;
		unsigned int _nMax;
		unsigned int _testCounter;
};

enum DistanceMode { STD, CYCLE };
using Ring = std::tuple<std::vector<double>, double, double, double>;
class HCMSampler : public RandomSampler {
	public:
		HCMSampler(const std::vector<ConstrainedCouple>& parameters, std::vector<NetworkConfiguration> def, unsigned int nMax, double eps, unsigned int (*nTests)(std::vector<Ring>), DistanceMode dmode);
		void addRing(const NetworkConfiguration& center, double dist, double std, double reward);
		void shiftCenter();
		NetworkConfiguration rebuildConfiguration(const std::vector<double>& sampled, std::vector<unsigned int> location={}) const;
		void printRings() const;
		unsigned int updateThreshold() const;
		std::vector<double> normedSample(std::discrete_distribution<int>& discreteDist, double minimumLatticeRadius=0.0);
		virtual void addToBase(NetworkConfiguration features, double reward);
		virtual NetworkConfiguration randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
		virtual std::vector<NetworkConfiguration> randomActionsFromSample(unsigned int count, unsigned int maxAttempts, const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
		virtual bool hasForcedAction();
		virtual NetworkConfiguration takeForcedAction(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
		virtual void notifyDecision(const NetworkConfiguration& configuration);
		virtual void selectedAction(const NetworkConfiguration& configuration);
		virtual NetworkConfiguration operator()(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());

	protected:
		unsigned int (*_nTests)(std::vector<Ring>);
		std::vector<Ring> _circulars;
		double _eps;
		double _dist;
		unsigned int _nMax;
		unsigned int _testCounter;
		DistanceMode _dmode;
		bool _paperProfile;
		bool _authorProfile;
		bool _fixedAuthorProfile;
		bool _restartProfile;
		double _globalEps;
		double _batchExploreFraction;
		double _batchMinRadius;
		double _batchMaxRadius;
		unsigned int _restartPatience;
		unsigned int _restartBurst;
		unsigned int _restartMaxBursts;
		unsigned int _restartBurstsStarted;
		double _restartMinDelta;
		double _restartTargetReward;
		unsigned int _restartStagnation;
		unsigned int _restartRemaining;
		unsigned int _haltonIndex;
		double _restartBestReward;
		NetworkConfiguration _restartLastConfiguration;
		bool _restartDecisionPending;
		bool _restartLastWasForced;
		NetworkConfiguration _restartBestConfiguration;
};

#endif
