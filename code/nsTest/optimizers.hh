#ifndef __OPTIMIZERS_HH
#define __OPTIMIZERS_HH

#include <vector>
#include <set>
#include <string>
#include <map>
#include <chrono>
#include <fstream>
#include <random>

#include "samplers.hh"

using History = std::vector<std::tuple<NetworkConfiguration, double>>;
class Optimizer {
	public:
		Optimizer(Sampler* sampler, unsigned int testPeriod = 1);
		unsigned int getTestPeriod() const;
		virtual ~Optimizer();
		void showDecisions() const;
		virtual bool readyForAnother() const;
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual NetworkConfiguration optimize() = 0;
		void setSeed(unsigned int seed);

	protected:
		Sampler* _sampler;
		History _history;
		std::default_random_engine _generator;
		std::uniform_real_distribution<double> _distribution;
		unsigned int _testPeriod;
};

class IdleOptimizer : public Optimizer {
	public:
		IdleOptimizer();
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual NetworkConfiguration optimize();

	protected:
		NetworkConfiguration _chosen;
};

class RandomNeighborOptimizer : public Optimizer {
	public:
		RandomNeighborOptimizer(Sampler* sampler);
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual NetworkConfiguration optimize();

	protected:
		bool _firstCollection = true;
		bool _secondCollection = false;
		unsigned int _n = 10;
		unsigned int _counter = 0;
		NetworkConfiguration _chosen;
};

class EpsilonGreedyOptimizer : public Optimizer {
	public:
		EpsilonGreedyOptimizer(Sampler* sampler, double epsilon);
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual NetworkConfiguration optimize();

	protected:
		double _epsilon;
		std::map<NetworkConfiguration, double> _results;
};

using GammaNormalSample = std::tuple<double, double, double, double, std::vector<double>, bool>;
class ThompsonGammaNormalOptimizer : public Optimizer {
	public:
		ThompsonGammaNormalOptimizer(Sampler* sampler, unsigned int sampleSize, double eps, unsigned int to_ig = 0, bool chain = true);
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual bool readyForAnother() const;
		virtual NetworkConfiguration optimize();

	protected:
		std::map<NetworkConfiguration, GammaNormalSample> _gammaNormals;
		NetworkConfiguration _chosen;
		unsigned int _testLeft;
		double _epsilon;
		unsigned int _to_ig;
		bool _chain;
		double _acquisitionEpsilon;
		unsigned int _acquisitionPoolSize;
		unsigned int _acquisitionNovelDraws;
		unsigned int _acquisitionMaxAttempts;
		unsigned int _acquisitionColdStart;
		unsigned int _acquisitionDecisionCount;
};

class NeuralBanditOptimizer : public Optimizer {
	public:
		NeuralBanditOptimizer(Sampler* sampler, unsigned int sampleSize, double eps, unsigned int to_ig = 0, bool chain = true, unsigned int seed = 0);
		virtual ~NeuralBanditOptimizer();
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual bool readyForAnother() const;
		virtual NetworkConfiguration optimize();

		protected:
			bool nbBridgeEnabled() const;
			bool ensureBridgeConnected();
			void closeBridge();
			bool sendHello();
			bool sendObserve(const NetworkConfiguration& configuration, double reward);
				bool askChoiceFromBridge(NetworkConfiguration& chosen);
				bool askConfigFromBridge(NetworkConfiguration& chosen);
				NetworkConfiguration fallbackOptimize();
				bool addCandidate(const NetworkConfiguration& candidate, std::vector<NetworkConfiguration>& candidates) const;
				std::vector<NetworkConfiguration> buildCandidatePool();
				std::vector<double> normalizedConfiguration(const NetworkConfiguration& configuration) const;
				NetworkConfiguration configurationFromNormalized(const std::vector<double>& values) const;
			std::string traceConfiguration(const NetworkConfiguration& configuration) const;
			void traceObservation(const NetworkConfiguration& configuration, double reward);
			void traceDecision(const NetworkConfiguration& configuration);
			void traceFallback(const NetworkConfiguration& configuration);
			NetworkConfiguration referenceConfiguration() const;

			ThompsonGammaNormalOptimizer _fallback;
			NetworkConfiguration _chosen;
			unsigned int _testLeft;
			double _epsilon;
			unsigned int _to_ig;
			bool _chain;
				bool _bridgeWarned;
				std::string _host;
				std::string _port;
				unsigned int _candidatePoolSize;
				unsigned int _candidateMaxAttempts;
				unsigned int _candidateNovelDraws;
				bool _guideWithFallback;
				unsigned int _fallbackColdStartObservations;
			bool _directConfigMode;
			bool _rawObserveMode;
			unsigned int _nbObservationCount;
			bool _hasBestConfiguration;
			NetworkConfiguration _bestConfiguration;
			double _bestReward;
			unsigned int _seed;
			unsigned int _bridgeConnectCount;
			unsigned int _runtimeStep;
			int _bridgeFd;
			std::map<NetworkConfiguration, std::vector<double>> _pendingRewards;
			std::map<NetworkConfiguration, std::pair<double, unsigned int>> _configurationRewardStats;
			bool _traceEnabled;
			std::string _tracePath;
			std::ofstream _traceFile;
			unsigned int _traceStep;
	};

/** Distributed Bayesian optimization from Bardou and Begin, MSWiM'22.
 *
 * One Gaussian process is maintained per AP/agent over the joint
 * configuration of its default-configuration communication neighborhood.
 * Local Expected-Improvement prescriptions are combined by the marginal
 * median consensus from Equation (10) of the paper.
 */
class InspireOptimizer : public Optimizer {
	public:
		InspireOptimizer(
			Sampler* sampler,
			const std::vector<std::vector<unsigned int>>& neighborhoods,
			unsigned int seed = 0);
		virtual ~InspireOptimizer();
		virtual void addToBase(
			NetworkConfiguration configuration,
			double reward,
			bool forward=true,
			std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual NetworkConfiguration optimize();

	private:
		struct LocalObservation {
			std::vector<double> configuration;
			double reward;
		};

		struct GaussianProcessModel {
			std::vector<std::vector<double>> features;
			std::vector<double> labels;
			std::vector<std::vector<double>> cholesky;
			std::vector<double> alpha;
			double lengthScale = 0.35;
			double signalVariance = 1.0;
			bool valid = false;
		};

		std::vector<double> localConfiguration(unsigned int agent, const NetworkConfiguration& configuration) const;
		NetworkConfiguration randomLocalConfiguration(unsigned int agent);
		bool validPair(double sensitivity, double power) const;
		double matern32(const std::vector<double>& lhs, const std::vector<double>& rhs, double lengthScale) const;
		bool choleskyDecompose(std::vector<std::vector<double>>& matrix) const;
		std::vector<double> choleskySolve(const std::vector<std::vector<double>>& lower, const std::vector<double>& rhs) const;
		GaussianProcessModel fitModel(unsigned int agent);
		std::pair<double, double> predict(const GaussianProcessModel& model, const std::vector<double>& configuration) const;
		double expectedImprovement(const GaussianProcessModel& model, const std::vector<double>& configuration) const;
		std::vector<double> prescribe(unsigned int agent, const GaussianProcessModel& model);
		double marginalMedian(std::vector<double> values) const;
		void openTrace();
		void traceDecision(const std::vector<NetworkConfiguration>& prescriptions, const NetworkConfiguration& consensus);

		std::vector<std::vector<unsigned int>> _neighborhoods;
		std::vector<std::vector<LocalObservation>> _localHistories;
		NetworkConfiguration _current;
		std::mt19937 _rng;
		unsigned int _window;
		unsigned int _restarts;
		unsigned int _sweeps;
		unsigned int _hyperPeriod;
		std::vector<double> _lengthScales;
		unsigned int _step;
		bool _traceEnabled;
		std::string _tracePath;
		std::ofstream _traceFile;
};

using NormalParameters = std::tuple<double, double, unsigned int>;
class ThompsonNormalOptimizer : public Optimizer {
	public:
		ThompsonNormalOptimizer(Sampler* sampler, double eps);
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual NetworkConfiguration optimize();

	protected:
		std::map<NetworkConfiguration, NormalParameters> _normals;
		double _epsilon;
};

#endif
