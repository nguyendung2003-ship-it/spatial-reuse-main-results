#ifndef __SIMULATION_HH
#define __SIMULATION_HH

#include <string>
#include <vector>
#include <set>
#include <cstdint>
#include "unistd.h"
#include <sys/time.h>

#include "json.hh"

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"

#include "optimizers.hh"
#include "samplers.hh"

using namespace ns3;

enum Optim { IDLEOPT, EGREEDY, THOMP_GAMNORM, THOMP_NORM, MARGIN, RANDNEIGHBOR, NB, INSPIRE };
enum Samp { UNIF, HGM, HCM };
enum Reward { AD_HOC, CUMTP, LOGPF };
enum Entry { DEF, DEGA };
enum Dist { LOG, SQRT, N2, N4 };
enum ChannelWidth { MHZ_20, MHZ_40, MHZ_80 };
enum StationThroughput { NONE=0, LOW=1, MEDIUM=2, HIGH=3 };

class Simulation {
	public:
		Simulation(Optim oId, Samp sId, Reward r, Entry e, DistanceMode dmode, ChannelWidth cw, Json::Value topo, std::vector<StationThroughput> stations_throughputs, double duration, double testDuration, bool uplink, std::string outputName, NetworkConfiguration defaultConf = NetworkConfiguration());
		pid_t getPID() const;
		void readTopology(Json::Value topo);
		void storeMetrics();
		double rewardFromThroughputs();
		double rewardFromThroughputs(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables);
		double adHocReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables) const;
		double adHocTieBreakReward(const std::vector<std::vector<double>>& throughputs, const std::vector<std::vector<double>>& attainables) const;
		double logPfReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables);
		double cumulatedThroughputReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables);
		double cumulatedThroughputFromThroughputs();
		std::vector<std::tuple<double, unsigned int>> subrewardFromThroughputs();
		std::vector<std::tuple<double, unsigned int>> inspireSelfishRewards() const;
		std::vector<std::vector<unsigned int>> inspireNeighborhoods() const;
		double fairnessFromThroughputs();
		std::vector<NetworkConfiguration> findDiagonalEntryPoints(unsigned int n) const;
		std::vector<NetworkConfiguration> findDegreeEntryPoints(double criterion=0.5) const;
		std::vector<NetworkConfiguration> findNHDegreeEntryPoints(double criterion=0.5) const;
		NetworkConfiguration handleClusterizedConfiguration(const NetworkConfiguration& configuration);
		std::vector<double> getLogDistanceRSSIs() const;
		std::vector<double> apThroughputsFromThroughputs();
		std::vector<double> staThroughputsFromThroughputs();
		std::vector<double> staPersFromPers();
		std::vector<std::vector<double>> attainableThroughputs() const;
		void computeThroughputsAndErrors();
		void applyDemandPhase(unsigned int phase);
		std::vector<unsigned int> agentMembers(unsigned int agentIndex) const;
			std::vector<double> buildAgentObservation(unsigned int agentIndex);
			std::string agentObservationsToString();
			double localLogPfReward(unsigned int agentIndex, const std::vector<std::vector<double>>& attainables);
			double ppoObjectiveReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables);
			double localObjectiveReward(unsigned int agentIndex, const std::vector<std::vector<double>>& attainables);
			double starvedStationRatio(const std::vector<unsigned int>& apIndices, const std::vector<std::vector<double>>& attainables) const;
			double cooperativeAgentReward(unsigned int agentIndex, const std::vector<std::vector<double>>& attainables);
		bool ppoBridgeEnabled() const;
		bool ensurePpoBridgeConnected();
		void closePpoBridge();
		std::vector<std::tuple<int, int>> queryPpoBridge();
		std::vector<NetworkConfiguration> findEntryPoints(int v = -1) const;
		std::vector<std::vector<unsigned int>> extractConflicts(NetworkConfiguration conf) const;
		NetworkConfiguration projectConfiguration(NetworkConfiguration configuration) const;
		NetworkConfiguration applyDeltaActions(const NetworkConfiguration& base, const std::vector<std::tuple<int, int>>& deltas) const;
		NetworkConfiguration applyPpoActions(const NetworkConfiguration& base, const std::vector<std::tuple<int, int>>& actions) const;
		void setupNewConfiguration(NetworkConfiguration configuration);
		void endOfTest();
		double stationThroughputToInterval(StationThroughput stt, double duration) const;
		void stationsThroughputsToInterval(const std::vector<StationThroughput>& stations_throughputs, double duration);
		std::vector<std::vector<StationThroughput>> buildDynamicDemandPhases(const std::vector<StationThroughput>& base) const;
		std::string stateVectorToString(unsigned int index) const;
		static std::string configurationToString(const NetworkConfiguration& config);
		static int channelNumber(ChannelWidth cw);
		static bool parameterConstraint(double sens, double pow);
		static unsigned int numberOfSamples(std::vector<GaussianT> gaussians);
		static unsigned int numberOfSamples_HCM(std::vector<Ring> circulars);
		static WifiNetDevice* getWifiDevice(Node* node);
		static WifiMac* getMAC(Node* node);
		static WifiMacHeader createAdHocMacHeader(Node* from, Node* to);
		static WifiMode getWifiMode(Node* from, Node* to);
		static unsigned int getMCSValue(Node* from, Node* to);
		static std::string getMCSClass(Node* from, Node* to);
		static double pathLoss(std::tuple<double, double, double> source, std::tuple<double, double, double> target, double txPower);
		static double distance(std::tuple<double, double, double> source, std::tuple<double, double, double> target);
		static std::vector<double> attainableThroughputsFromChannel(ChannelWidth cw);

	protected:
		pid_t _pid;
		Reward _rewardType;
		bool _changed;
		double _testDuration;
		unsigned int _testCounter;
		unsigned int _warmup_tests;
		std::vector<double> _rewards;
		std::vector<double> _fairness;
		std::vector<double> _cumulatedThroughput;
		std::vector<NetworkConfiguration> _configurations;
		std::vector<NetworkConfiguration> _entryPoints;
		std::vector<std::vector<double>> _apThroughputs;
		std::vector<std::vector<double>> _staThroughputs;
		std::vector<std::vector<double>> _staPERs;
		std::vector<std::string> _stateVectors;
		std::vector<std::string> _agentStateVectors;
		std::vector<double> _intervals;
		std::vector<StationThroughput> _currentStationThroughputs;
		std::vector<std::vector<StationThroughput>> _dynamicDemandPhases;
		std::vector<double> _positionAPX;
		std::vector<double> _positionAPY;
		std::vector<double> _positionAPZ;
		std::vector<double> _positionStaX;
		std::vector<double> _positionStaY;
		std::vector<double> _positionStaZ;
		std::vector<unsigned int> _clustersAP;
		std::vector<std::vector<double>> _throughputs;
		std::vector<std::vector<double>> _pers;
		std::vector<std::vector<unsigned int>> _associations;
		std::vector<NetDeviceContainer> _devices;
		NetworkConfiguration _configuration;
		std::vector<std::tuple<int, int>> _previousDeltaActions;
		Optimizer* _optimizer;
		int _ppoBridgeFd = -1;
		bool _ppoBridgeWarned = false;
		bool _dynamicScenario = false;
		bool _mobility = false;
		bool _tcpTransport = false;
		bool _ppoAbsoluteActions = false;
		bool _phaseAwareObservations = false;
		double _ppoActionPenalty = 0.01;
		double _duration = 0.0;
		unsigned int _dynamicPhase = 0;
		NodeContainer _nodesAP;
		std::vector<NodeContainer> _nodesSta;
		std::vector<ApplicationContainer> _serversPerAp;
		std::vector<std::vector<unsigned int>> _lastRxPackets;
		std::vector<std::vector<unsigned int>> _lastLostPackets;
		std::vector<std::vector<uint64_t>> _lastRxBytes;
		ChannelWidth _channel_width;
		std::vector<double> _heAttainableThroughputs = std::vector<double>({
			120e6, 230e6, 330e6, 430e6, 610e6, 780e6, 850e6, 890e6, 1070e6, 1180e6, 1220e6, 1400e6
		});
		unsigned int _packetSize = 8 * 1464;
		int _defaultSensibility = -82;
		int _defaultPower = 20;
		bool _warmed = false;
		double _cumulative = 0.0;
		double _ema = -1.0;
};

#endif
