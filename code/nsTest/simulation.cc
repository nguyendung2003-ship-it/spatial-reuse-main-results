#include "simulation.hh"
#include "bridge_utils.hh"
#include "runtime_timing.hh"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>

namespace {
	double clampUnit(double value) {
		return std::max(0.0, std::min(1.0, value));
	}

	int deltaSign(double value) {
		if (value > 0.0) return 1;
		if (value < 0.0) return -1;
		return 0;
	}

	double normalizedSensitivity(double sensibility) {
		return clampUnit((sensibility + 82.0) / 20.0);
	}

	double normalizedPower(double txPower) {
		return clampUnit((txPower - 1.0) / 20.0);
	}

	double normalizedRssi(double rssi) {
		return clampUnit((rssi + 100.0) / 80.0);
	}

	double demandLevel(StationThroughput demand) {
		switch (demand) {
			case NONE: return 0.0;
			case LOW: return 1.0 / 3.0;
			case MEDIUM: return 2.0 / 3.0;
			case HIGH: return 1.0;
		}

		return 0.0;
	}

	std::string joinDoubles(const std::vector<double>& values) {
		std::ostringstream stream;
		for (unsigned int i = 0; i < values.size(); i++) {
			if (i > 0) stream << ",";
			stream << values[i];
		}

		return stream.str();
	}

	std::string joinDemands(const std::vector<StationThroughput>& values) {
		std::ostringstream stream;
		for (unsigned int i = 0; i < values.size(); i++) {
			if (i > 0) stream << ",";
			stream << static_cast<unsigned int>(values[i]);
		}

		return stream.str();
	}

	std::string envOrDefault(const char* name, const std::string& fallback) {
		const char* value = std::getenv(name);
		return value == nullptr || std::strlen(value) == 0 ? fallback : std::string(value);
	}

	double boundedEnvDouble(const char* name, double fallback, double minimum, double maximum) {
		const char* value = std::getenv(name);
		if (value == nullptr || std::strlen(value) == 0) return fallback;
		char* end = nullptr;
		double parsed = std::strtod(value, &end);
		if (end == value || *end != '\0' || !std::isfinite(parsed)) return fallback;
		return std::max(minimum, std::min(maximum, parsed));
	}

	std::string lowerString(std::string value) {
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
			return std::tolower(c);
		});
		return value;
	}

}

/**
 * Create a simulation in a dedicated process
 *
 * @param oId Optim the optimizer to use
 * @param sId Samp the sampler to use
 * @param r Reward the reward to use
 * @param topo Json::value the JSON representation of the network topology
 * @param duration double the simulation duration
 * @param testDuration double the test duration
 * @param outputName std::string the output file name
 * @param beta double the beta parameter for FSCORE reward
 */
Simulation::Simulation(Optim oId, Samp sId, Reward r, Entry e, DistanceMode dmode, ChannelWidth cw, Json::Value topo, std::vector<StationThroughput> stations_throughputs, double duration, double testDuration, bool uplink, std::string outputName, NetworkConfiguration defaultConf) : _rewardType(r), _changed(false), _testDuration(testDuration), _testCounter(0), _channel_width(cw) {
	this->_pid = fork();
	if (this->_pid == 0) {
		// Child process
		this->_dynamicScenario = std::getenv("NSTEST_DYNAMIC") != nullptr;
		this->_mobility = std::getenv("NSTEST_MOBILITY") != nullptr;
		this->_tcpTransport = lowerString(envOrDefault("NSTEST_TRANSPORT", "udp")) == "tcp";
		this->_duration = duration;
		this->_phaseAwareObservations = std::getenv("NSTEST_PHASE_OBS") != nullptr;
		this->_ppoActionPenalty = std::max(0.0, std::atof(envOrDefault("NSTEST_ACTION_PENALTY", "0.01").c_str()));
		std::string actionMode = lowerString(envOrDefault("NSTEST_ACTION_MODE", "delta"));
		this->_ppoAbsoluteActions = actionMode == "absolute" || actionMode == "abs";

		// Random generator
		struct timeval time;
    gettimeofday(&time,NULL);
		unsigned int seed = (time.tv_sec * 10) + (time.tv_usec / 10);
		if (std::getenv("NSTEST_SEED") != nullptr) {
			seed = std::max(1, std::atoi(std::getenv("NSTEST_SEED")));
		}
		unsigned int agentSeed = seed;
		if (std::getenv("NSTEST_AGENT_SEED") != nullptr) {
			agentSeed = std::max(1, std::atoi(std::getenv("NSTEST_AGENT_SEED")));
		}
		RngSeedManager::SetSeed(seed);
		std::srand(seed);
		std::default_random_engine randomInitGenerator(seed + 7919);

		// Topology
		this->readTopology(topo);

		// Default configuration
		if (this->_positionAPX.size() != defaultConf.size()) {
			defaultConf.clear();
			for (unsigned int i = 0; i < this->_positionAPX.size(); i++) {
				defaultConf.push_back(std::tuple<double, double>(this->_defaultSensibility, this->_defaultPower));
			}
		}
		if (std::getenv("NSTEST_STATIC_SP") != nullptr) {
			std::string staticConfig = envOrDefault("NSTEST_STATIC_SP", "");
			std::replace(staticConfig.begin(), staticConfig.end(), ':', ',');
			std::stringstream stream(staticConfig);
			std::string sensText, powerText;
			if (std::getline(stream, sensText, ',') && std::getline(stream, powerText, ',')) {
				double sens = std::atof(sensText.c_str());
				double power = std::atof(powerText.c_str());
				defaultConf = NetworkConfiguration(this->_positionAPX.size(), std::make_tuple(sens, power));
				defaultConf = this->projectConfiguration(defaultConf);
			} else {
				std::cerr << "ns3-static: invalid NSTEST_STATIC_SP='" << staticConfig
									<< "', expected sens,power" << std::endl;
			}
		}
		if (std::getenv("NSTEST_STATIC_CONFIG") != nullptr) {
			std::string staticConfig = envOrDefault("NSTEST_STATIC_CONFIG", "");
			std::replace(staticConfig.begin(), staticConfig.end(), ':', ',');
			staticConfig.erase(
				std::remove_if(staticConfig.begin(), staticConfig.end(), [](char c) {
					return c == '(' || c == ')' || std::isspace(static_cast<unsigned char>(c));
				}),
				staticConfig.end());
			std::stringstream entries(staticConfig);
			std::string entry;
			NetworkConfiguration parsed;
			while (std::getline(entries, entry, ';')) {
				std::stringstream pair(entry);
				std::string sensText, powerText;
				if (!std::getline(pair, sensText, ',') || !std::getline(pair, powerText, ',')) {
					parsed.clear();
					break;
				}
				parsed.push_back(std::make_tuple(std::atof(sensText.c_str()), std::atof(powerText.c_str())));
			}
			if (parsed.size() == this->_positionAPX.size()) {
				defaultConf = this->projectConfiguration(parsed);
			} else {
				std::cerr << "ns3-static: invalid NSTEST_STATIC_CONFIG; expected "
								<< this->_positionAPX.size() << " semicolon-separated sens,power pairs"
								<< std::endl;
			}
		}

		// Index of channel to use during the simulation
		int channel = Simulation::channelNumber(this->_channel_width), numberOfAPs = this->_positionAPX.size(), numberOfStas = 0;
		for (std::vector<unsigned int> assocs: this->_associations)
			numberOfStas += assocs.size();

		double warmup_time = 2 * this->_testDuration;
		this->_warmup_tests = ceil(warmup_time / this->_testDuration);

		double applicationStart = 2.0, applicationEnd = applicationStart + duration + this->_warmup_tests * this->_testDuration;

		// ns-3 expires ALIVE ARP entries after 120 seconds by default.  The
		// original experiments also lasted 120 seconds, so their initial ARP
		// warm-up remained valid for the whole measured horizon.  Longer runs
		// otherwise cross that boundary: some dense/HCM configurations cannot
		// reliably refresh the broadcast ARP exchange, causing active UDP flows
		// to disappear abruptly even when the WLAN configuration is fixed.
		// Keep the cache alive beyond this simulation so extending the learning
		// horizon does not change the traffic workload at t=120 s.
		double arpAliveTimeout = boundedEnvDouble(
			"NSTEST_ARP_ALIVE_TIMEOUT",
			applicationEnd + 1.0,
			1.0,
			24.0 * 60.0 * 60.0);
		Config::SetDefault(
			"ns3::ArpCache::AliveTimeout",
			TimeValue(Seconds(arpAliveTimeout)));

		// Adapt interval according to the current traffic demand phase.
		this->_dynamicDemandPhases = this->buildDynamicDemandPhases(stations_throughputs);
		this->applyDemandPhase(0);
		// for (int i = 0; i < numberOfAPs; i++) intervalsCross[i] = this->_intervalCross * this->_associations[i].size();

		// APs creation and configuration
		// At the start, they're all configured with 802.11 default conf
		this->_nodesAP.Create(numberOfAPs);
		std::vector<YansWifiPhyHelper> wifiPhy(numberOfAPs); // One PHY for each AP
		for(int i = 0; i < numberOfAPs; i++) {
			// wifiPhy[i] = YansWifiPhyHelper::Default();
			wifiPhy[i].Set("Antennas", UintegerValue(2));
			// 2 spatial streams to support htMcs from 8 to 15 with short GI
			wifiPhy[i].Set("MaxSupportedTxSpatialStreams", UintegerValue(2));
			wifiPhy[i].Set("MaxSupportedRxSpatialStreams", UintegerValue(2));
			wifiPhy[i].Set("ChannelNumber", UintegerValue(channel));
			wifiPhy[i].Set("RxSensitivity", DoubleValue(std::get<0>(defaultConf[i])));
			wifiPhy[i].Set("CcaEdThreshold", DoubleValue(std::get<0>(defaultConf[i])));
			wifiPhy[i].Set("TxPowerStart", DoubleValue(std::get<1>(defaultConf[i])));
			wifiPhy[i].Set("TxPowerEnd", DoubleValue(std::get<1>(defaultConf[i])));
		}

		// Stations creation and configuration
		this->_nodesSta = std::vector<NodeContainer>(numberOfAPs);
		for(int i = 0; i < numberOfAPs; i++) this->_nodesSta[i].Create(this->_associations[i].size());
		// One phy for every station
		YansWifiPhyHelper staWifiPhy; // = YansWifiPhyHelper::Default();
		staWifiPhy.Set("Antennas", UintegerValue(2));
		staWifiPhy.Set("MaxSupportedTxSpatialStreams", UintegerValue(2));
		staWifiPhy.Set("MaxSupportedRxSpatialStreams", UintegerValue(2));
		staWifiPhy.Set("ChannelNumber", UintegerValue(channel));
		staWifiPhy.Set("RxSensitivity", DoubleValue(this->_defaultSensibility));
		staWifiPhy.Set("CcaEdThreshold", DoubleValue(this->_defaultSensibility));

		// Propagation model, same for everyone
		YansWifiChannelHelper wifiChannel;
		wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
		wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel");
		Ptr<YansWifiChannel> channelPtr = wifiChannel.Create ();
		// Attribution to stations and models
		staWifiPhy.SetChannel(channelPtr);
		for(int i = 0; i < numberOfAPs; i++) wifiPhy[i].SetChannel(channelPtr);

		// 802.11ax protocol
		WifiHelper wifi;
		wifi.SetStandard (WIFI_STANDARD_80211ax_5GHZ );
		std::string rateManager = lowerString(envOrDefault("NSTEST_RATE_MANAGER", "minstrel"));
		if (rateManager == "constant" || rateManager == "constant-rate" || rateManager == "constantrate") {
			wifi.SetRemoteStationManager(
				"ns3::ConstantRateWifiManager",
				"DataMode", StringValue(envOrDefault("NSTEST_CONSTANT_DATA_MODE", "HeMcs4")),
				"ControlMode", StringValue(envOrDefault("NSTEST_CONSTANT_CONTROL_MODE", "HeMcs0")));
		} else if (rateManager == "ideal" || rateManager == "ideal-wifi" || rateManager == "idealwifimanager") {
			wifi.SetRemoteStationManager("ns3::IdealWifiManager");
		} else {
			wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");
		}


		// Configure Infrastructure mode and SSID
		std::vector<NetDeviceContainer> devices(numberOfAPs);//Un groupe de Sta par AP
		Ssid ssid = Ssid ("ns380211");

		// Mac for Stations
		WifiMacHelper wifiMac;
		wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
		for (int i = 0; i < numberOfAPs; i++) devices[i] = wifi.Install(staWifiPhy, wifiMac, this->_nodesSta[i]);
		// Mac for APs
		this->_devices = std::vector<NetDeviceContainer>(numberOfAPs);
		wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
		for (int i = 0; i < numberOfAPs; i++) this->_devices[i] = wifi.Install(wifiPhy[i], wifiMac, this->_nodesAP.Get(i));

		// Mobility for devices. APs remain fixed; dynamic scenarios make STAs drift.
		MobilityHelper apMobility;
		Ptr<ListPositionAllocator> apPositionAlloc = CreateObject<ListPositionAllocator>();
		for (int i = 0; i < numberOfAPs; i++) apPositionAlloc->Add(Vector(this->_positionAPX[i], this->_positionAPY[i], this->_positionAPZ[i]));
		apMobility.SetPositionAllocator(apPositionAlloc);
		apMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
		apMobility.Install(this->_nodesAP);

		double minX = *std::min_element(this->_positionStaX.begin(), this->_positionStaX.end()),
					 maxX = *std::max_element(this->_positionStaX.begin(), this->_positionStaX.end()),
					 minY = *std::min_element(this->_positionStaY.begin(), this->_positionStaY.end()),
					 maxY = *std::max_element(this->_positionStaY.begin(), this->_positionStaY.end());
		for (double x: this->_positionAPX) {
			minX = std::min(minX, x);
			maxX = std::max(maxX, x);
		}
		for (double y: this->_positionAPY) {
			minY = std::min(minY, y);
			maxY = std::max(maxY, y);
		}
		double mobilityMargin = 10.0;
		Rectangle mobilityBounds(minX - mobilityMargin, maxX + mobilityMargin, minY - mobilityMargin, maxY + mobilityMargin);
		for(int i = 0; i < numberOfAPs; i++) {
			MobilityHelper staMobility;
			Ptr<ListPositionAllocator> staPositionAlloc = CreateObject<ListPositionAllocator>();
			for (unsigned int j = 0 ; j < this->_associations[i].size(); j++) {
				unsigned int staId = this->_associations[i][j];
				staPositionAlloc->Add(Vector(this->_positionStaX[staId], this->_positionStaY[staId], this->_positionStaZ[staId]));
			}
			staMobility.SetPositionAllocator(staPositionAlloc);
			if (this->_mobility) {
				staMobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
																		 "Bounds", RectangleValue(mobilityBounds),
																		 "Speed", StringValue("ns3::UniformRandomVariable[Min=0.2|Max=1.4]"),
																		 "Distance", DoubleValue(5.0));
			} else {
				staMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
			}
			staMobility.Install(this->_nodesSta[i]);
		}

		//IP stack and addresses
		InternetStackHelper internet;
		for(int i = 0; i < numberOfAPs; i++) internet.Install(this->_nodesSta[i]);
		internet.Install(this->_nodesAP);

		Ipv4AddressHelper ipv4;
		ipv4.SetBase("10.1.0.0", "255.255.0.0");
		Ipv4InterfaceContainer apInterfaces;
		for(int i = 0; i < numberOfAPs; i++) {
			apInterfaces = ipv4.Assign(this->_devices[i]);
			ipv4.Assign(devices[i]);
		}

		// Traffic sinks are installed on all stations. PacketSink counts TCP
		// payload bytes; UdpServer retains the packet-loss measurement for UDP.
		uint16_t port = 4000;
		UdpServerHelper udpServer(port);
		PacketSinkHelper tcpSink("ns3::TcpSocketFactory",
			InetSocketAddress(Ipv4Address::GetAny(), port));
		this->_serversPerAp = std::vector<ApplicationContainer>(numberOfAPs);
		for(int i = 0; i < numberOfAPs; i++) {
			ApplicationContainer apps = this->_tcpTransport
				? tcpSink.Install(this->_nodesSta[i])
				: udpServer.Install(this->_nodesSta[i]);
			apps.Start(Seconds(0));
			apps.Stop(Seconds(applicationEnd));
			this->_serversPerAp[i] = apps;
		}

		if (uplink) {
				ApplicationContainer apps = this->_tcpTransport
					? tcpSink.Install(this->_nodesAP)
					: udpServer.Install(this->_nodesAP);
				apps.Start(Seconds(0));
				apps.Stop(Seconds(applicationEnd));
			}

		// Each demand phase has its own sender, so TCP and UDP receive the same
		// offered payload rate and traffic changes at the same phase boundaries.
			UdpClientHelper clientCT;
			UdpClientHelper clientCTARP;
			auto installTcpFlow = [this, port](Ptr<Node> sender, Ipv4Address destination,
					double payloadBitRate, double start, double stop) {
				OnOffHelper client("ns3::TcpSocketFactory", InetSocketAddress(destination, port));
				client.SetAttribute("DataRate", DataRateValue(DataRate(
					static_cast<uint64_t>(std::max(1.0, std::round(payloadBitRate))))));
				client.SetAttribute("PacketSize", UintegerValue(this->_packetSize / 8));
				client.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1000000]"));
				client.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
				ApplicationContainer apps = client.Install(sender);
				apps.Start(Seconds(start));
				apps.Stop(Seconds(stop));
			};
			unsigned sta_idx = 0;
			unsigned int trafficPhases = this->_dynamicScenario ? this->_dynamicDemandPhases.size() : 1;
			double trafficPhaseDuration = (applicationEnd - applicationStart) / trafficPhases;

			double step = applicationStart / (numberOfStas + 1.0);
			for(int i = 0; i < numberOfAPs; i++) {
					for(unsigned int j = 0; j < this->_associations[i].size(); j++) {
							unsigned int globalStaId = this->_associations[i][j];
							// IPv4 instance of the station
							Ipv4Address addr = this->_nodesSta[i].Get(j)
								->GetObject<Ipv4>()
							->GetAddress(1, 0) // Loopback (1-0)
							.GetLocal();

						if (!this->_tcpTransport) {
							// UDP's original ARP prefill stays unchanged. TCP's connection
							// setup resolves ARP during the normal warmup interval.
							clientCTARP.SetAttribute("RemoteAddress", AddressValue(addr));
							clientCTARP.SetAttribute("RemotePort", UintegerValue(port));
							clientCTARP.SetAttribute("MaxPackets", UintegerValue(2));
							clientCTARP.SetAttribute("Interval", TimeValue(Seconds(step / 2.0)));
							clientCTARP.SetAttribute("PacketSize", UintegerValue(this->_packetSize / 8));
							ApplicationContainer apps_arp = clientCTARP.Install(this->_nodesAP.Get(i));
							apps_arp.Start(Seconds(sta_idx * step));
							apps_arp.Stop(Seconds((sta_idx + 1) * step));
						}
							sta_idx++;

							for (unsigned int phase = 0; phase < trafficPhases; phase++) {
								StationThroughput demand = this->_dynamicDemandPhases[phase][globalStaId];
								if (demand == NONE) continue;

								double phaseStart = this->_dynamicScenario ? applicationStart + phase * trafficPhaseDuration : applicationStart,
											 phaseStop = this->_dynamicScenario ? std::min(applicationEnd, phaseStart + trafficPhaseDuration) : applicationEnd,
											 interval = this->stationThroughputToInterval(demand, duration);
								if (phaseStop <= phaseStart) continue;

								if (this->_tcpTransport) {
									installTcpFlow(this->_nodesAP.Get(i), addr,
										this->_packetSize / interval, phaseStart, phaseStop);
								} else {
									clientCT.SetAttribute("RemoteAddress", AddressValue(addr));
									clientCT.SetAttribute("RemotePort", UintegerValue(port));
									clientCT.SetAttribute("MaxPackets", UintegerValue(1e9));
									clientCT.SetAttribute("Interval", TimeValue(Seconds(interval)));
									clientCT.SetAttribute("PacketSize", UintegerValue(this->_packetSize / 8));
									ApplicationContainer apps = clientCT.Install(this->_nodesAP.Get(i));
									apps.Start(Seconds(phaseStart));
									apps.Stop(Seconds(phaseStop));
								}

								// If uplink, installation the other way around with smaller demand.
								if (uplink) {
									Ipv4Address apAddr = this->_nodesAP.Get(i)
										->GetObject<Ipv4>()
										->GetAddress(1, 0) // Loopback (1-0)
										.GetLocal();

									if (this->_tcpTransport) {
										installTcpFlow(this->_nodesSta[i].Get(j), apAddr,
											this->_packetSize / (15.0 * interval), phaseStart, phaseStop);
									} else {
										clientCT.SetAttribute("RemoteAddress", AddressValue(apAddr));
										clientCT.SetAttribute("RemotePort", UintegerValue(port));
										clientCT.SetAttribute("MaxPackets", UintegerValue(1e9));
										clientCT.SetAttribute("Interval", TimeValue(Seconds(15.0 * interval)));
										clientCT.SetAttribute("PacketSize", UintegerValue(this->_packetSize / 8));
										ApplicationContainer apps = clientCT.Install(this->_nodesSta[i].Get(j));
										apps.Start(Seconds(phaseStart));
										apps.Stop(Seconds(phaseStop));
									}
								}
							}
					}
			}

			if (this->_dynamicScenario) {
				for (unsigned int phase = 0; phase < trafficPhases; phase++) {
					Simulator::Schedule(Seconds(applicationStart + phase * trafficPhaseDuration), &Simulation::applyDemandPhase, this, phase);
				}
			}

			Simulator::Stop(Seconds(applicationEnd+0.01));

		// Optimization relative objects
		// Init callback for configuration changes
		switch (e) {
			case DEF: this->_entryPoints = {defaultConf}; break;
		case DEGA:
			this->_entryPoints = this->findDegreeEntryPoints();
			std::vector<NetworkConfiguration> confs_nh = this->findNHDegreeEntryPoints(),
														def = {defaultConf};
			std::string hcmProfile = lowerString(envOrDefault("NSTEST_HCM_PROFILE", "batch"));
			bool paperHcm = sId == HCM && (hcmProfile == "paper" || hcmProfile == "paper-batch" || hcmProfile == "paper_batch");
			bool defaultFirstHcm = sId == HCM && (
				hcmProfile == "author-restart" || hcmProfile == "author_restart" ||
				hcmProfile == "author-fixed-restart" || hcmProfile == "author_fixed_restart");
			if (defaultFirstHcm) {
				std::vector<NetworkConfiguration> degreeEntries = this->_entryPoints;
				this->_entryPoints = def;
				this->_entryPoints.insert(this->_entryPoints.end(), degreeEntries.begin(), degreeEntries.end());
				this->_entryPoints.insert(this->_entryPoints.end(), confs_nh.begin(), confs_nh.end());
			} else {
				if (!paperHcm)
					this->_entryPoints.insert(this->_entryPoints.end(), confs_nh.begin(), confs_nh.end());
				this->_entryPoints.insert(this->_entryPoints.end(), def.begin(), def.end());
			}
			break;
		}

		// for (NetworkConfiguration nc: this->_entryPoints) {
		// 	for (std::tuple<double, double> t: nc) std::cout << "(" << std::get<0>(t) << "," << std::get<1>(t) << "),";
		// 	std::cout << std::endl;
		// }

		if (sId != HCM && sId != HGM)
			this->_testCounter = this->_entryPoints.size();
		unsigned int nParams = std::set<unsigned int>(this->_clustersAP.begin(), this->_clustersAP.end()).size();
		if (std::getenv("NSTEST_RANDOM_INIT") != nullptr && nParams > 0) {
			std::uniform_int_distribution<int> sensitivityDist(-82, -63);
			NetworkConfiguration randomEntry;
			for (unsigned int i = 0; i < nParams; i++) {
				int sensitivity = sensitivityDist(randomInitGenerator);
				int maxPower = std::max(1, std::min(21, -62 - sensitivity));
				std::uniform_int_distribution<int> powerDist(1, maxPower);
				randomEntry.push_back(std::make_tuple(sensitivity, powerDist(randomInitGenerator)));
			}
			this->_entryPoints = {randomEntry};
		}
		this->_configuration = this->_entryPoints[0];
		this->_previousDeltaActions = std::vector<std::tuple<int, int>>(this->_configuration.size(), std::make_tuple(0, 0));

		this->setupNewConfiguration(this->_configuration);
		Simulator::Schedule(Seconds(applicationStart+testDuration), &Simulation::endOfTest, this);

		// Init containers for throughput calculation
		for (int i = 0; i < numberOfAPs; i++) {
			unsigned int nStas = this->_associations[i].size();
			this->_throughputs.push_back(std::vector<double>(nStas, 0));
			this->_pers.push_back(std::vector<double>(nStas, 0));
			this->_lastRxPackets.push_back(std::vector<unsigned int>(nStas, 0));
			this->_lastLostPackets.push_back(std::vector<unsigned int>(nStas, 0));
			this->_lastRxBytes.push_back(std::vector<uint64_t>(nStas, 0));
		}

		// Parameters to optimize
		std::vector<ConstrainedCouple> parameters(nParams);
		for (unsigned int i = 0; i < nParams; i++) {
			parameters[i] = ConstrainedCouple(SingleParameter(-82, -62, 1), SingleParameter(1, 21, 1), &parameterConstraint);
		}

		// Sampler to use
		Sampler* sampler = nullptr;
		switch (sId) {
			case UNIF: sampler = new UniformSampler(parameters); break;
			case HGM: sampler = new HGMTSampler(parameters, this->_entryPoints, 6, 1.0 / (numberOfStas + 1.0), 1.0, &numberOfSamples); break;
			case HCM: sampler = new HCMSampler(parameters, this->_entryPoints, 6, 1.0 / (numberOfStas + 1.0), &numberOfSamples_HCM, dmode); break;
		}

		// Optimizer to use
			switch (oId) {
			case IDLEOPT: this->_optimizer = new IdleOptimizer(); break;
			case EGREEDY: this->_optimizer = new EpsilonGreedyOptimizer(sampler, 0.1); break;
			case THOMP_GAMNORM: this->_optimizer = new ThompsonGammaNormalOptimizer(sampler, 3, 0.1, 0, sId != HGM); break;
			case THOMP_NORM: this->_optimizer = new ThompsonNormalOptimizer(sampler, 0.1); break;
			case MARGIN: this->_optimizer = nullptr; break;
			case RANDNEIGHBOR: this->_optimizer = new RandomNeighborOptimizer(sampler); break;
			case NB: this->_optimizer = new NeuralBanditOptimizer(sampler, 3, 0.1, 0, sId != HGM, agentSeed); break;
			case INSPIRE: this->_optimizer = new InspireOptimizer(sampler, this->inspireNeighborhoods(), agentSeed); break;
		}

		std::cout << "ns3-debug: the simulation begins" << std::endl;
		Simulator::Run();

		Simulator::Destroy();

		// Stringstream for vector data
		std::ofstream myfile;
		myfile.open ("./scratch/nsTest/data/" + outputName);
		myfile << "rew\tfair\tcum\taps\tstas\tpers\tconf\tstate\tagent_state" << std::endl;
		for (unsigned int i = 0; i < this->_rewards.size(); i++) {
			std::stringstream aps, stas, pers;
			std::string delimiter = ",";
			copy(this->_apThroughputs[i].begin(), this->_apThroughputs[i].end(), std::ostream_iterator<double>(aps, delimiter.c_str()));
			copy(this->_staThroughputs[i].begin(), this->_staThroughputs[i].end(), std::ostream_iterator<double>(stas, delimiter.c_str()));
			copy(this->_staPERs[i].begin(), this->_staPERs[i].end(), std::ostream_iterator<double>(pers, delimiter.c_str()));

			std::string apsData = aps.str();
			std::string stasData = stas.str();
			std::string persData = pers.str();
			apsData = apsData.substr(0, apsData.size() - 1);
			stasData = stasData.substr(0, stasData.size() - 1);
			persData = persData.substr(0, persData.size() - 1);

			myfile << this->_rewards[i] << "\t" << this->_fairness[i] << "\t" << this->_cumulatedThroughput[i] << "\t"
						 << apsData << "\t" << stasData << "\t" << persData << "\t"
						 << this->configurationToString(this->_configurations[i]) << "\t"
						 << this->_stateVectors[i] << "\t"
						 << this->_agentStateVectors[i] << std::endl;
		}
		myfile.close();

		// Free sampler and optimizer
		this->closePpoBridge();
		delete sampler;
		if (this->_optimizer != nullptr)
			delete this->_optimizer;

		exit(0);
	}
}

/**
 * Return the PID of the simulation
 *
 * @return the PID of the simulation
 */
pid_t Simulation::getPID() const {
	return this->_pid;
}

/**
 * Compute the adequate reward from throughputs of STAs
 *
 * @return the adequate reward computation
 */
double Simulation::rewardFromThroughputs() {
	std::vector<std::vector<double>> attainables = this->attainableThroughputs();
	switch (this->_rewardType) {
		case AD_HOC: return this->adHocReward(this->_throughputs, attainables); break;
		case CUMTP: return this->cumulatedThroughputReward(this->_throughputs, attainables); break;
		case LOGPF: return this->logPfReward(this->_throughputs, attainables); break;
	}

	return -1;
}

/**
 * Compute the adequate reward from throughputs of STAs
 *
 * @return the adequate reward computation
 */
double Simulation::rewardFromThroughputs(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables) {
	switch (this->_rewardType) {
		case AD_HOC: return this->adHocReward(throughputs, attainables); break;
		case CUMTP: return this->cumulatedThroughputReward(throughputs, attainables); break;
		case LOGPF: return this->logPfReward(throughputs, attainables); break;
	}

	return -1;
}

/**
 * This function is called at each end of test
 */
void Simulation::endOfTest() {
	this->_testCounter++;

  // Compute the throughput
  this->computeThroughputsAndErrors();

	if (!this->_warmed && this->_testCounter >= this->_warmup_tests) {
		this->_warmed = true;
		this->_testCounter = 1;
	}

	if (this->_warmed) {
		// Store metrics
		this->storeMetrics();
		// Compute reward accordingly
		double rew = this->rewardFromThroughputs();
		double optimizerReward = rew;
		static const double adHocTieBreakWeight =
			boundedEnvDouble("NSTEST_ADHOC_TIEBREAK_WEIGHT", 0.0, 0.0, 0.05);
		if (this->_rewardType == AD_HOC && adHocTieBreakWeight > 0.0) {
			// Preserve the published ADHOC metric in every output/EMA/regret field.
			// Only the optimizer sees this bounded continuous secondary signal.  Its
			// small fixed weight cannot dominate a one-station ADHOC tier change.
			optimizerReward += adHocTieBreakWeight * this->adHocTieBreakReward(
				this->_throughputs, this->attainableThroughputs());
		}
		std::vector<std::tuple<double, unsigned int>> subrews =
			dynamic_cast<InspireOptimizer*>(this->_optimizer) != nullptr
				? this->inspireSelfishRewards()
				: this->subrewardFromThroughputs();

		double alpha = 2.0 / 31.0;
		if (this->_ema < 0.0) this->_ema = rew;
		else this->_ema = alpha * rew + (1 - alpha) * this->_ema;
		this->_cumulative += rew;

		// std::cout << "Reward at t = " << this->_testCounter * this->_testDuration << ": " << rew << " (Cum: " << this->_cumulative << ", EMA: " << this->_ema << ")" << std::endl << std::endl;

		// Add config and reward to sampler
		NetworkConfiguration configuration = this->_configuration;
		if (this->ppoBridgeEnabled()) {
			configuration = this->applyPpoActions(this->_configuration, this->queryPpoBridge());
		} else if (this->_optimizer != nullptr) {
			{
				RuntimeTiming::Scoped updateTimer("training_update", this->_testCounter);
				this->_optimizer->addToBase(this->_configuration, optimizerReward, true, subrews);
			}

			// Use the optimizer to get another configuration
			if (this->_testCounter / this->_optimizer->getTestPeriod() < this->_entryPoints.size() && this->_optimizer->readyForAnother()) {
				configuration = this->_entryPoints[this->_testCounter / this->_optimizer->getTestPeriod()];
			} else {
				RuntimeTiming::Scoped decisionTimer("inference_end_to_end", this->_testCounter);
				configuration = this->_optimizer->optimize();
			}
		} else if (!this->_changed) {
			this->_changed = true;
			configuration = {};
			double margin = 5;
			std::vector<double> rssis = this->getLogDistanceRSSIs();
			double obsspd_min = -82, obsspd_max = -62,
					tx_min = 1, tx_max = 21;
			for (double rssi: rssis) {
				double sens = std::max(obsspd_min, std::min(obsspd_max, round(rssi - margin)));
				double tx = std::max(tx_min, std::min(tx_max, -62 - sens));
				configuration.push_back(std::make_tuple(sens, tx));
			}
		}

		// for (std::tuple<double, double> t: configuration) {
		// 	std::cout << "(" << std::get<0>(t) << ", " << std::get<1>(t) << "), ";
		// }
		// std::cout << std::endl;

		// Set up the new configuration
		setupNewConfiguration(configuration);
	}

  // Next scheduling for recurrent callback
  Simulator::Schedule(Seconds(this->_testDuration), &Simulation::endOfTest, this);
}

/**
 * Build the network configuration according to the topo clusters
 *
 * @param configuration NetworkConfiguration the network configuration clusterized
 *
 * @return the unclusterized network configuration
 */
NetworkConfiguration Simulation::handleClusterizedConfiguration(const NetworkConfiguration& configuration) {
	unsigned int numberAps = this->_positionAPX.size();
	NetworkConfiguration unclusterized(numberAps);
	unsigned int k = 0;
	for (std::tuple<double, double> couple: configuration) {
		for (unsigned int i = 0; i < numberAps; i++) {
			if (this->_clustersAP[i] == k) {
				unclusterized[i] = couple;
			}
		}
		k++;
	}

	return unclusterized;
}

/**
 * Store the network metrics in dedicated containers.
 * This method should be called AFTER computeThroughputsAndErrors.
 */
void Simulation::storeMetrics() {
	this->_configurations.push_back(this->handleClusterizedConfiguration(this->_configuration));

	double rew = this->rewardFromThroughputs();
	this->_rewards.push_back(rew);
	// std::cout << "Reward: " << this->rewardFromThroughputs() << " vs. " << this->otherRewardFromThroughputs() << std::endl;

	double fairness = this->fairnessFromThroughputs();
	this->_fairness.push_back(fairness);
	// std::cout << "Fairness: " << fairness << std::endl;

	double cumThrough = this->cumulatedThroughputFromThroughputs();
	this->_cumulatedThroughput.push_back(cumThrough);
	// std::cout << "CumThrough: " << cumThrough << std::endl << std::endl;

	this->_apThroughputs.push_back(this->apThroughputsFromThroughputs());
	this->_staThroughputs.push_back(this->staThroughputsFromThroughputs());
	this->_staPERs.push_back(this->staPersFromPers());
	this->_stateVectors.push_back(this->stateVectorToString(this->_rewards.size() - 1));
	this->_agentStateVectors.push_back(this->agentObservationsToString());
}

std::vector<unsigned int> Simulation::agentMembers(unsigned int agentIndex) const {
	std::vector<unsigned int> members;
	for (unsigned int ap = 0; ap < this->_clustersAP.size(); ap++) {
		if (this->_clustersAP[ap] == agentIndex) {
			members.push_back(ap);
		}
	}
	if (members.empty() && agentIndex < this->_positionAPX.size()) {
		members.push_back(agentIndex);
	}

	return members;
}

double Simulation::starvedStationRatio(const std::vector<unsigned int>& apIndices, const std::vector<std::vector<double>>& attainables) const {
	unsigned int starved = 0, active = 0;
	for (unsigned int ap: apIndices) {
		if (ap >= this->_throughputs.size() || ap >= attainables.size()) continue;
		unsigned int activeOnAp = 0;
		for (double attainable: attainables[ap]) {
			if (attainable > 0.0) activeOnAp++;
		}
		if (activeOnAp == 0) continue;

		for (unsigned int sta = 0; sta < this->_throughputs[ap].size(); sta++) {
			if (attainables[ap][sta] <= 0.0) continue;
			active++;
			if (this->_throughputs[ap][sta] < 0.1 * attainables[ap][sta] / activeOnAp) {
				starved++;
			}
		}
	}

	return active > 0 ? ((double) starved) / active : 0.0;
}

double Simulation::localLogPfReward(unsigned int agentIndex, const std::vector<std::vector<double>>& attainables) {
	std::vector<std::vector<double>> localThroughputs, localAttainables;
	for (unsigned int ap: this->agentMembers(agentIndex)) {
		if (ap < this->_throughputs.size() && ap < attainables.size()) {
			localThroughputs.push_back(this->_throughputs[ap]);
			localAttainables.push_back(attainables[ap]);
		}
	}

	return this->logPfReward(localThroughputs, localAttainables);
}

double Simulation::ppoObjectiveReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables) {
	double reward = this->rewardFromThroughputs(throughputs, attainables);
	if (this->_rewardType != CUMTP) return reward;

	double reference = this->cumulatedThroughputReward(attainables, attainables);
	return reference > 0.0 ? reward / reference : 0.0;
}

double Simulation::localObjectiveReward(unsigned int agentIndex, const std::vector<std::vector<double>>& attainables) {
	std::vector<std::vector<double>> localThroughputs, localAttainables;
	for (unsigned int ap: this->agentMembers(agentIndex)) {
		if (ap < this->_throughputs.size() && ap < attainables.size()) {
			localThroughputs.push_back(this->_throughputs[ap]);
			localAttainables.push_back(attainables[ap]);
		}
	}

	return this->ppoObjectiveReward(localThroughputs, localAttainables);
}

double Simulation::cooperativeAgentReward(unsigned int agentIndex, const std::vector<std::vector<double>>& attainables) {
	std::vector<unsigned int> allAps(this->_throughputs.size());
	for (unsigned int i = 0; i < allAps.size(); i++) allAps[i] = i;

	double global = this->ppoObjectiveReward(this->_throughputs, attainables);
	if (this->_rewardType == LOGPF) {
		global -= 0.2 * this->starvedStationRatio(allAps, attainables);
	}
	double local = this->localObjectiveReward(agentIndex, attainables);
	double actionPenalty = 0.0;
	if (agentIndex < this->_previousDeltaActions.size()) {
		actionPenalty = this->_ppoActionPenalty * (std::abs(std::get<0>(this->_previousDeltaActions[agentIndex])) + std::abs(std::get<1>(this->_previousDeltaActions[agentIndex]))) / 2.0;
	}

	return 0.65 * global + 0.35 * local - actionPenalty;
}

bool Simulation::ppoBridgeEnabled() const {
	return std::getenv("NSTEST_PPO_BRIDGE") != nullptr;
}

bool Simulation::ensurePpoBridgeConnected() {
	if (this->_ppoBridgeFd >= 0) return true;

	std::string host = envOrDefault("NSTEST_PPO_HOST", "127.0.0.1");
	std::string port = envOrDefault("NSTEST_PPO_PORT", "9876");
	std::string error;
	if (!BridgeUtils::connectTcp(host, port, this->_ppoBridgeFd, error) && !this->_ppoBridgeWarned) {
		std::cerr << "ns3-ppo: cannot connect to " << host << ":" << port
							<< " (" << error << "); using zero deltas until the bridge is available" << std::endl;
		this->_ppoBridgeWarned = true;
	}
	if (this->_ppoBridgeFd >= 0) this->_ppoBridgeWarned = false;

	return this->_ppoBridgeFd >= 0;
}

void Simulation::closePpoBridge() {
	if (this->_ppoBridgeFd >= 0) {
		close(this->_ppoBridgeFd);
		this->_ppoBridgeFd = -1;
	}
}

std::vector<std::tuple<int, int>> Simulation::queryPpoBridge() {
	std::vector<std::tuple<int, int>> actions;
	for (std::tuple<double, double> couple: this->_configuration) {
		if (this->_ppoAbsoluteActions) {
			actions.push_back(std::make_tuple((int) std::get<0>(couple), (int) std::get<1>(couple)));
		} else {
			actions.push_back(std::make_tuple(0, 0));
		}
	}
	if (this->_configuration.empty() || !this->ensurePpoBridgeConnected()) return actions;

	std::vector<std::vector<double>> attainables = this->attainableThroughputs();
	std::vector<std::vector<double>> observations;
	std::vector<double> rewards;
	for (unsigned int agent = 0; agent < this->_configuration.size(); agent++) {
		observations.push_back(this->buildAgentObservation(agent));
		rewards.push_back(this->cooperativeAgentReward(agent, attainables));
	}

	unsigned int obsDim = observations.empty() ? 0 : observations[0].size();
	std::ostringstream payload;
	payload << "STEP " << this->_testCounter << " " << observations.size() << " " << obsDim << "\n";
	for (unsigned int agent = 0; agent < observations.size(); agent++) {
		payload << agent << " " << rewards[agent];
		for (double value: observations[agent]) payload << " " << value;
		payload << "\n";
	}
	payload << "END\n";

	if (!BridgeUtils::sendAll(this->_ppoBridgeFd, payload.str())) {
		this->closePpoBridge();
		return actions;
	}

	std::string response;
	if (!BridgeUtils::receiveLine(this->_ppoBridgeFd, response)) {
		this->closePpoBridge();
		return actions;
	}

	std::istringstream parser(response);
	std::string token;
	parser >> token;
	if (token != "ACTIONS") {
		std::cerr << "ns3-ppo: invalid bridge response '" << response << "'; keeping current configuration" << std::endl;
		return actions;
	}

	for (unsigned int agent = 0; agent < actions.size(); agent++) {
		int first = 0, second = 0;
		if (!(parser >> first >> second)) break;
		if (this->_ppoAbsoluteActions) {
			actions[agent] = std::make_tuple(
				std::max(-82, std::min(-63, first)),
				std::max(1, std::min(21, second)));
		} else {
			actions[agent] = std::make_tuple(
				std::max(-1, std::min(1, first)),
				std::max(-1, std::min(1, second)));
		}
	}

	return actions;
}

std::vector<double> Simulation::buildAgentObservation(unsigned int agentIndex) {
	std::vector<double> obs;
	std::vector<unsigned int> members = this->agentMembers(agentIndex);
	std::vector<std::vector<double>> attainables = this->attainableThroughputs();
	NetworkConfiguration unclusterized = this->handleClusterizedConfiguration(this->_configuration);

	std::tuple<double, double> agentConf = agentIndex < this->_configuration.size() ? this->_configuration[agentIndex] : std::make_tuple((double) this->_defaultSensibility, (double) this->_defaultPower);
	double sens = std::get<0>(agentConf),
				 power = std::get<1>(agentConf);

	double totalThroughput = 0.0, totalAttainable = 0.0, minSat = 1.0, satSum = 0.0, perSum = 0.0, perMax = 0.0, demandSum = 0.0;
	unsigned int activeStations = 0, totalStations = 0;
	for (unsigned int ap: members) {
		if (ap >= this->_throughputs.size() || ap >= attainables.size()) continue;
		for (unsigned int sta = 0; sta < this->_throughputs[ap].size(); sta++) {
			unsigned int staId = this->_associations[ap][sta];
			totalStations++;
			if (staId < this->_currentStationThroughputs.size()) {
				demandSum += demandLevel(this->_currentStationThroughputs[staId]);
			}
			if (attainables[ap][sta] <= 0.0) continue;
			double sat = clampUnit(this->_throughputs[ap][sta] / attainables[ap][sta]);
			totalThroughput += this->_throughputs[ap][sta];
			totalAttainable += attainables[ap][sta];
			minSat = std::min(minSat, sat);
			satSum += sat;
			perSum += this->_pers[ap][sta];
			perMax = std::max(perMax, this->_pers[ap][sta]);
			activeStations++;
		}
	}

	unsigned int maxStaPerAp = 1;
	for (const std::vector<unsigned int>& assoc: this->_associations) {
		maxStaPerAp = std::max<unsigned int>(maxStaPerAp, assoc.size());
	}

	double meanSat = activeStations > 0 ? satSum / activeStations : 0.0;
	double avgPer = activeStations > 0 ? perSum / activeStations : 0.0;
	double demandMean = totalStations > 0 ? demandSum / totalStations : 0.0;
	double load = clampUnit(((double) totalStations) / (maxStaPerAp * std::max<unsigned int>(1, members.size())));
	double starvedRatio = this->starvedStationRatio(members, attainables);
	double apSatisfaction = totalAttainable > 0.0 ? clampUnit(totalThroughput / totalAttainable) : 0.0;

	std::vector<std::vector<unsigned int>> conflicts = this->extractConflicts(unclusterized);
	std::set<unsigned int> neighborAps;
	for (unsigned int ap: members) {
		if (ap >= conflicts.size()) continue;
		for (unsigned int neighbor: conflicts[ap]) {
			if (std::find(members.begin(), members.end(), neighbor) == members.end()) {
				neighborAps.insert(neighbor);
			}
		}
	}

	std::vector<double> meanNeighbor(4, 0.0), maxNeighbor(4, 0.0);
	double strongestRssi = 0.0;
	for (unsigned int neighbor: neighborAps) {
		unsigned int neighborAgent = neighbor < this->_clustersAP.size() ? this->_clustersAP[neighbor] : neighbor;
		std::tuple<double, double> neighborConf = neighborAgent < this->_configuration.size() ? this->_configuration[neighborAgent] : unclusterized[neighbor];
		double neighborThroughput = 0.0, neighborAttainable = 0.0;
		if (neighbor < this->_throughputs.size() && neighbor < attainables.size()) {
			for (unsigned int sta = 0; sta < this->_throughputs[neighbor].size(); sta++) {
				neighborThroughput += this->_throughputs[neighbor][sta];
				neighborAttainable += attainables[neighbor][sta];
			}
		}
		double neighborSat = neighborAttainable > 0.0 ? clampUnit(neighborThroughput / neighborAttainable) : 0.0;
		double rssi = -200.0;
		for (unsigned int ap: members) {
			rssi = std::max(rssi, Simulation::pathLoss(
				std::make_tuple(this->_positionAPX[neighbor], this->_positionAPY[neighbor], this->_positionAPZ[neighbor]),
				std::make_tuple(this->_positionAPX[ap], this->_positionAPY[ap], this->_positionAPZ[ap]),
				std::get<1>(neighborConf)));
		}
		std::vector<double> neighborFeatures({
			normalizedSensitivity(std::get<0>(neighborConf)),
			normalizedPower(std::get<1>(neighborConf)),
			neighborSat,
			normalizedRssi(rssi)
		});
		for (unsigned int i = 0; i < neighborFeatures.size(); i++) {
			meanNeighbor[i] += neighborFeatures[i];
			maxNeighbor[i] = std::max(maxNeighbor[i], neighborFeatures[i]);
		}
		strongestRssi = std::max(strongestRssi, normalizedRssi(rssi));
	}
	if (!neighborAps.empty()) {
		for (double& value: meanNeighbor) value /= neighborAps.size();
	}

	obs.push_back(normalizedSensitivity(sens));
	obs.push_back(normalizedPower(power));
	obs.push_back(clampUnit((-62.0 - (sens + power)) / 20.0));
	obs.push_back(apSatisfaction);
	obs.push_back(activeStations > 0 ? minSat : 0.0);
	obs.push_back(meanSat);
	obs.push_back(starvedRatio);
	obs.push_back(clampUnit(avgPer));
	obs.push_back(clampUnit(perMax));
	obs.push_back(load);
	obs.push_back(clampUnit(demandMean));
	obs.push_back(this->_positionAPX.size() > 1 ? clampUnit(((double) neighborAps.size()) / (this->_positionAPX.size() - 1)) : 0.0);
	obs.insert(obs.end(), meanNeighbor.begin(), meanNeighbor.end());
	obs.insert(obs.end(), maxNeighbor.begin(), maxNeighbor.end());
	obs.push_back(strongestRssi);
	obs.push_back(!this->_rewards.empty() ? clampUnit(this->_rewards.back()) : 0.0);
	obs.push_back(this->_ema >= 0.0 ? clampUnit(this->_ema) : 0.0);
	obs.push_back(clampUnit(this->localObjectiveReward(agentIndex, attainables)));
	if (agentIndex < this->_previousDeltaActions.size()) {
		obs.push_back(std::get<0>(this->_previousDeltaActions[agentIndex]));
		obs.push_back(std::get<1>(this->_previousDeltaActions[agentIndex]));
	} else {
		obs.push_back(0.0);
		obs.push_back(0.0);
	}
	obs.push_back(this->_dynamicScenario ? 1.0 : 0.0);
	obs.push_back(this->_dynamicDemandPhases.size() > 1 ? ((double) this->_dynamicPhase) / (this->_dynamicDemandPhases.size() - 1) : 0.0);
	if (this->_phaseAwareObservations) {
		unsigned int phaseCount = std::max<unsigned int>(1, this->_dynamicDemandPhases.size());
		unsigned int phase = std::min<unsigned int>(this->_dynamicPhase, phaseCount - 1);
		for (unsigned int i = 0; i < 4; i++) {
			obs.push_back(i == phase ? 1.0 : 0.0);
		}
		double elapsed = this->_duration > 0.0 ? clampUnit((this->_testCounter * this->_testDuration) / this->_duration) : 0.0;
		double phaseStart = ((double) phase) / phaseCount;
		double phaseProgress = clampUnit((elapsed - phaseStart) * phaseCount);
		obs.push_back(phaseProgress);
		obs.push_back(elapsed);
	}

	return obs;
}

std::string Simulation::agentObservationsToString() {
	std::vector<std::vector<double>> attainables = this->attainableThroughputs();
	std::ostringstream stream;
	for (unsigned int agent = 0; agent < this->_configuration.size(); agent++) {
		if (agent > 0) stream << "|";
		std::vector<double> obs = this->buildAgentObservation(agent);
		stream << agent << ":" << this->cooperativeAgentReward(agent, attainables);
		for (double value: obs) {
			stream << "," << value;
		}
	}

	return stream.str();
}

std::vector<std::vector<double>> Simulation::attainableThroughputs() const {
	std::vector<std::vector<double>> attainables;
	std::vector<double> references = Simulation::attainableThroughputsFromChannel(this->_channel_width);
	unsigned int i = 0;
	for (NodeContainer::Iterator iap = this->_nodesAP.Begin(); iap != this->_nodesAP.End(); iap++, i++) {
    Node* ap = GetPointer(*iap);
		std::vector<double> attainablesSta;
		unsigned int j = 0;
		for (NodeContainer::Iterator istation = this->_nodesSta[i].Begin(); istation != this->_nodesSta[i].End(); istation++, j++) {
			unsigned int sta_idx = this->_associations[i][j];
			Node* sta = GetPointer(*istation);
			unsigned int mcsValue = Simulation::getMCSValue(ap, sta);
			if (sta_idx < this->_currentStationThroughputs.size() && this->_currentStationThroughputs[sta_idx] == NONE) {
				attainablesSta.push_back(0.0);
				continue;
			}
			attainablesSta.push_back(std::max(1.0, std::min(this->_packetSize / this->_intervals[sta_idx], references[mcsValue])));
		}

		attainables.push_back(attainablesSta);
  }

	return attainables;
}

std::vector<double> Simulation::attainableThroughputsFromChannel(ChannelWidth cw) {
	std::vector<double> at;
	switch (cw) {
		case MHZ_20: return {30e6, 55e6, 85e6, 106e6, 165e6, 215e6, 240e6, 265e6, 315e6, 345e6, 390e6, 430e6};
		case MHZ_40: return {55e6, 110e6, 165e6, 215e6, 320e6, 400e6, 455e6, 500e6, 580e6, 645e6, 700e6, 785e6};
		case MHZ_80: return {120e6, 230e6, 330e6, 430e6, 610e6, 780e6, 850e6, 890e6, 1070e6, 1180e6, 1220e6, 1400e6};
	}

	return {};
}

double Simulation::adHocReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables) const {
	double starvRew = 1, noStarvRew = 1, nStarv = 0, nNoStarv = 0;
	for (unsigned int i = 0; i < throughputs.size(); i++) {
		double n = throughputs[i].size();
		for (unsigned int j = 0; j < n; j++) {
			if (attainables[i][j] <= 0.0) continue;
      double threshold = 0.1 * attainables[i][j] / n;
      if (throughputs[i][j] < threshold) {
        throughputs[i][j] = std::max(throughputs[i][j], 1.0);
        starvRew *= throughputs[i][j] / threshold;
				nStarv++;
      } else {
				noStarvRew *= std::min(1.0, throughputs[i][j] / attainables[i][j]);
				nNoStarv++;
			}
    }
	}

  double n = nStarv + nNoStarv;
	if (n == 0) return 0.0;
  // Compute global reward
  return (nStarv * starvRew + nNoStarv * (noStarvRew + n)) / (n * (n + 1.0));
}

double Simulation::adHocTieBreakReward(
		const std::vector<std::vector<double>>& throughputs,
		const std::vector<std::vector<double>>& attainables) const {
	double closestStarved = 0.0;
	double nonStarvedLogSum = 0.0;
	unsigned int nStarved = 0, nNonStarved = 0;
	for (unsigned int i = 0; i < throughputs.size() && i < attainables.size(); i++) {
		double n = throughputs[i].size();
		if (n <= 0.0) continue;
		unsigned int count = std::min(throughputs[i].size(), attainables[i].size());
		for (unsigned int j = 0; j < count; j++) {
			if (attainables[i][j] <= 0.0) continue;
			double threshold = 0.1 * attainables[i][j] / n;
			if (throughputs[i][j] < threshold) {
				// The closest starved STA is the cheapest route to the next primary
				// ADHOC tier; unlike the product in ADHOC this does not underflow.
				closestStarved = std::max(
					closestStarved, clampUnit(throughputs[i][j] / threshold));
				nStarved++;
			} else {
				double normalized = clampUnit(throughputs[i][j] / attainables[i][j]);
				nonStarvedLogSum += std::log(std::max(normalized, 1.0e-12));
				nNonStarved++;
			}
		}
	}

	if (nStarved > 0) return closestStarved;
	if (nNonStarved > 0) return std::exp(nonStarvedLogSum / nNonStarved);
	return 0.0;
}

std::vector<std::tuple<double, unsigned int>> Simulation::subrewardFromThroughputs() {
	std::vector<std::vector<double>> attainables = this->attainableThroughputs();
	std::vector<std::tuple<double, unsigned int>> subrewards;

	for (unsigned int i = 0; i < this->_throughputs.size(); i++) {
		std::tuple<double, unsigned int> t;
		switch (this->_rewardType) {
			case AD_HOC: t = std::make_tuple(this->adHocReward({this->_throughputs[i]}, {attainables[i]}), this->_throughputs[i].size()); break;
			case CUMTP: t = std::make_tuple(this->cumulatedThroughputReward({this->_throughputs[i]}, {attainables[i]}), this->_throughputs[i].size()); break;
			case LOGPF: t = std::make_tuple(this->logPfReward({this->_throughputs[i]}, {attainables[i]}), this->_throughputs[i].size()); break;
		}
		subrewards.push_back(t);
	}

	return subrewards;
}

std::vector<std::tuple<double, unsigned int>> Simulation::inspireSelfishRewards() const {
	unsigned int numberOfAgents = this->_configuration.size();
	std::vector<double> rewards(numberOfAgents, 0.0);
	std::vector<unsigned int> stationCounts(numberOfAgents, 0);
	std::vector<std::vector<double>> attainables = this->attainableThroughputs();
	for (unsigned int ap = 0; ap < this->_throughputs.size() && ap < this->_clustersAP.size(); ap++) {
		unsigned int agent = this->_clustersAP[ap];
		if (agent >= numberOfAgents) continue;
		std::vector<double> activeThroughputs;
		std::vector<double> activeAttainables;
		for (unsigned int sta = 0; sta < this->_throughputs[ap].size(); sta++) {
			if (sta < this->_associations[ap].size()) {
				unsigned int globalSta = this->_associations[ap][sta];
				if (globalSta < this->_currentStationThroughputs.size() && this->_currentStationThroughputs[globalSta] == NONE) continue;
			}
			activeThroughputs.push_back(this->_throughputs[ap][sta]);
			if (ap < attainables.size() && sta < attainables[ap].size()) {
				activeAttainables.push_back(attainables[ap][sta]);
			}
		}
		if (this->_rewardType == AD_HOC) {
			// Controlled ADHOC comparison: make the BO observation use the
			// same starvation-aware utility as the global reward.
			if (!activeThroughputs.empty() && activeThroughputs.size() == activeAttainables.size()) {
				rewards[agent] += this->adHocReward({activeThroughputs}, {activeAttainables});
				stationCounts[agent] += activeThroughputs.size();
			}
		} else {
			// Paper Equation (1): local log-throughput reward. One bit/s is
			// a numerical floor for zero-packet intervals.
			for (double throughput: activeThroughputs) {
				rewards[agent] += std::log(std::max(1.0, throughput));
			}
			stationCounts[agent] += activeThroughputs.size();
		}
	}

	std::vector<std::tuple<double, unsigned int>> result;
	for (unsigned int agent = 0; agent < numberOfAgents; agent++) {
		result.push_back(std::make_tuple(rewards[agent], stationCounts[agent]));
	}
	return result;
}

std::vector<std::vector<unsigned int>> Simulation::inspireNeighborhoods() const {
	unsigned int numberOfAgents = this->_configuration.size();
	std::vector<std::set<unsigned int>> neighborhoodSets(numberOfAgents);
	for (unsigned int agent = 0; agent < numberOfAgents; agent++) neighborhoodSets[agent].insert(agent);

	// The paper defines N_i using communication range under the IEEE 802.11
	// default configuration, independently of configurations explored later.
	NetworkConfiguration defaults(this->_positionAPX.size(), std::make_tuple(-82.0, 20.0));
	std::vector<std::vector<unsigned int>> conflicts = this->extractConflicts(defaults);
	for (unsigned int ap = 0; ap < conflicts.size() && ap < this->_clustersAP.size(); ap++) {
		unsigned int agent = this->_clustersAP[ap];
		if (agent >= numberOfAgents) continue;
		for (unsigned int otherAp: conflicts[ap]) {
			if (otherAp >= this->_clustersAP.size()) continue;
			unsigned int otherAgent = this->_clustersAP[otherAp];
			if (otherAgent >= numberOfAgents) continue;
			// Control-frame communication is treated as an undirected relation,
			// matching i in N_j iff j in N_i used by Equations (3)-(4).
			neighborhoodSets[agent].insert(otherAgent);
			neighborhoodSets[otherAgent].insert(agent);
		}
	}

	std::vector<std::vector<unsigned int>> neighborhoods(numberOfAgents);
	for (unsigned int agent = 0; agent < numberOfAgents; agent++) {
		neighborhoods[agent].assign(neighborhoodSets[agent].begin(), neighborhoodSets[agent].end());
	}
	return neighborhoods;
}

double Simulation::logPfReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables) {
	double logT = 0.0, logTA = 0.0;
  for (unsigned int i = 0; i < throughputs.size(); i++) {
		double n = throughputs[i].size();
		for (unsigned int j = 0; j < n; j++) {
			if (attainables[i][j] <= 0.0) continue;
			logT += log(std::max(1.0, std::min(throughputs[i][j], attainables[i][j])));
			logTA += log(attainables[i][j]);
    }
	}
  // Compute global reward
  return logTA > 0.0 ? logT / logTA : 0.0;
}

/**
 * Compute the fairness of the network (Jain's index).
 *
 * @return the fairness
 */
double Simulation::fairnessFromThroughputs() {
  double squareOfMean = 0, meanOfSquares = 0;
	unsigned int n = 0;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++)
    for (unsigned int j = 0; j < this->_throughputs[i].size(); j++) {
			unsigned int staId = this->_associations[i][j];
			if (staId < this->_currentStationThroughputs.size() && this->_currentStationThroughputs[staId] == NONE) continue;
			double tMbps = this->_throughputs[i][j] / 1.0e6;
      squareOfMean += tMbps;
			meanOfSquares += tMbps * tMbps;
			n++;
    }
	if (n == 0 || meanOfSquares == 0.0) return 0.0;

	squareOfMean *= squareOfMean / (n * n);
	meanOfSquares /= n;

	return squareOfMean / meanOfSquares;
}

/**
 * Compute the cumulated throughput.
 *
 * @return the cumulated throughput
 */
double Simulation::cumulatedThroughputReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables) {
  double cumThroughput = 0;
  for (unsigned int i = 0; i < throughputs.size(); i++)
    for (unsigned int j = 0; j < throughputs[i].size(); j++)
			if (attainables[i][j] > 0.0)
				cumThroughput += throughputs[i][j];

	return cumThroughput;
}

/**
 * Compute the cumulated throughput.
 *
 * @return the cumulated throughput
 */
double Simulation::cumulatedThroughputFromThroughputs() {
  double cumThroughput = 0;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++)
    for (unsigned int j = 0; j < this->_throughputs[i].size(); j++)
			cumThroughput += this->_throughputs[i][j];

	return cumThroughput;
}

/**
 * Compute the throughput of each AP.
 *
 * @return the throughput of each AP
 */
std::vector<double> Simulation::apThroughputsFromThroughputs() {
  std::vector<double> apThroughputs;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++) {
		double apThroughput = 0;
		for (unsigned int j = 0; j < this->_throughputs[i].size(); j++)
			apThroughput += this->_throughputs[i][j];
		apThroughputs.push_back(apThroughput);
	}

	return apThroughputs;
}

/**
 * Compute the throughput of each STA.
 *
 * @return the throughput of each STA
 */
std::vector<double> Simulation::staThroughputsFromThroughputs() {
  std::vector<double> staThroughputs;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++)
		for (unsigned int j = 0; j < this->_throughputs[i].size(); j++)
			staThroughputs.push_back(this->_throughputs[i][j]);

	return staThroughputs;
}

/**
 * Compute the PER of each STA.
 *
 * @return the PER of each STA
 */
std::vector<double> Simulation::staPersFromPers() {
  std::vector<double> staPers;
  for (unsigned int i = 0; i < this->_pers.size(); i++)
		for (unsigned int j = 0; j < this->_pers[i].size(); j++)
			staPers.push_back(this->_pers[i][j]);

	return staPers;
}

/**
 * Compute the throughput of each station
 */
void Simulation::computeThroughputsAndErrors() {
  // Compute throughput for each server app
  for (unsigned int i = 0; i < this->_serversPerAp.size(); i++) {
    std::vector<double> staAPThroughputs(this->_lastRxPackets[i].size());
		std::vector<double> staPERs(this->_lastLostPackets[i].size());
    for (unsigned int j = 0; j < this->_serversPerAp[i].GetN(); j++) {
			if (this->_tcpTransport) {
				PacketSink* sink = dynamic_cast<PacketSink*>(GetPointer(this->_serversPerAp[i].Get(j)));
				uint64_t receivedBytes = sink->GetTotalRx();
				uint64_t testReceivedBytes = receivedBytes - this->_lastRxBytes[i][j];
				staAPThroughputs[j] = 8.0 * testReceivedBytes / this->_testDuration;
				// UDP packet loss is not defined for TCP's reliable byte stream.
				staPERs[j] = -1.0;
				this->_lastRxBytes[i][j] = receivedBytes;
				continue;
			}
      // Received bytes since the start of the simulation
			UdpServer* server = dynamic_cast<UdpServer*>(GetPointer(this->_serversPerAp[i].Get(j)));
			unsigned int lostPackets = server->GetLost();
			unsigned int testLostPackets = lostPackets - this->_lastLostPackets[i][j];
			unsigned int receivedPackets = server->GetReceived();
			unsigned int testReceivedPackets = receivedPackets - this->_lastRxPackets[i][j];
			double per = testLostPackets + testReceivedPackets != 0 ? ((double) testLostPackets) / (testLostPackets + testReceivedPackets) : 0;

			// std::cout << testLostPackets << " and " << testReceivedPackets << " => " << per << std::endl;

      double rxBits = this->_packetSize * testReceivedPackets;
      // Compute the throughput considering only unseen bytes
      double throughput = rxBits / this->_testDuration; // bit/s
			// std::cout << i << " " << j << ": " << throughput << std::endl;
      // Update containers
      staAPThroughputs[j] = throughput;
			staPERs[j] = per;
      this->_lastRxPackets[i][j] = receivedPackets;
			this->_lastLostPackets[i][j] = lostPackets;
      // Log
      // std::cout << "Station " << j << " of AP " << i << " : " << (throughput / 1e6) << " Mbit/s" << std::endl;
    }
    this->_throughputs[i] = staAPThroughputs;
		this->_pers[i] = staPERs;
  }
}

/**
 * Set up a new configuration in the simulation
 *
 * @param configuration the configuration to set
 */
NetworkConfiguration Simulation::projectConfiguration(NetworkConfiguration configuration) const {
	for (std::tuple<double, double>& couple: configuration) {
		double sensibility = std::max(-82.0, std::min(-62.0, std::get<0>(couple)));
		double txPower = std::max(1.0, std::min(21.0, std::get<1>(couple)));
		if (sensibility + txPower > -62.0) {
			txPower = -62.0 - sensibility;
			if (txPower < 1.0) {
				txPower = 1.0;
				sensibility = -63.0;
			}
		}
		std::get<0>(couple) = std::max(-82.0, std::min(-62.0, sensibility));
		std::get<1>(couple) = std::max(1.0, std::min(21.0, txPower));
	}

	return configuration;
}

NetworkConfiguration Simulation::applyDeltaActions(const NetworkConfiguration& base, const std::vector<std::tuple<int, int>>& deltas) const {
	NetworkConfiguration next = base;
	for (unsigned int i = 0; i < next.size() && i < deltas.size(); i++) {
		std::get<0>(next[i]) += std::max(-1, std::min(1, std::get<0>(deltas[i])));
		std::get<1>(next[i]) += std::max(-1, std::min(1, std::get<1>(deltas[i])));
	}

	return this->projectConfiguration(next);
}

NetworkConfiguration Simulation::applyPpoActions(const NetworkConfiguration& base, const std::vector<std::tuple<int, int>>& actions) const {
	if (!this->_ppoAbsoluteActions) {
		return this->applyDeltaActions(base, actions);
	}

	NetworkConfiguration next = base;
	for (unsigned int i = 0; i < next.size() && i < actions.size(); i++) {
		std::get<0>(next[i]) = std::get<0>(actions[i]);
		std::get<1>(next[i]) = std::get<1>(actions[i]);
	}

	return this->projectConfiguration(next);
}

void Simulation::setupNewConfiguration(NetworkConfiguration configuration) {
	configuration = this->projectConfiguration(configuration);
	if (this->_previousDeltaActions.size() != configuration.size()) {
		this->_previousDeltaActions = std::vector<std::tuple<int, int>>(configuration.size(), std::make_tuple(0, 0));
	}
	for (unsigned int i = 0; i < configuration.size() && i < this->_configuration.size(); i++) {
		this->_previousDeltaActions[i] = std::make_tuple(
			deltaSign(std::get<0>(configuration[i]) - std::get<0>(this->_configuration[i])),
			deltaSign(std::get<1>(configuration[i]) - std::get<1>(this->_configuration[i])));
	}

  int nNodes = this->_devices.size();
	NetworkConfiguration unclusterized = this->handleClusterizedConfiguration(configuration);
	for (int i = 0; i < nNodes; i++) {
    Ptr<WifiPhy> phy = dynamic_cast<WifiNetDevice*>(GetPointer((this->_devices[i].Get(0))))->GetPhy();

    double sensibility = std::get<0>(unclusterized[i]),
           txPower = std::get<1>(unclusterized[i]);
    phy->SetTxPowerEnd(txPower);
    phy->SetTxPowerStart(txPower);
    phy->SetRxSensitivity(sensibility);
    phy->SetCcaEdThreshold(sensibility);
  }

	this->_configuration = configuration;
}

/**
 * Extract topology information from its JSON representation
 *
 * @param topo JSON representation of the topology
 */
void Simulation::readTopology(Json::Value topo) {
  // APs
	unsigned int c = 0;
	bool clusterized = false;
	if (topo["aps"][0].isMember("cluster")) clusterized = true;

  for (Json::Value ap: topo["aps"]) {
    this->_positionAPX.push_back(ap["x"].asDouble());
    this->_positionAPY.push_back(ap["y"].asDouble());
    this->_positionAPZ.push_back(ap["z"].asDouble());

		if (clusterized) {
			this->_clustersAP.push_back(ap["cluster"].asUInt64());
		} else {
			this->_clustersAP.push_back(c);
			c++;
		}

    std::vector<unsigned int> assoc;
    for (Json::Value sta: ap["stas"]) {
      assoc.push_back(sta.asUInt());
    }
    this->_associations.push_back(assoc);
  }

  // Stations
  for (Json::Value sta: topo["stations"]) {
    this->_positionStaX.push_back(sta["x"].asDouble());
    this->_positionStaY.push_back(sta["y"].asDouble());
    this->_positionStaZ.push_back(sta["z"].asDouble());
  }
}

/**
 * Boolean constraint for parameter combination
 *
 * @param sens double the sensibility
 * @param pow double the transmission power
 *
 * @return true if the constraint is validated, false otherwise
 */
bool Simulation::parameterConstraint(double sens, double pow) {
  return sens <= std::max(-82.0, std::min(-62.0, -82.0 + 20.0 - pow));
}

/**
 * Compute the number of samples for a given gaussian mixture
 *
 * @param gaussians Container the gaussian mixture
 *
 * @return the number of tests to do before updating the mixture
 */
unsigned int Simulation::numberOfSamples(std::vector<GaussianT> gaussians) {
  double s = 0;
  for (GaussianT g: gaussians) {
		s += 0.5 * 2.0 * std::get<0>(g).size() * std::get<1>(g) / 0.05;
	}

  return round(s);
}

double Simulation::stationThroughputToInterval(StationThroughput stt, double duration) const {
	switch (stt) {
		case NONE: return duration;
		case LOW: return this->_packetSize / 500e3;
		case MEDIUM: return this->_packetSize / 5e6;
		case HIGH: return this->_packetSize / 50e6;
	}

	return duration;
}

void Simulation::stationsThroughputsToInterval(const std::vector<StationThroughput>& stations_throughputs, double duration) {
	this->_intervals.clear();
	for (StationThroughput stt: stations_throughputs) {
		this->_intervals.push_back(this->stationThroughputToInterval(stt, duration));
	}
}

void Simulation::applyDemandPhase(unsigned int phase) {
	if (this->_dynamicDemandPhases.empty()) return;
	this->_dynamicPhase = std::min<unsigned int>(phase, this->_dynamicDemandPhases.size() - 1);
	this->_currentStationThroughputs = this->_dynamicDemandPhases[this->_dynamicPhase];
	this->stationsThroughputsToInterval(this->_currentStationThroughputs, std::max(1.0, this->_testDuration));
}

std::vector<std::vector<StationThroughput>> Simulation::buildDynamicDemandPhases(const std::vector<StationThroughput>& base) const {
	if (!this->_dynamicScenario || base.empty()) return {base};

	std::string profile = lowerString(envOrDefault("NSTEST_DYNAMIC_PROFILE", "legacy"));
	if (profile == "contrast" || profile == "contrasted") {
		std::vector<StationThroughput> allHigh(base.size(), HIGH);
		std::vector<StationThroughput> hotspotA(base.size(), NONE);
		std::vector<StationThroughput> allLow(base.size(), LOW);
		std::vector<StationThroughput> hotspotB(base.size(), NONE);

		unsigned int split = (this->_associations.size() + 1) / 2;
		for (unsigned int ap = 0; ap < this->_associations.size(); ap++) {
			for (unsigned int staId: this->_associations[ap]) {
				if (staId >= base.size()) continue;
				if (ap < split) {
					hotspotA[staId] = HIGH;
				} else {
					hotspotB[staId] = HIGH;
				}
			}
		}

		return {allHigh, hotspotA, allLow, hotspotB};
	}

	std::vector<StationThroughput> normal = base;
	std::vector<StationThroughput> flash(base.size(), LOW);
	std::vector<StationThroughput> peak(base.size(), HIGH);
	std::vector<StationThroughput> recovery(base.size(), MEDIUM);

	for (unsigned int ap = 0; ap < this->_associations.size(); ap++) {
		for (unsigned int staId: this->_associations[ap]) {
			flash[staId] = ap % 2 == 0 ? HIGH : LOW;
		}
	}

	for (unsigned int i = 0; i < recovery.size(); i++) {
		if (i % 5 == 0) recovery[i] = NONE;
		else if (i % 2 == 0) recovery[i] = LOW;
	}

	return {normal, flash, peak, recovery};
}

std::string Simulation::stateVectorToString(unsigned int index) const {
	double totalDisplacement = 0.0;
	unsigned int stationCount = 0;
	if (this->_mobility) {
		for (unsigned int ap = 0; ap < this->_nodesSta.size(); ap++) {
			for (unsigned int sta = 0; sta < this->_nodesSta[ap].GetN(); sta++) {
				unsigned int stationId = this->_associations[ap][sta];
				Vector current = this->_nodesSta[ap].Get(sta)->GetObject<MobilityModel>()->GetPosition();
				double dx = current.x - this->_positionStaX[stationId];
				double dy = current.y - this->_positionStaY[stationId];
				double dz = current.z - this->_positionStaZ[stationId];
				totalDisplacement += std::sqrt(dx * dx + dy * dy + dz * dz);
				stationCount++;
			}
		}
	}
	std::ostringstream state;
	state << "dynamic=" << (this->_dynamicScenario ? 1 : 0)
				<< ";mobility=" << (this->_mobility ? 1 : 0)
				<< ";mean_displacement_m=" << (stationCount ? totalDisplacement / stationCount : 0.0)
				<< ";transport=" << (this->_tcpTransport ? "tcp" : "udp")
				<< ";rate_manager=" << lowerString(envOrDefault("NSTEST_RATE_MANAGER", "minstrel"))
				<< ";phase=" << this->_dynamicPhase
				<< ";demand=" << joinDemands(this->_currentStationThroughputs)
				<< ";reward=" << this->_rewards[index]
				<< ";fair=" << this->_fairness[index]
				<< ";cum=" << this->_cumulatedThroughput[index]
				<< ";aps=" << joinDoubles(this->_apThroughputs[index])
				<< ";stas=" << joinDoubles(this->_staThroughputs[index])
				<< ";pers=" << joinDoubles(this->_staPERs[index])
				<< ";conf=" << Simulation::configurationToString(this->_configurations[index]);

	return state.str();
}

/**
 * Compute the number of samples for a given gaussian mixture
 *
 * @param gaussians Container the gaussian mixture
 *
 * @return the number of tests to do before updating the mixture
 */
unsigned int Simulation::numberOfSamples_HCM(std::vector<Ring> circulars) {
  return 2 * circulars.size();
}

/**
 * Map channel index to a real channel.
 *
 * Channel number must be in {36, 40, etc.} for the 5GHz band.
 * The web page: https://www.nsnam.org/docs/models/html/wifi-user.html
 * Read in particular the WifiPhy::ChannelNumber section
 *
 * @param i int the index to map
 *
 * @return a channel corresponding to the mapped index
 */
int Simulation::channelNumber(ChannelWidth cw) {
  switch (cw) {
    case MHZ_20: return 36;
    case MHZ_40: return 38;
    case MHZ_80: return 42;
    default:
      std::cerr << "Error channelNumber(): the index is negative, null, or greater than the number of channels (12 - 40MHz)." << std::endl;
      return 42;
  }
}

/**
 * Turn a network configuration to a convenient string representation
 *
 * @param config Container the configuration
 *
 * @return a convenient representation of the configuration
 */
std::string Simulation::configurationToString(const NetworkConfiguration& config) {
	std::string result = "";
	for (unsigned int i = 0; i < config.size(); i++) {
		result += "(" + std::to_string(std::get<0>(config[i])) + "," + std::to_string(std::get<1>(config[i])) + ")";
		if (i < config.size() - 1)
			result += ",";
	}

	return result;
}

std::vector<NetworkConfiguration> Simulation::findDegreeEntryPoints(double criterion) const {
	int tx = 20;
	double avg_deg = 1000;
	unsigned int n = this->_positionAPX.size();
	NetworkConfiguration conf;
	do {
		conf = NetworkConfiguration(n, std::make_tuple(-62 - tx, tx));
		std::vector<std::vector<unsigned int>> conflicts = this->extractConflicts(conf);
		unsigned int sum = 0;
		for (std::vector<unsigned int> c: conflicts) {
			sum += c.size();
		}
		avg_deg = ((double) sum) / n;
		tx--;
	} while (avg_deg > criterion);

	return {conf};
}

std::vector<NetworkConfiguration> Simulation::findNHDegreeEntryPoints(double criterion) const {
	double avg_deg = 1000;
	unsigned int n = this->_positionAPX.size();
	NetworkConfiguration conf(n, std::make_tuple(-82, 20));
	do {
		std::vector<std::vector<unsigned int>> conflicts = this->extractConflicts(conf);
		unsigned int max_idx = 0;
		unsigned int sum = conflicts[0].size();
		for (unsigned int i = 1; i < conflicts.size(); i++) {
			unsigned int nconflicts = conflicts[i].size();
			if (nconflicts > conflicts[max_idx].size())
				max_idx = i;
			sum += nconflicts;
		}

		avg_deg = ((double) sum) / n;

		if (avg_deg < criterion)
			break;

		std::get<1>(conf[max_idx]) = std::get<1>(conf[max_idx]) - 1;
		std::get<0>(conf[max_idx]) = -62 - std::get<1>(conf[max_idx]);
	} while (avg_deg > criterion);

	return {conf};
}

std::vector<NetworkConfiguration> Simulation::findDiagonalEntryPoints(unsigned int n) const {
	unsigned int conf_size = this->_positionAPX.size();
	std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
	std::vector<NetworkConfiguration> confs;
	for (unsigned int i = 0; i < n; i++) {
		NetworkConfiguration conf;
		for (unsigned int j = 0; j < conf_size; j++) {
			double max_dist = 0;
			for (unsigned int sId: this->_associations[j]) {
				double d = Simulation::distance(std::make_tuple(this->_positionAPX[j], this->_positionAPY[j], this->_positionAPZ[j]), std::make_tuple(this->_positionStaX[sId], this->_positionStaY[sId], this->_positionStaZ[sId]));
				if (d > max_dist)
					max_dist = d;
			}
			unsigned int minTx = std::min(std::max(1.0, ceil(-82.0 + 46.67 + 10 * 3 * log10(max_dist))), 21.0);
			std::uniform_int_distribution<int> dist(minTx, 21);
			int tx = dist(generator);
			conf.push_back(std::make_tuple(-62 - tx, tx));
		}
		confs.push_back(conf);
	}

	return confs;
}

std::vector<NetworkConfiguration> Simulation::findEntryPoints(int v) const {
	unsigned int n = this->_positionAPX.size();
	std::vector<unsigned int> order(n, 0);
	for (unsigned int i = 0; i < n; i++) order[i] = i;
	NetworkConfiguration def_conf(n, std::make_tuple(-82, 20)),
											 conf = def_conf,
											 prev_conf = conf;
	std::vector<std::vector<unsigned int>> conflicts = extractConflicts(conf);
	std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
	do {
		prev_conf = conf;
		std::shuffle(order.begin(), order.end(), generator);

		for (unsigned int k: order) {
			std::vector<double> rxs;
			for (unsigned int l: conflicts[k]) {
				rxs.push_back(Simulation::pathLoss(std::make_tuple(this->_positionAPX[k], this->_positionAPY[k], this->_positionAPZ[k]), std::make_tuple(this->_positionAPX[l], this->_positionAPY[l], this->_positionAPZ[l]), std::get<1>(conf[l])));
			}

			if (!rxs.empty()) {
				std::vector<double>::iterator elem;
				if (v == -1 || v >= (int) rxs.size()) {
					elem = std::min_element(rxs.begin(), rxs.end());
				} else {
					std::nth_element(rxs.begin(), rxs.begin() + v - 1, rxs.end(), std::greater<double>());
					elem = rxs.begin() + v - 1;
				}
				double new_sens = std::max(std::min(ceil(*elem - 1), -62.0), -82.0);
				double new_tx = -62 - new_sens;
				conf[k] = std::make_tuple(new_sens, new_tx);
			}
		}
		// for (std::tuple<double, double> t: conf) {
		// 	std::cout << "(" << std::get<0>(t) << "," << std::get<1>(t) << ")";
		// }
		// std::cout << std::endl;
	} while (conf != prev_conf);

	return {conf};
}

std::vector<std::vector<unsigned int>> Simulation::extractConflicts(NetworkConfiguration conf) const {
	std::vector<std::vector<unsigned int>> conflicts;
	for (unsigned int i = 0; i < this->_positionAPX.size(); i++) {
		std::vector<unsigned int> conflictsi;
		for (unsigned int j = 0; j < this->_positionAPX.size(); j++) {
			if (i != j) {
				double rx = Simulation::pathLoss(std::make_tuple(this->_positionAPX[i], this->_positionAPY[i], this->_positionAPZ[i]), std::make_tuple(this->_positionAPX[j], this->_positionAPY[j], this->_positionAPZ[j]), std::get<1>(conf[i]));
				if (rx >= std::get<0>(conf[j]))
					conflictsi.push_back(j);
			}
		}

		conflicts.push_back(conflictsi);
	}

	return conflicts;
}

std::vector<double> Simulation::getLogDistanceRSSIs() const {
	std::vector<double> rssis;
	for (unsigned int i = 0; i < this->_positionAPX.size(); i++) {
		double rssi = -200;
		for (unsigned int j = 0; j < this->_positionAPX.size(); j++) {
			if (i != j) {
				double rx = Simulation::pathLoss(std::make_tuple(this->_positionAPX[i], this->_positionAPY[i], this->_positionAPZ[i]), std::make_tuple(this->_positionAPX[j], this->_positionAPY[j], this->_positionAPZ[j]), std::get<1>(this->_configuration[i]));
				if (rssi < rx)
					rssi = rx;
			}
		}
		rssis.push_back(rssi);
	}

	return rssis;
}

double Simulation::pathLoss(std::tuple<double, double, double> source, std::tuple<double, double, double> target, double txPower) {
	double d = Simulation::distance(source, target);
	return txPower - 46.67 - 10 * 3 * log10(d);
}

double Simulation::distance(std::tuple<double, double, double> source, std::tuple<double, double, double> target) {
	return sqrt(pow(std::get<0>(source) - std::get<0>(target), 2) + pow(std::get<1>(source) - std::get<1>(target), 2) + pow(std::get<2>(source) - std::get<2>(target), 2));
}

WifiNetDevice* Simulation::getWifiDevice(Node* node) {
	return dynamic_cast<WifiNetDevice*>(GetPointer(node->GetDevice(0)));
}

WifiMac* Simulation::getMAC(Node* node) {
	return GetPointer(getWifiDevice(node)->GetMac());
}

WifiMacHeader Simulation::createAdHocMacHeader(Node* from, Node* to) {
	WifiMac* macFrom = getMAC(from), *macTo = getMAC(to);
	WifiMacHeader wmh;
	wmh.SetAddr1(macTo->GetAddress());
	wmh.SetAddr2(macFrom->GetBssid());
	wmh.SetAddr3(macFrom->GetAddress());
	wmh.SetType(WIFI_MAC_DATA);
	wmh.SetDsFrom();

	return wmh;
}

WifiMode Simulation::getWifiMode(Node* from, Node* to) {
	WifiNetDevice* from_wnd = getWifiDevice(from);
	WifiMacHeader adhoc = createAdHocMacHeader(from, to);
	return GetPointer(from_wnd->GetRemoteStationManager())->GetDataTxVector(adhoc).GetMode();
}

unsigned int Simulation::getMCSValue(Node* from, Node* to) {
	WifiMode wm = getWifiMode(from, to);
	return wm.GetMcsValue();
}

std::string Simulation::getMCSClass(Node* from, Node* to) {
	WifiMode wm = getWifiMode(from, to);
	std::string modClass = "";
	switch (wm.GetModulationClass()) {
		case WIFI_MOD_CLASS_HT: modClass = "Ht"; break;
		case WIFI_MOD_CLASS_VHT: modClass = "Vht"; break;
		case WIFI_MOD_CLASS_HE: modClass = "He"; break;
		default: break;
	}

	return modClass;
}
