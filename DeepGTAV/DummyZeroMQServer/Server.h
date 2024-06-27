//#include "Scenario.h"
//#include "Functions.h"
#include <chrono>
#include <zmq.hpp>
//#include <zmqpp/zmqpp.hpp>
//#include <zmqpp.hpp>

//#pragma comment(lib, "libzmq-v141-mt-gd-4_3_3.lib")

class Server {
private:
	//zmqpp::context context;
	//zmqpp::socket socket;

	zmq::context_t ctx;
	zmq::socket_t socket;


	std::clock_t lastSentMessageTime = std::clock();


public:
	//Scenario scenario;

	//bool clientStarted = false;

	Server(unsigned int port);
	void checkRecvMessage();
	void checkSendMessage();
	//void checkClient();
};