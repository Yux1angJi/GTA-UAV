#pragma once

#include "Scenario.h"
#include "Functions.h"
#include <chrono>
#include <zmq.hpp>
//#include <zmqpp/zmqpp.hpp>
//#include <zmqpp.hpp>



class Server {
private:
	zmq::context_t ctx;
	zmq::socket_t socket;




public:
	Scenario scenario;
	
	bool clientStarted = false;
	std::clock_t lastSentMessageTime = std::clock();


	Server(unsigned int port);
	void checkRecvMessage();
	void checkSendMessage();
	//void checkClient();
};