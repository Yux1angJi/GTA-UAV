#include "Server.h"
//#include <thread>

#include <string>

#include "lib/rapidjson/stringbuffer.h"
#include "lib/main.h"
#include "Functions.h"

//#include <zmqpp.hpp>

#include <zmq.hpp>


//using namespace rapidjson;
using namespace std;

Server::Server(unsigned int port) {
	log("Server::Server initialize");
	socket = zmq::socket_t(ctx, zmq::socket_type::pair);
	socket.bind("tcp://127.0.0.1:8000");
}

//void Server::checkClient() {
//
//}

void Server::checkRecvMessage() {
	//log("Server::checkRecvMessage");

	zmq::message_t message;

	zmq::pollitem_t items[] = {
		{static_cast<void*>(socket), 0, ZMQ_POLLIN, 0}
	};

	zmq::poll(&items[0], 1, 1);

	if (items[0].revents & ZMQ_POLLIN) {
		socket.recv(&message);
		string jsonText;
		Document d;
		jsonText = message.to_string();

		d.Parse(jsonText);



		// Handle received Message:

		if (d.HasMember("commands")) {
			printf("Commands received\n");
			const Value& commands = d["commands"];
			scenario.setCommands(commands["throttle"].GetFloat(), commands["brake"].GetFloat(), commands["steering"].GetFloat());
		}
		else if (d.HasMember("config")) {
			//Change the message values and keep the others the same
			printf("Config received\n");
			const Value& config = d["config"];
			const Value& sc = config["scenario"];
			const Value& dc = config["dataset"];
			scenario.config(sc, dc);
		}
		else if (d.HasMember("start")) {
			//Set the message values and randomize the others. Start sending the messages
			printf("Start received\n");
			const Value& config = d["start"];
			const Value& sc = config["scenario"];
			const Value& dc = config["dataset"];
			scenario.start(sc, dc);

			clientStarted = true;
			//sendOutputs = true;
		}
		else if (d.HasMember("stop")) {
			//Stop sendig messages, keep client connected
			printf("Stop received\n");
			//sendOutputs = false;
			scenario.stop();
			clientStarted = false;
		}
		else if (d.HasMember("StartRecording")) {
			scenario.setRecording_active(true);
		}
		else if (d.HasMember("StopRecording")) {
			scenario.setRecording_active(false);
		}
		else if (d.HasMember("GoToLocation")) {
			const Value& target = d["GoToLocation"];
			scenario.goToLocation(target["x"].GetFloat(), target["y"].GetFloat(), target["z"].GetFloat(), target["speed"].GetFloat());
		}
		else if (d.HasMember("TeleportToLocation")) {
			const Value& target = d["TeleportToLocation"];
			scenario.teleportToLocation(target["x"].GetFloat(), target["y"].GetFloat(), target["z"].GetFloat());
		}
		else if (d.HasMember("SetCameraPositionAndRotation")) {
			printf("New Camera Settings received\n");
			const Value& camset = d["SetCameraPositionAndRotation"];
			scenario.setCameraPositionAndRotation(camset["x"].GetFloat(), camset["y"].GetFloat(), camset["z"].GetFloat(), camset["rot_x"].GetFloat(), camset["rot_y"].GetFloat(), camset["rot_z"].GetFloat());
		}
		else if (d.HasMember("CreatePed")) {
			const Value& pd = d["CreatePed"];
			scenario.createPed(pd["model"].GetString(), pd["relativeForward"].GetFloat(), pd["relativeRight"].GetFloat(), pd["relativeUp"].GetFloat(), pd["heading"].GetFloat(), pd["placeOnGround"].GetBool(), pd["animDict"].GetString(), pd["animName"].GetString());
		}
		else if (d.HasMember("CreateVehicle")) {
			const Value& vd = d["CreateVehicle"];
			scenario.createVehicle(vd["model"].GetString(), vd["relativeForward"].GetFloat(), vd["relativeRight"].GetFloat(), vd["heading"].GetFloat(), vd["color"].GetInt(), vd["color2"].GetInt(), vd["placeOnGround"].GetBool(), vd["withLifeJacketPed"].GetBool());
		}
		else if (d.HasMember("SetWeather")) {
			const Value& wd = d["SetWeather"];
			scenario.setWeather(wd["weather"].GetString());
		}
		else if (d.HasMember("SetClockTime")) {
			const Value& ct = d["SetClockTime"];
			scenario.setClockTime(ct["hour"].GetInt(), ct["minute"].GetInt(), ct["second"].GetInt());
		}

		else {
			return; //Invalid message
		}
	}

}






void Server::checkSendMessage() {
	log("Server::CheckSendMessage");

		
	// TODO  Maybe some speed improvement could be made here by using the buffers more efficiently (zero copy)
		
	// Note that scenario.generateMessage() calls exporter.screenCapturer.capture() so this ordering is relevant
	StringBuffer messageJSON = scenario.generateMessage();
	string data = messageJSON.GetString();

	// Send Image TODO
	int len = scenario.exporter.screenCapturer->length;
	UINT8 * pixels = scenario.exporter.screenCapturer->pixels;
	socket.send(zmq::buffer(pixels, len));

	// Send JSON
	socket.send(zmq::buffer(data), zmq::send_flags::none);

	lastSentMessageTime = std::clock();

	
}