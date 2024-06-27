#include "Server.h"
#include <thread>
#include "lib/rapidjson/document.h"
#include "lib/rapidjson/stringbuffer.h"
#include "lib/main.h"
#include "Constants.h"

#include <zmq.hpp>

using namespace rapidjson;


Server::Server(unsigned int port) {
	struct sockaddr_in server;
	freopen("deepgtav.log", "w", stdout);

	printf("\nInitializing Winsock...");
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		printf("Failed. Error Code: %d", WSAGetLastError());
	}
	printf("Initialized.\n");

	if ((ServerSocket = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
		printf("Could not create socket: %d", WSAGetLastError());
	}
	printf("Socket created.\n");

	server.sin_family = AF_INET;
	server.sin_addr.s_addr = INADDR_ANY;
	server.sin_port = htons(port);

	if (bind(ServerSocket, (struct sockaddr *)&server, sizeof(server)) == SOCKET_ERROR) {
		printf("Bind failed with error code: %d", WSAGetLastError());
	}
	printf("Bind done.\n");

	printf("Listening...\n");
	if (listen(ServerSocket, 1) == SOCKET_ERROR) {
		printf("Could not listen: %d", WSAGetLastError());
	}

	if (ioctlsocket(ServerSocket, FIONBIO, &iMode) != NO_ERROR) {
		printf("Server ioctlsocket failed");
	}

}

void Server::checkClient(){
	SOCKET tmpSocket = SOCKET_ERROR;
	tmpSocket = accept(ServerSocket, NULL, NULL);
	if (tmpSocket != SOCKET_ERROR) {
		printf("Connection accepted.\n");
		ClientSocket = tmpSocket;
		if (ioctlsocket(ClientSocket, FIONBIO, &iMode) != NO_ERROR) {
			printf("Client ioctlsocket failed");
			return;
		}
		clientConnected = true;
	}
}

void Server::checkRecvMessage() {
	int result;
	Document d;
	int error;
	
	if (recvMessageLen == 0) {
		recv(ClientSocket, (char*)&recvMessageLen, 4, 0); //Receive message len first
		error = WSAGetLastError();
		if (error == WSAEWOULDBLOCK) return;
		if (error != 0) {
			printf("\nError receiving message length: %d", error);
			resetState();
			return;
		}
	}

	while (bytesRead < recvMessageLen){
		result = recv(ClientSocket, json + bytesRead, recvMessageLen - bytesRead, 0);
		error = WSAGetLastError();
		if (error == WSAEWOULDBLOCK) return;
		if (error != 0 || result < 1) {
			printf("\nError receiving message: %d", error);
			resetState();
			return;
		}
		bytesRead = bytesRead + result;
	}

	json[bytesRead] = '\0';
	bytesRead = 0;
	recvMessageLen = 0;

	d.Parse(json);

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
		sendOutputs = true;
	}
	else if (d.HasMember("stop")) {
		//Stop sendig messages, keep client connected
		printf("Stop received\n");
		sendOutputs = false;
		scenario.stop();
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
		scenario.createPed(pd["model"].GetFloat(), pd["relativeForward"].GetFloat(), pd["relativeRight"].GetFloat(), pd["heading"].GetFloat(), pd["task"].GetInt());
	}
	else if (d.HasMember("CreateVehicle")) {
		const Value& vd = d["CreateVehicle"];
		scenario.createVehicle(vd["model"].GetString(), vd["relativeForward"].GetFloat(), vd["relativeRight"].GetFloat(), vd["heading"].GetFloat(), vd["color"].GetInt(), vd["color2"].GetInt());
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

void Server::checkSendMessage() {
	log("Server::CheckSendMessage");
	int error;
	int r;

    //Return from this function to allow time gaps between series
    if (scenario.collectTracking) {
        if (scenario.trSeriesGap) {
            if (((float)(std::clock() - lastSentMessage) / CLOCKS_PER_SEC) < scenario.trSeriesGapTime) {
                return;
            }
            //Set the series gap to false once we have passed the gap time
            scenario.trSeriesGap = false;
        }
    }

	if (sendOutputs && (((float)(std::clock() - lastSentMessage) / CLOCKS_PER_SEC) > (1.0 / scenario.rate))) {
		if (messageSize == 0) {
			message = scenario.generateMessage();
			chmessage = message.GetString();
			messageSize = message.GetSize();
		}		

		if (!frameSent) {
			if (!readyToSend) {
				send(ClientSocket, (const char*)&scenario.exporter.screenCapturer->length, sizeof(scenario.exporter.screenCapturer->length), 0);
				error = WSAGetLastError();
				if (error == WSAEWOULDBLOCK) return;
				if (error != 0) {
					printf("\nError sending frame length: %d", error);
					resetState();
					return;
				}
				readyToSend = true;
				sendMessageLen = 0;
			}

			while (readyToSend && (sendMessageLen < scenario.exporter.screenCapturer->length)) {
				r = send(ClientSocket, (const char*)(scenario.exporter.screenCapturer->pixels + sendMessageLen), scenario.exporter.screenCapturer->length - sendMessageLen, 0);
				error = WSAGetLastError();
				if (error == WSAEWOULDBLOCK) return;
				if (error != 0 || r <= 1) {
					printf("\nError sending frame: %d", error);
					resetState();
					return;
				}
				sendMessageLen = sendMessageLen + r;
			}
			readyToSend = false;
			frameSent = true;
		}

		if (frameSent) {
			if (!readyToSend) {
				send(ClientSocket, (const char*)&messageSize, sizeof(messageSize), 0);
				error = WSAGetLastError();
				if (error == WSAEWOULDBLOCK) return;
				if (error != 0) {
					printf("\nError sending message length: %d", error);
					resetState();
					return;
				}
				readyToSend = true;
				sendMessageLen = 0;
			}

			while (readyToSend && (sendMessageLen < messageSize)) {
				r = send(ClientSocket, (const char*)(chmessage + sendMessageLen), messageSize - sendMessageLen, 0);
				error = WSAGetLastError();
				if (error == WSAEWOULDBLOCK) return;
				if (error != 0 || r <= 1) {
					printf("\nError sending message: %d", error);
					resetState();
					return;
				}
				sendMessageLen = sendMessageLen + r;
			}
			readyToSend = false;
			messageSize = 0;
			frameSent = false;
		}
		lastSentMessage = std::clock();
	}
	log("Server: Message is sent");
}

void Server::resetState() {
	shutdown(ClientSocket, SD_SEND);
	closesocket(ClientSocket);

	clientConnected = false;
	sendOutputs = false;
	bytesRead = 0;
	recvMessageLen = 0;
	sendMessageLen = 0;
	readyToSend = false;
	frameSent = false;
	messageSize = 0;

	scenario.stop();
}