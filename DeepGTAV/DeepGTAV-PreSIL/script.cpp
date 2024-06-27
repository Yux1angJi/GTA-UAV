/*
	THIS FILE IS A PART OF GTA V SCRIPT HOOK SDK
				http://dev-c.com			
			(C) Alexander Blade 2015
*/

#include "lib/script.h"
#include "Server.h"
#include "Functions.h"

void ScriptMain()
{

	log("ScriptMain Started");
	Server server(8000);
	while (true) {

		server.checkRecvMessage();
		
		if (server.clientStarted) {
			if (((float)(std::clock() - server.lastSentMessageTime) / CLOCKS_PER_SEC) > (1.0 / server.scenario.rate)) {
				server.checkSendMessage();
			}
			server.scenario.run();
		}
		scriptWait(0);
	}
}
