// =================================================================================
// Memory 
// =================================================================================
namespace Memory
{
	// Init / Shutdown
	void Init(); // initializes hooking engine
	void CleanUp();

	// Memory Operations
	void Copy(DWORD64 pAddress, BYTE* bData, size_t stSize);
	void Set(DWORD64 pAddress, BYTE* bData, size_t stSize);
	bool Compare(const BYTE* pData, const BYTE* bMask, const char* szMask);
	DWORD64 Find(DWORD64 dwAddress, DWORD dwLength, const BYTE* bMask, const char* szMask);

	// Module Operations
	char* GetModulePath(HMODULE hPath);
	char* GetModulePath(char* sPath);
	DWORD64 GetModuleSize(HMODULE hModule);

	// Hooking
	bool HookFunction(DWORD64 pAddress, void* pDetour, void** ppOriginal);
	bool HookLibraryFunction(char* sLibrary, char* sName, void* pDetour, void** ppOriginal);
}

// =================================================================================
// Game Memory 
// =================================================================================
#include "GameMemory.h"