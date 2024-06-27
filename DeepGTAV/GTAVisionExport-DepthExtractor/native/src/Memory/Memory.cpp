// =================================================================================
// Includes
// =================================================================================
#include "Includes.h"
#include <Psapi.h>
#include "Memory.h"

// =================================================================================
// Copy
// =================================================================================
void Memory::Copy(DWORD64 pAddress, BYTE* bData, size_t stSize)
{
	DWORD dwOldProtection;
	VirtualProtect((void*)pAddress, stSize, PAGE_EXECUTE_READWRITE, &dwOldProtection);
	memcpy((void*)pAddress, (void*)bData, stSize);
	VirtualProtect((void*)pAddress, stSize, dwOldProtection, &dwOldProtection);
}

// =================================================================================
// Set
// =================================================================================
void Memory::Set(DWORD64 pAddress, BYTE* bData, size_t stSize)
{
	DWORD dwOldProtection;
	VirtualProtect((void*)pAddress, stSize, PAGE_EXECUTE_READWRITE, &dwOldProtection);
	memset((void*)pAddress, (int)bData, stSize);
	VirtualProtect((void*)pAddress, stSize, dwOldProtection, &dwOldProtection);
}

// =================================================================================
// FindPattern
// =================================================================================
bool Memory::Compare(const BYTE* pData, const BYTE* bMask, const char* sMask)
{
	for (; *sMask; ++sMask, ++pData, ++bMask)
		if (*sMask == 'x' && *pData != *bMask)
			return false;

	return *sMask == NULL;
}
DWORD64 Memory::Find(DWORD64 dwAddress, DWORD dwLength, const BYTE* bMask, const char* sMask)
{
	for (DWORD i = 0; i < dwLength; i++)
		if (Compare((BYTE*)(dwAddress + i), bMask, sMask))
			return (DWORD64)(dwAddress + i);

	return 0;
}

// =================================================================================
// Module Path
// =================================================================================
char* Memory::GetModulePath(HMODULE hModule)
{
	// Path
	char* path = new char[512];
	if (GetModuleFileName(hModule, path, 512) == 0)
		return NULL;

	// Return
	return path;
}
char* Memory::GetModulePath(char* sPath)
{
	// Module Handle
	HMODULE hModule = GetModuleHandle(sPath);
	if (hModule == NULL)
		return NULL;

	// Path
	return GetModulePath(hModule);
}

// =================================================================================
// Module Size
// =================================================================================
DWORD64 Memory::GetModuleSize(HMODULE hModule)
{
	// Double-Check
	if (hModule == NULL) return 0;
	
	// Module Info
	MODULEINFO info;
	GetModuleInformation(GetCurrentProcess(), hModule, &info, sizeof(MODULEINFO));

	// Size
	return info.SizeOfImage;
}