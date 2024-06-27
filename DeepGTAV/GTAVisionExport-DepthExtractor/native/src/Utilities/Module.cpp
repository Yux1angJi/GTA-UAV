#include "Module.h"

bool GetModuleInfo(MODULEENTRY32 &out_info, CHAR *lpModuleName, DWORD dwProcessId)
{
   MODULEENTRY32 ModuleEntry = {0};
   HANDLE hSnapShot = ::CreateToolhelp32Snapshot( TH32CS_SNAPMODULE, dwProcessId );
   if(!hSnapShot) {
      return false;
   }

   ModuleEntry.dwSize = sizeof(ModuleEntry);
   BOOL bModule = ::Module32First( hSnapShot, &ModuleEntry );
   while(bModule) {
      if(!strcmp( ModuleEntry.szModule, lpModuleName ) ) {
          out_info = ModuleEntry;
         ::CloseHandle( hSnapShot );
         return true;
      }
      bModule = ::Module32Next( hSnapShot, &ModuleEntry );
   }
   ::CloseHandle( hSnapShot );
   return false;
}

size_t GetAllModuleInfo(std::vector<MODULEENTRY32> &out_info, DWORD dwProcessId)
{
    MODULEENTRY32 ModuleEntry = {0};
    HANDLE hSnapShot = ::CreateToolhelp32Snapshot( TH32CS_SNAPMODULE, dwProcessId );
    if(!hSnapShot) {
        return false;
    }

    ModuleEntry.dwSize = sizeof(ModuleEntry);
    BOOL bModule = ::Module32First( hSnapShot, &ModuleEntry );
    while(bModule) {
        out_info.push_back(ModuleEntry);
        bModule = ::Module32Next( hSnapShot, &ModuleEntry );
    }
    ::CloseHandle( hSnapShot );
    return out_info.size();
}




bool IsAddressInD3D11DLL(void *address, DWORD dwProcessId)
{
    static MODULEENTRY32 s_d3d11dll = {0};
    if(s_d3d11dll.dwSize==0) {
        GetModuleInfo(s_d3d11dll, "d3d11.dll", dwProcessId);
    }

    void *range_begin = s_d3d11dll.modBaseAddr;
    void *range_end = (char*)range_begin + s_d3d11dll.modBaseSize;
    return address >= range_begin && address < range_end;
}

bool DetectNvidiaNSight()
{
    std::vector<MODULEENTRY32> modules;
    GetAllModuleInfo(modules);
    for(std::vector<MODULEENTRY32>::iterator i=modules.begin(); i!=modules.end(); ++i) {
        if(strcmp(i->szModule, "Nvda.Graphics.Interception")!=NULL) {
            return true;
        }
    }
    return false;
}
