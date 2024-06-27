#ifndef _ist_D3DHookInterface_Utilities_Module_h_
#define _ist_D3DHookInterface_Utilities_Module_h_

#include <windows.h>
#include <TlHelp32.h>
#include <vector>
#include <intrin.h>


/// vtable の取得/設定
/// たぶんこれは C++ 的に違法だと思いますが、少なくとも VC では機能します。
template<class T> inline void** get_vtable(T _this) { return ((void***)_this)[0]; }
template<class T> inline void   set_vtable(T _this, void **vtable) { ((void***)_this)[0] = vtable; }


/// 指定のプロセス内の指定の名前のモジュール情報を取得
/// dwProcessId: 0 だと current process 扱いになります
bool GetModuleInfo(MODULEENTRY32 &out_info, char *lpModuleName, DWORD dwProcessId=0);

/// 指定のプロセス内の全モジュール情報を取得
size_t GetAllModuleInfo(std::vector<MODULEENTRY32> &out_info, DWORD dwProcessId=0);


/// 指定のアドレスが d3d11.dll モジュール内かを調べます
bool IsAddressInD3D11DLL(void *address, DWORD dwProcessId=0);
/// return address が d3d11.dll モジュール内なら true を返します
#define IsReturnAddressInD3D11DLL() IsAddressInD3D11DLL(_ReturnAddress())


/// NVIDIA Nsight のモジュールを検出したら true を返します。
/// hook と Nsight を併用したらクラッシュするので、Nsight を検出したら hook しないようにする必要があります。
/// その判別用に用意されています。
bool DetectNvidiaNSight();


#endif // _ist_D3DHookInterface_Utilities_Module_h_
