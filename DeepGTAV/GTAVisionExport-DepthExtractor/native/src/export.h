#ifndef GTA_VISION_NATIVE_EXPORT_H
#define GTA_VISION_NATIVE_EXPORT_H
#include <d3d11.h>
#include <Eigen/Core>
void ExtractDepthBuffer(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Resource* tex);
void ExtractColorBuffer(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Resource* tex);
void ExtractConstantBuffer(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Buffer* buf);
void CopyIfRequested();
struct rage_matrices {
	Eigen::Matrix4f M;
	Eigen::Matrix4f MV;
	Eigen::Matrix4f MVP;
	Eigen::Matrix4f Vinv;
}; 

extern "C" {
	__declspec(dllexport) int export_get_depth_buffer(void** buf, bool updateWithOffsetDepth = false);
	__declspec(dllexport) int export_get_color_buffer(void** buf);
	__declspec(dllexport) int export_get_stencil_buffer(void** buf);
    __declspec(dllexport) int export_get_previous_depth_stencil_buffers(void** dBuf, void** sBuf, int &stencilSize);
	__declspec(dllexport) int export_get_constant_buffer(rage_matrices* buf);
}
#endif