#include "export.h"
#include "nativeCaller.h"
#include "natives.h"
#include <d3d11.h>
#include <cassert>
#include <wrl/client.h>
#include <system_error>
#include <memory>
#include <vector>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <Eigen/Core>
using Microsoft::WRL::ComPtr;
using std::unique_ptr;
using std::vector;
using std::mutex;
using std::condition_variable;
using std::unique_lock;
using std::swap;
using Eigen::Matrix4f;
using std::chrono::high_resolution_clock;
using std::chrono::time_point;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
static ComPtr<ID3D11Device> lastDev;
static ComPtr<ID3D11DeviceContext> lastCtx;
static ComPtr<ID3D11Texture2D> depthRes;
static ComPtr<ID3D11Texture2D> colorRes;
static ComPtr<ID3D11Buffer> constantBuf;
static vector<unsigned char> depthBuf;
static vector<unsigned char> colorBuf;
static vector<unsigned char> stencilBuf;
static bool requestPrevBuffers = false;
static bool prevBufferReadyToExtract = false;
static rage_matrices constants;
static bool request_copy = false;
static mutex copy_mtx;
static condition_variable copy_cv;

static time_point<high_resolution_clock> last_depth_time;
static time_point<high_resolution_clock> last_color_time;
static time_point<high_resolution_clock> last_constant_time;

static char* logDir = getenv("GTAV_LOG_FILE");
static bool LOGGING = false;
static void log(std::string str) {
    if (LOGGING && logDir != NULL) {
        FILE* f = fopen(logDir, "a");
        fprintf(f, str.c_str());
        fprintf(f, "\n");
        fclose(f);
    }
}

static void unpack_depth(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Resource* src, vector<unsigned char>& dst, vector<unsigned char>& stencil)
{
	HRESULT hr = S_OK;
	int screenResX;
	int screenResY;
	ComPtr<ID3D11Texture2D> src_tex;
	GRAPHICS::_GET_SCREEN_ACTIVE_RESOLUTION(&screenResX, &screenResY);
	
	hr = src->QueryInterface(__uuidof(ID3D11Texture2D), &src_tex);
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
	D3D11_TEXTURE2D_DESC src_desc;
	src_tex->GetDesc(&src_desc);
	assert(src_desc.Format == DXGI_FORMAT_R32G8X24_TYPELESS);
	D3D11_MAPPED_SUBRESOURCE src_map = { 0 };
	hr = ctx->Map(src, 0, D3D11_MAP_READ, 0, &src_map);
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
	if (dst.size() != src_desc.Height * src_desc.Width * 4) dst = vector<unsigned char>(src_desc.Height * src_desc.Width * 4);
	if (stencil.size() != src_desc.Height * src_desc.Width) stencil = vector<unsigned char>(src_desc.Height * src_desc.Width);
	
	if (screenResX >= src_desc.Width)
	{
		for (int x = 0; x < src_desc.Width; ++x)
		{
			for (int y = 0; y < src_desc.Height; ++y)
			{
				const float* src_f = (const float*)((const char*)src_map.pData + src_map.RowPitch*y + (x * 8));
				unsigned char* dst_p = &dst[src_desc.Width * 4 * y + (x * 4)];
				unsigned char* stencil_p = &stencil[src_desc.Width * y + x];
				memmove(dst_p, src_f, 4);
				memmove(stencil_p, src_f + 1, 1);
			}
		}
	}
	else
	{
		// resample, for when depth map is bigger than screen image.
		float scale = ((float) src_desc.Width) / ((float) screenResX);
		dst = vector<unsigned char>(screenResY * screenResX * 4);

		for (int x = 0; x < screenResX; ++x) // screenResX
		{
			int scaledX = int(x*scale);
			
			for (int y = 0; y < screenResY; ++y) //screenResY
			{
				int scaledY = int(y*scale);
				const float* src_f = (const float*)((const char*)src_map.pData + int(src_map.RowPitch*scaledY + (scaledX * 8)));
				unsigned char* dst_p = &dst[screenResX * 4 * y + (x * 4)];
				unsigned char* stencil_p = &stencil[screenResX * y + x];
				memmove(dst_p, src_f, 4);
				memmove(stencil_p, src_f + 1, 1);
			}
		}
	}
		
	ctx->Unmap(src, 0);
}
static ComPtr<ID3D11Texture2D> CreateTexHelper(ID3D11Device* dev, DXGI_FORMAT fmt, int width, int height, int samples)
{
	D3D11_TEXTURE2D_DESC desc = { 0 };
	desc.Format = fmt;
	desc.ArraySize = 1;
	desc.BindFlags = 0;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	desc.Height = height;
	desc.Width = width;
	desc.MipLevels = 1;
	desc.MiscFlags = 0;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D11_USAGE_STAGING;
	ComPtr<ID3D11Texture2D> result;
	HRESULT hr = S_OK;
	hr = dev->CreateTexture2D(&desc, nullptr, &result);
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
	return result;

}
static ComPtr<ID3D11Buffer> CreateStagingBuffer(ID3D11Device* dev, int size) {
	ComPtr<ID3D11Buffer> result;
	D3D11_BUFFER_DESC desc = { 0 };
	desc.BindFlags = 0;
	desc.ByteWidth = size;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	desc.MiscFlags = 0;
	desc.Usage = D3D11_USAGE_STAGING;
	dev->CreateBuffer(&desc, nullptr, &result);
	return result;
}

void CreateTextureIfNeeded(ID3D11Device* dev, ID3D11Resource* for_res, ComPtr<ID3D11Texture2D>* tex_target)
{
	ComPtr<ID3D11Texture2D> tex;
	HRESULT hr = S_OK;
	D3D11_TEXTURE2D_DESC desc = { 0 };
	hr = for_res->QueryInterface(__uuidof(ID3D11Texture2D), &tex);
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
	tex->GetDesc(&desc);
	if(*tex_target == nullptr)
	{
		*tex_target = CreateTexHelper(dev, desc.Format, desc.Width, desc.Height, desc.SampleDesc.Count);
	}
	D3D11_TEXTURE2D_DESC tex_desc;
	(*tex_target)->GetDesc(&tex_desc);
	if (tex_desc.Width != desc.Width || tex_desc.Height != desc.Height)
	{
		*tex_target = CreateTexHelper(dev, desc.Format, desc.Width, desc.Height, desc.SampleDesc.Count);
	}
}
int getBitsPerPixel(DXGI_FORMAT fmt)
{
	switch(fmt)
	{
	case DXGI_FORMAT_B8G8R8A8_UNORM:
		return 4;
	default:
		return -1;
	}
}
void copyTexToVector(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Resource* res, vector<unsigned char>& buffer)
{
	ComPtr<ID3D11Texture2D> tex;
	HRESULT hr;
	hr = res->QueryInterface(__uuidof(ID3D11Texture2D), &tex);
	D3D11_TEXTURE2D_DESC desc;
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
	tex->GetDesc(&desc);
	ComPtr<ID3D11Texture2D> tex_copy;
	CreateTextureIfNeeded(dev, tex.Get(), &tex_copy);
	ctx->CopyResource(tex_copy.Get(), tex.Get());
	D3D11_MAPPED_SUBRESOURCE map;
	auto bpp = getBitsPerPixel(desc.Format);
	if (bpp == -1) throw std::invalid_argument("unsupported resource type");
	hr = ctx->Map(tex_copy.Get(), 0, D3D11_MAP_READ, 0, &map);
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
	if (buffer.size() != desc.Height * desc.Width * bpp) buffer = vector<unsigned char>(desc.Height * desc.Width * bpp);
	for(int y = 0; y < desc.Height; ++y)
	{
		for (int x = 0; x < desc.Width; ++x) {
			unsigned char* p = &buffer[y * desc.Width * bpp + (x*4)];
			unsigned char* b = (unsigned char*)map.pData + map.RowPitch*y + (x*4);
			p[0] = b[2];
			p[1] = b[1];
			p[2] = b[0];
			p[3] = b[3];
			//memmove(p, b, desc.Width * bpp);
		}

	}
	ctx->Unmap(tex_copy.Get(), 0);
	
}
void CopyIfRequested()
{
	unique_lock<mutex> lk(copy_mtx);
	if(request_copy)
	{
		unpack_depth(lastDev.Get(), lastCtx.Get(), depthRes.Get(), depthBuf, stencilBuf);
		copyTexToVector(lastDev.Get(), lastCtx.Get(), colorRes.Get(), colorBuf);
		request_copy = false;
		lk.unlock(); //unlock the mutex so that and woken threads don't immediately block on it
		copy_cv.notify_all();
	}
}
void ExtractDepthBuffer(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Resource* res)
{
    log("ExtractDepthBuffer");
    if (requestPrevBuffers || !prevBufferReadyToExtract) {
        log("Saving extracted depth buffer");
        lastDev = dev;
        lastCtx = ctx;
        CreateTextureIfNeeded(dev, res, &depthRes);
        ctx->CopyResource(depthRes.Get(), res);
        last_depth_time = std::chrono::high_resolution_clock::now();
        //unpack_depth(dev, ctx, res, depthBuf, stencilBuf);

        if (requestPrevBuffers) {
            log("Set requestPrev to false/readytoExtract to true");
            requestPrevBuffers = false;
            prevBufferReadyToExtract = true;
        }
    }
}

void ExtractColorBuffer(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Resource* tex)
{
	lastDev = dev;
	lastCtx = ctx;
	CreateTextureIfNeeded(dev, tex, &colorRes);
	ctx->CopyResource(colorRes.Get(), tex);
	last_color_time = high_resolution_clock::now();
	//copyTexToVector(dev, ctx, tex, colorBuf);
	
}

void ExtractConstantBuffer(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Buffer* buf) {
	lastDev = dev;
	lastCtx = ctx;
	D3D11_BUFFER_DESC desc = { 0 };
	buf->GetDesc(&desc);
	if (constantBuf == nullptr) {
		constantBuf = CreateStagingBuffer(dev, desc.ByteWidth);
	}
	ctx->CopyResource(constantBuf.Get(), buf);
	last_constant_time = high_resolution_clock::now();
	
}

extern "C" {
	__declspec(dllexport) int export_get_depth_buffer(void** buf, bool updateWithOffsetDepth)
	{
        if (updateWithOffsetDepth) {
            requestPrevBuffers = true;
        }
		if (lastDev == nullptr || lastCtx == nullptr || depthRes == nullptr) return -1;
		unpack_depth(lastDev.Get(), lastCtx.Get(), depthRes.Get(), depthBuf, stencilBuf);
		*buf = &depthBuf[0];
		return depthBuf.size();
	}
	__declspec(dllexport) int export_get_color_buffer(void** buf)
	{
		if (lastDev == nullptr || lastCtx == nullptr || colorRes == nullptr) return -1;
		copyTexToVector(lastDev.Get(), lastCtx.Get(), colorRes.Get(), colorBuf);
		*buf = &colorBuf[0];
		return colorBuf.size();
	}
	__declspec(dllexport) int export_get_stencil_buffer(void** buf)
	{
		if (lastDev == nullptr || lastCtx == nullptr || depthRes == nullptr) return -1;
		unpack_depth(lastDev.Get(), lastCtx.Get(), depthRes.Get(), depthBuf, stencilBuf);
		*buf = &stencilBuf[0];
		return stencilBuf.size();
	}
	__declspec(dllexport) int export_get_constant_buffer(rage_matrices* buf) {
		if (constantBuf == nullptr) return -1;
		D3D11_MAPPED_SUBRESOURCE res = { 0 };
		lastCtx->Map(constantBuf.Get(), 0, D3D11_MAP_READ, 0, &res);
		memmove(buf, res.pData, sizeof(constants));
		lastCtx->Unmap(constantBuf.Get(), 0);
		return sizeof(rage_matrices);
	}

	__declspec(dllexport) long long int export_get_last_depth_time() {
		return duration_cast<milliseconds>(last_depth_time.time_since_epoch()).count();
	}
	__declspec(dllexport) long long int export_get_last_color_time() {
		return duration_cast<milliseconds>(last_color_time.time_since_epoch()).count();
	}
	__declspec(dllexport) long long int export_get_last_constant_time() {
		return duration_cast<milliseconds>(last_constant_time.time_since_epoch()).count();
	}

	__declspec(dllexport) long long int export_get_current_time() {
		return duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
	}

    __declspec(dllexport) int export_get_previous_depth_stencil_buffers(void** dBuf, void** sBuf, int &stencilSize)
    {
        if (!prevBufferReadyToExtract) {
            return -1;
        }
        log("Exporting previous buffers");
        if (lastDev == nullptr || lastCtx == nullptr || depthRes == nullptr) return -1;
        unpack_depth(lastDev.Get(), lastCtx.Get(), depthRes.Get(), depthBuf, stencilBuf);
        *dBuf = &depthBuf[0];
        *sBuf = &stencilBuf[0];
        stencilSize = stencilBuf.size();
        int size = depthBuf.size();
        prevBufferReadyToExtract = false;
        return size;
    }
}	
