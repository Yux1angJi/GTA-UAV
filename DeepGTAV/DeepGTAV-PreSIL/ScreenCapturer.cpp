#include "ScreenCapturer.h"
#include "lib/natives.h"
#include <stdlib.h>

ScreenCapturer::ScreenCapturer(int Width, int Height){
	windowWidth = Width;
	windowHeight = Height;

	// Round up the scan line size to a multiple of 4
	length = ((windowWidth * 3 + 3) / 4 * 4) * windowHeight;

	hWindowDC = GetDC(NULL);
	hCaptureDC = CreateCompatibleDC(hWindowDC);
	hCaptureBitmap = CreateCompatibleBitmap(hWindowDC, windowWidth, windowHeight);
	SelectObject(hCaptureDC, hCaptureBitmap);
	SetStretchBltMode(hCaptureDC, COLORONCOLOR);

	pixels = (UINT8*)malloc(length);
	info.biSize = sizeof(BITMAPINFOHEADER);
	info.biPlanes = 1;
	info.biBitCount = 24;
	info.biWidth = windowWidth;
	info.biHeight = -windowHeight;
	info.biCompression = BI_RGB;
	info.biSizeImage = 0;
}

ScreenCapturer::~ScreenCapturer(){
	free(pixels);
	ReleaseDC(hWnd, hWindowDC);
	DeleteDC(hCaptureDC);
	DeleteObject(hCaptureBitmap);
}

void ScreenCapturer::capture() {
	BitBlt(hCaptureDC, 0, 0, windowWidth, windowHeight, hWindowDC, 0, 0, SRCCOPY);
	GetDIBits(hCaptureDC, hCaptureBitmap, 0, windowHeight, pixels, (BITMAPINFO*)&info, DIB_RGB_COLORS);
}