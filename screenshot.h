#pragma once

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "Gdi32.lib")
#pragma comment(lib, "user32.lib")

//#include <ws2tcpip.h>
//#include <Windows.h>
//#include <wingdi.h>

#include "pomocne_funkcje.h"

struct screenshot {
    HWND hDesktopWnd;
    HDC hDesktopDC;
    HDC hCaptureDC;

    BITMAPINFO MyBMInfo = { 0 };

    HBITMAP hCaptureBitmap;

    int cap_width;
    int cap_height;

    uint8_t* buffer;

    screenshot(int cap_width, int cap_height) :
        cap_width(cap_width), cap_height(cap_height) {
        buffer = (uint8_t*)malloc(cap_width * cap_height * 3);
        //printf("Zlap okno do capurowania\n");
        //Sleep(5000);
        hDesktopWnd = GetForegroundWindow();//FindWindowW(NULL, L"Obraz w obrazie");
        ASSERT_Z_INSTRUKCJA(hDesktopWnd != nullptr, printf("Kod errora: %lld\n", GetLastError()););
        hDesktopDC = GetDC(hDesktopWnd);
        hCaptureDC = CreateCompatibleDC(hDesktopDC);

        hCaptureBitmap = CreateCompatibleBitmap(hDesktopDC, cap_width, cap_height);
        SelectObject(hCaptureDC, hCaptureBitmap);
        BitBlt(hCaptureDC, 0, 0, cap_width, cap_height, hDesktopDC, 0, 20, SRCCOPY);

        MyBMInfo.bmiHeader.biSize = sizeof(MyBMInfo.bmiHeader);
        GetDIBits(hDesktopDC, hCaptureBitmap, 0, 0, NULL, &MyBMInfo, DIB_RGB_COLORS);

        MyBMInfo.bmiHeader.biBitCount = 24;
        MyBMInfo.bmiHeader.biCompression = BI_RGB;
        MyBMInfo.bmiHeader.biHeight = -MyBMInfo.bmiHeader.biHeight;
    }

    void screen_cap(unsigned char* output) {
        //SelectObject(hCaptureDC, hCaptureBitmap); //Moze dzia³a bez tego ale wole nie ryzykowac
        BitBlt(hCaptureDC, 0, 0, cap_width, cap_height, hDesktopDC, 0, 20, SRCCOPY);

        GetDIBits(hDesktopDC, hCaptureBitmap, 0, cap_height, buffer, &MyBMInfo, DIB_RGB_COLORS);
        for (int i = 0, j = 0; i < cap_width * cap_height * 3; i += 3, j+= 4) {
            output[j + 0] = buffer[i + 2];
            output[j + 1] = buffer[i + 1];
            output[j + 2] = buffer[i];
            output[j + 3] = 0xFF;
        }
    }

    ~screenshot(){
        free(buffer);
        // tu chyba brakuje destuktora tych windowsowych obiektów
    }
};


