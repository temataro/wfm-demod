/*
Example to get you started :

linux : gcc main.c -lX11 -lXrandr -lGL
windows : gcc main.c -lopengl32 -lgdi32
macos : gcc main.c -framework Cocoa -framework CoreVideo -framework OpenGL
-framework IOKit
*/

#define RGFW_IMPLEMENTATION
#include "../thirdparty/RGFW.h"

u8 icon[4 * 3 * 3] = {0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF,
                      0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF,
                      0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0x00,
                      0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF};

int main()
{
    RGFW_window *win =
        RGFW_createWindow("name", RGFW_RECT(100, 100, 500, 500), (u64)0);

    RGFW_window_setIcon(win, icon, RGFW_AREA(3, 3), 4);

    while (RGFW_window_shouldClose(win) == RGFW_FALSE)
    {
        while (RGFW_window_checkEvent(win))
        {
            if (win->event.type == RGFW_quit
                || RGFW_isPressed(win, RGFW_escape))
                break;
        }

        RGFW_window_swapBuffers(win);

        glClearColor(1, 1, 1, 1);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    RGFW_window_close(win);
}
