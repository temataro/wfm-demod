#define RGFW_IMPLEMENTATION
#define RGFW_OPENGL /* if this line is not added, OpenGL functions will not \
                       be included */
#include "../thirdparty/RGFW.h"
#include <time.h>

int main(void)
{
    RGFW_rect rect = {
        100, 100, // x, y
        800, 600 // w, h
    };
    RGFW_window *win = RGFW_createWindow(
        "Hello RGFW", rect, (RGFW_windowFloating | RGFW_windowFocusOnShow)
    );
 
    while (!RGFW_window_shouldClose(win))
    {   
        while (RGFW_window_checkEvent(win)) {
          switch (win->event.type) {
             case RGFW_quit:
               break;
             case RGFW_keyPressed:
               break;
             case RGFW_mousePosChanged:
                break;
          }
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0x18 / 255.f, 0x18 / 255.f, 0x18 / 255.f, 1.f);

        RGFW_window_swapBuffers(win); // end_loop: update the display
        sleep(100);
    }

    RGFW_window_close(win);

    return 0;
}
