#include <nanogui/nanogui.h>
#include <iostream>
#include "gui.h"

int main() {
    try {
        nanogui::init();

        {
            // Create our custom screen
            FutabaScreen screen(800, 600);
            
            // Enter the continuous real-time render loop
            screen.renderLoop();
            
        } // 'screen' goes out of scope here, triggering the destructor to clean up OpenGL/CUDA
        
        nanogui::shutdown();

    } catch (const std::exception &e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}