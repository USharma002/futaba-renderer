#include <nanogui/nanogui.h>
#include <iostream>
#include "gui.h"

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    std::cout << "Debug: Main started" << std::endl;
    std::string scenePath = (argc > 1) ? argv[1] : "";
    try {
        nanogui::init();
        std::cout << "Debug: Nanogui initialized successfully" << std::endl;

        {
            std::cout << "Debug: Creating FutabaScreen..." << std::endl;
            // Create our custom screen
            FutabaScreen screen(800, 600, scenePath);
            std::cout << "Debug: FutabaScreen created successfully" << std::endl;
            
            // Enter the continuous real-time render loop
            std::cout << "Debug: Entering render loop..." << std::endl;
            screen.renderLoop();
            std::cout << "Debug: Exited render loop successfully" << std::endl;
            
        } // 'screen' goes out of scope here, triggering the destructor to clean up OpenGL/CUDA
        
        std::cout << "Debug: Shutting down Nanogui..." << std::endl;
        nanogui::shutdown();
        std::cout << "Debug: Nanogui shutdown completed" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}