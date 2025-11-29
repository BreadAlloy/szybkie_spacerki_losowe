#pragma warning(disable:4996)

#include "imgui_i_grafika_setup.h"

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <string>

#include "graf.h"
#include "spacer_losowy.h"

#include "transformaty_wyspecializowane.h"

#include "pomocne_funkcje.h"

#include "spacer_2D_wizualizacja.h"

#include "testy_czy_dziala.h"

// Main code
int main(int ccc, char** aaa)
{
    //assert(false);
    /*        Test czy CUDA dzia³a          */
    //int mmmmain(int argc, char** argv);
    //mmmmain(ccc, aaa);
    //printf("dddddddddd%s%d%s\n");
    //std::cout << "Test%s%s%s\n";
    /*        Test czy CUDA dzia³a          */

    static_assert(sizeof(void*) == 8, "Powinno byc 8 bytow w pointerze");

    GLFWwindow* window; ImGuiIO* io = new ImGuiIO;
    int ret = imgui_i_grafika_setup(window, io);
    if(ret != 0) return ret;
    //ImGuiIO& io = temp;

    test_spaceru_klasyczny_dyskretny TSKD;
    test_spaceru_kwantowy_dyskretny TSQD;

    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();   

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (show_demo_window){
            ImGui::ShowDemoWindow(&show_demo_window);
            ImPlot::ShowDemoWindow(&show_demo_window);
        }

        TSKD.pokaz_okno(*io);
        TSQD.pokaz_okno(*io);

        //processInput(window); // z grafiki
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // czesciowo z grafiki
        //renderScene(window); // z grafiki

        ImGui::Render();
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    delete io;
    //shutdown(window); // z grafiki

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
