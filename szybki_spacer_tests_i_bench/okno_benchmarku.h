#pragma once

#include "imgui_i_grafika_setup.h"

struct okno_benchmarku {
    GLFWwindow* window = nullptr;
    ImGuiIO* io = nullptr;

    okno_benchmarku(std::string nazwa_okna_glownego) {
        io = new ImGuiIO;
        int ret = imgui_i_grafika_setup(window, io, nazwa_okna_glownego);
        ASSERT_Z_ERROR_MSG(ret == 0, "Cos nie tak z oknem\n");
    }

    bool tick_start() {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        return !glfwWindowShouldClose(window);
    }

    void tick_finish() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // czesciowo z grafiki

        ImGui::Render();
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }


    ~okno_benchmarku() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot3D::DestroyContext();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        delete io;

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};


struct rezultat_benchu_2 {

    // { parametr, czas[ms] }

    std::vector<float> parametry;
    std::vector<float> czasy;

    std::string nazwa;

    rezultat_benchu_2(std::string nazwa)
        : nazwa(nazwa) {
    }

    void pokaz_dane() const {
        ImPlot::PlotLine(nazwa.c_str(),
            parametry.data(), czasy.data(),
            parametry.size());

        ImPlot::PlotScatter(nazwa.c_str(),
            parametry.data(), czasy.data(),
            parametry.size());
    }

    void zaloguj(float parametr, float czas) {
        parametry.push_back(parametr);
        czasy.push_back(czas);
    }
};

struct rezultat_benchu_3 {

    // { parametr_1, parametr_2, czas[ms] }

    std::vector<float> parametry_1;
    std::vector<float> parametry_2;
    std::vector<float> czasy;

    std::string nazwa;

    rezultat_benchu_3(std::string nazwa)
        : nazwa(nazwa) {
    }

    void pokaz_dane(uint32_t parametry_1_count, uint32_t parametry_2_count, ImPlot3DSpec& spec) const {       
        ImPlot3D::PlotSurface(nazwa.c_str(),
            parametry_1.data(), parametry_2.data(), czasy.data(),
            parametry_1_count, parametry_2_count, 0.0, 0.0, spec);
    }

    void zaloguj(float parametr_1, float parametr_2, float czas) {
        parametry_1.push_back(parametr_1);
        parametry_2.push_back(parametr_2);
        czasy.push_back(czas);
    }
};
