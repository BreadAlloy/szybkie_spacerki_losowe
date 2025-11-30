#pragma once

#pragma warning(push, 0)
// Nie chce widzieæ losowych warningow z grafiki komputerowaej

#include "src/SOIL/SOIL.h"
#include "glew.h"

#include <GLFW/glfw3.h>
#include "glm.hpp"
#include "ext.hpp"

#pragma warning(pop)

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "implot.h"

#include "pomocne_funkcje.h"

struct grafika {
    unsigned char* data = nullptr;
    GLuint texture = 0;
    uint32_t width = 0;
    uint32_t height = 0;

    grafika(uint32_t Width, uint32_t Height) {
        width = Width;
        height = Height;
        data = (unsigned char*)malloc(Width * Height * sizeof(unsigned char) * 4);
    }

    grafika(const char* path) {
        LoadTextureFromFile(path);
    }

    ~grafika() {
        if (data != nullptr) {
            free(data);
        }
        if (texture != 0) {
            glDeleteTextures(1, &texture);
        }
    }

    void display_image(ImGuiIO& io, float skala_obrazu = 1.0f) const {
        float my_tex_w = (float)width * skala_obrazu;
        float my_tex_h = (float)height * skala_obrazu;

        static bool use_text_color_for_tint = false;
        ImGui::Text("%.0fx%.0f", my_tex_w, my_tex_h);
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 uv_min = ImVec2(0.0f, 0.0f);                 // Top-left
        ImVec2 uv_max = ImVec2(1.0f, 1.0f);                 // Lower-right
        ImVec4 tint_col = use_text_color_for_tint ? ImGui::GetStyleColorVec4(ImGuiCol_Text) : ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // No tint
        ImVec4 border_col = ImGui::GetStyleColorVec4(ImGuiCol_Border);

        ImGui::Image((ImTextureID)(intptr_t)texture, ImVec2(width * skala_obrazu, height * skala_obrazu));

        if (ImGui::BeginItemTooltip())
        {
            float region_sz = 32.0f;
            float region_x = io.MousePos.x - pos.x - region_sz * 0.5f;
            float region_y = io.MousePos.y - pos.y - region_sz * 0.5f;
            float zoom = 4.0f;
            if (region_x < 0.0f) { region_x = 0.0f; }
            else if (region_x > my_tex_w - region_sz) { region_x = my_tex_w - region_sz; }
            if (region_y < 0.0f) { region_y = 0.0f; }
            else if (region_y > my_tex_h - region_sz) { region_y = my_tex_h - region_sz; }
            ImGui::Text("Min: (%.2f, %.2f)", region_x, region_y);
            ImGui::Text("Max: (%.2f, %.2f)", region_x + region_sz, region_y + region_sz);
            ImVec2 uv0 = ImVec2((region_x) / my_tex_w, (region_y) / my_tex_h);
            ImVec2 uv1 = ImVec2((region_x + region_sz) / my_tex_w, (region_y + region_sz) / my_tex_h);
            ImGui::Image((ImTextureID)(intptr_t)texture, ImVec2(region_sz * zoom * skala_obrazu, region_sz * zoom * skala_obrazu), uv0, uv1, tint_col, border_col);
            ImGui::EndTooltip();
        }

    }

    bool LoadTextureFromMemory()
    {
        int image_width = width;
        int image_height = height;
        unsigned char* image_data = data;
        if (image_data == nullptr) {
            ASSERT_Z_ERROR_MSG(false, "Nie ma nic w danych grafiki\n");
            return false;
        }
        // Create a OpenGL texture identifier
        GLuint image_texture;
        glGenTextures(1, &image_texture);
        glBindTexture(GL_TEXTURE_2D, image_texture);

        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Upload pixels into texture
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);

        texture = image_texture;

        ASSERT_Z_ERROR_MSG(glIsTexture(texture), "Cos nie tak z tekstura\n");

        free(data);
        data = nullptr;

        return true;
    }

    void LoadTextureFromFile(const char* path) {
        data = SOIL_load_image(path, (int*)&width, (int*)&height, 0, SOIL_LOAD_RGBA);
        ASSERT_Z_ERROR_MSG(data != nullptr, "Nie ma nic w danych grafiki\n");
        LoadTextureFromMemory();
    }
};

#include "spacer_losowy.h"

template <typename towar, typename transformata>
__host__ grafika* grafika_P_dla_kraty_2D(spacer_losowy<towar, transformata>& spacer, spacer::dane_iteracji<towar>& iteracja, uint32_t width, uint32_t height, double* suma_ptr = nullptr);

template <typename towar, typename transformata>
__host__ void plot_grafike_dla_kraty_2D(spacer_losowy<towar, transformata>& spacer, uint64_t pokazywana_grafika, graf& przestrzen, grafika* G, uint32_t width, uint32_t height, float skala_obrazu);

