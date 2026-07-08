# ---=== CONFIG ===---

cmake_minimum_required(VERSION 3.3.0) # Bo to mam

project("SzybkieSpacerkiLosowe"  LANGUAGES CXX CUDA C)
# C dla SOIL

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VERBOSE true)

# set(CMAKE_C_COMPILER "C:\\msys64\\ucrt64\\bin\\gcc.exe ")
# set(CMAKE_CXX_COMPILER "C:\\msys64\\ucrt64\\bin\\g++.exe ")

# cmake_policy(SET CMP0141 NEW)
# set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_DEBUG_FLAGS})  # enable cuda-gdb

# ---=== INSTALL DEPENDENCIES ===---

include(FetchContent)

# Setup OpenGL
cmake_policy(SET CMP0072 NEW) # Pefer GLVND over legacy GL libraries
find_package(OpenGL REQUIRED)

# Setup GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY "https://github.com/glfw/glfw"
    GIT_TAG "3.3.8"
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

# Setup ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY "https://github.com/ocornut/imgui"
    GIT_TAG "v1.92.7"
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(imgui)
set(IMGUI_SOURCE
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
add_library(imgui STATIC ${IMGUI_SOURCE})
target_include_directories(imgui PUBLIC "${imgui_SOURCE_DIR};${imgui_SOURCE_DIR}/backends/")
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# Setup ImPlot
FetchContent_Declare(
    implot
    GIT_REPOSITORY "https://github.com/epezent/implot"
    GIT_TAG "v0.17"
    GIT_PROGRESS TRUE
)

# To by wypadalo wykonac tylko raz przy pobraniu
message("Adding to imconfig.h")
file(APPEND ${imgui_SOURCE_DIR}/imconfig.h
"
#undef ImDrawIdx
// opcja 1 fixa z readme implot
#define ImDrawIdx unsigned int"
)

FetchContent_MakeAvailable(implot)
set(IMPLOT_SOURCE
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_demo.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)


add_library(implot STATIC ${IMPLOT_SOURCE})
target_include_directories(implot PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot PUBLIC imgui)

# Setup ImPlot3D
FetchContent_Declare(
    implot3d
    GIT_REPOSITORY "https://github.com/brenocq/implot3d"
    GIT_TAG "v0.4"
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(implot3d)

set(IMPLOT3D_SOURCE
    ${implot3d_SOURCE_DIR}/implot3d.cpp
    ${implot3d_SOURCE_DIR}/implot3d_demo.cpp
    ${implot3d_SOURCE_DIR}/implot3d_items.cpp
    ${implot3d_SOURCE_DIR}/implot3d_meshes.cpp
)

add_library(implot3d STATIC ${IMPLOT3D_SOURCE})
target_include_directories(implot3d PUBLIC ${implot3d_SOURCE_DIR})
target_link_libraries(implot3d PUBLIC imgui)
if(IMPLOT3D_STRICT_WARNINGS)
    if(MSVC)
        target_compile_options(implot3d PRIVATE /W4 /WX)
    else()
        target_compile_options(implot3d PRIVATE -Wall -Wextra -Werror)
    endif()
endif()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)



# ---=== PROJECT COMPILE ===---

find_package( CUDA  REQUIRED )

set(spacer_GLOBAL_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/Common # wszystko ma cude
    szybki_spacer_lib
)

set(imgui_INCLUDE_DIRS
    ${glfw_SOURCE_DIR}/include
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${implot_SOURCE_DIR}
    ${implot3d_SOURCE_DIR}
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_library(szybki_spacer_lib STATIC
    szybki_spacer_lib/graf.cpp
    szybki_spacer_lib/graf.h 

    szybki_spacer_lib/losowe.cpp
    szybki_spacer_lib/losowe.h 

    szybki_spacer_lib/rozniczka.cpp 
    szybki_spacer_lib/rozniczka.cu 
    szybki_spacer_lib/rozniczka.cuh

    szybki_spacer_lib/spacer_losowy.cpp 
    szybki_spacer_lib/spacer_losowy.cu 
    szybki_spacer_lib/spacer_losowy.h

    szybki_spacer_lib/transformata_fouriera.cu
    szybki_spacer_lib/transformata_fouriera.cuh

    szybki_spacer_lib/transformaty.cu 
    szybki_spacer_lib/transformaty.h
)

target_include_directories ( szybki_spacer_lib PUBLIC 
    ${spacer_GLOBAL_INCLUDE_DIRS}
)

set_target_properties( szybki_spacer_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON )


add_library(SOIL STATIC
    SOIL/image_DXT.c
    SOIL/image_helper.c
    SOIL/SOIL.c
    SOIL/stb_image_aug.c
)


add_library(szybki_spacer_pomoce STATIC
    szybki_spacer_pomoce/alg_liniowa.cpp 
    szybki_spacer_pomoce/alg_liniowa.h

    szybki_spacer_pomoce/definicje_typowych_macierzy.cpp 
    szybki_spacer_pomoce/definicje_typowych_macierzy.h 

    szybki_spacer_pomoce/grafika.cpp 
    szybki_spacer_pomoce/grafika.h 

    szybki_spacer_pomoce/imgui_spacer.cpp 
    szybki_spacer_pomoce/imgui_spacer.h 
)

target_include_directories( szybki_spacer_pomoce PUBLIC
    ${imgui_INCLUDE_DIRS}
    ${spacer_GLOBAL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/SOIL
)

set_target_properties( szybki_spacer_pomoce PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
target_link_libraries( szybki_spacer_pomoce PUBLIC SOIL)


add_executable(szybki_spacerek szybkie_spacerki_losowe.cpp)

target_link_libraries(szybki_spacerek PRIVATE implot implot3d)
target_link_libraries(szybki_spacerek PUBLIC
    szybki_spacer_lib szybki_spacer_pomoce
)

target_include_directories(szybki_spacerek PUBLIC
    ${imgui_INCLUDE_DIRS}
    ${spacer_GLOBAL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/eksperymenty
    ${CMAKE_CURRENT_SOURCE_DIR}/szybki_spacer_pomoce
)

# Silence OpenGL deprecation warnings on macOS
if(APPLE)
    target_compile_definitions(SzybkieSpacerkiLosowe PRIVATE GL_SILENCE_DEPRECATION)
endif()



# ---=== UNIT TESTS AND BENCHMARKS ===---
enable_testing()

set( tests_INCLUDE_DIRS
    ${imgui_INCLUDE_DIRS}
    ${spacer_GLOBAL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/eksperymenty
    ${CMAKE_CURRENT_SOURCE_DIR}/szybki_spacer_pomoce
)



add_executable(
  liczba_iteracji_skalowanie_bench
  szybki_spacer_tests_i_bench/liczba_iteracji_skalowanie_bench.cc
)

target_include_directories(liczba_iteracji_skalowanie_bench PUBLIC
    ${tests_INCLUDE_DIRS}
)

target_link_libraries(
  liczba_iteracji_skalowanie_bench
  szybki_spacer_lib
  szybki_spacer_pomoce
  implot implot3d
  GTest::gtest_main
)



add_executable(
  cuda_parametry_bench
  szybki_spacer_tests_i_bench/cuda_parametry_bench.cc
)

target_include_directories(cuda_parametry_bench PUBLIC
    ${tests_INCLUDE_DIRS}
)

target_link_libraries(
  cuda_parametry_bench
  szybki_spacer_lib
  szybki_spacer_pomoce
  implot implot3d
  GTest::gtest_main
)



add_executable(
  rozmiar_grafu_bench
  szybki_spacer_tests_i_bench/rozmiar_grafu_bench.cc
)

target_include_directories(rozmiar_grafu_bench PUBLIC
    ${tests_INCLUDE_DIRS}
)

target_link_libraries(
  rozmiar_grafu_bench
  szybki_spacer_lib
  szybki_spacer_pomoce
  implot implot3d
  GTest::gtest_main
)



add_executable(
  czy_cuda_taka_jak_cpu_test
  szybki_spacer_tests_i_bench/czy_cuda_taka_jak_cpu_test.cc
)

target_include_directories(czy_cuda_taka_jak_cpu_test PUBLIC
    ${tests_INCLUDE_DIRS}
)

target_link_libraries(
  czy_cuda_taka_jak_cpu_test
  szybki_spacer_lib
  szybki_spacer_pomoce
  implot implot3d
  GTest::gtest_main
)



include(GoogleTest)
gtest_discover_tests(liczba_iteracji_skalowanie_bench)
gtest_discover_tests(cuda_parametry_bench)
gtest_discover_tests(rozmiar_grafu_bench)

gtest_discover_tests(czy_cuda_taka_jak_cpu_test)




# So that clangd sees things installed by cmake
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
#     file(COPY ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/)
# endif()