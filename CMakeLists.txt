cmake_minimum_required(VERSION 3.15)
project(Oefeningen CXX)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output map voor executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Zoek alle main.cpp bestanden
file(GLOB_RECURSE EXERCISES RELATIVE ${CMAKE_SOURCE_DIR} src/*/main.cpp)

add_compile_options(-O0)

foreach(MAIN_FILE ${EXERCISES})
    # Haal de foldernaam (bv. oefening1)
    get_filename_component(EXERCISE_DIR ${MAIN_FILE} DIRECTORY)
    get_filename_component(EXERCISE_NAME ${EXERCISE_DIR} NAME)

    add_executable(${EXERCISE_NAME} ${MAIN_FILE})
    target_include_directories(${EXERCISE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

    if(${EXERCISE_NAME} STREQUAL "oefening11" OR ${EXERCISE_NAME} STREQUAL "oefening12")
        target_link_libraries(${EXERCISE_NAME} PRIVATE MPI::MPI_CXX)
    endif()

    # Kopieer de resources-map naast de executable
    add_custom_command(TARGET ${EXERCISE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:${EXERCISE_NAME}>/resources
    )
endforeach()
