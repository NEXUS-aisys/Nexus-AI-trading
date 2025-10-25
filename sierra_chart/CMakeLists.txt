cmake_minimum_required(VERSION 3.16)
project(NexusAI_Lite)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find gRPC and Protobuf
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# Generate protobuf files
set(PROTO_FILES "../proto/nexus_trading.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_FILES})

# Create source files list
set(SOURCES
    NexusAI_Lite.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

set(HEADERS
    ${PROTO_HDRS}
    ${GRPC_HDRS}
)

# Create shared library
add_library(NexusAI_Lite SHARED ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(NexusAI_Lite PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../proto
    ${Protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(NexusAI_Lite PRIVATE
    ${Protobuf_LIBRARIES}
    gRPC::grpc++
)

# Compiler definitions
target_compile_definitions(NexusAI_Lite PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _WIN32_WINNT=0x0A00
)

# Set output directory
set_target_properties(NexusAI_Lite PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Copy to Sierra Chart after build
add_custom_command(TARGET NexusAI_Lite POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:NexusAI_Lite> "C:/SierraChart/DataFiles/NexusAI_Lite.dll"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:NexusAI_Lite> "${CMAKE_CURRENT_SOURCE_DIR}/NexusAI_Lite.dll"
    COMMENT "Copying DLL to Sierra Chart and local folder"
)
