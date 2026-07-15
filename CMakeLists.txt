project                     (encoding)
cmake_minimum_required      (VERSION 3.12)

option                      (mfa_thread     "tbb, kokkos, sycl, or serial (default serial)"                                 serial  )
option                      (s3d_infile     "Path/filename of s3d dataset (for unit testing)"                                       )
option                      (mfa_build_examples "Build MFA examples (default true)"                                         ON      )
option                      (mfa_build_tests "Build MFA tests (default true)"                                               ON      )
option                      (mfa_python     "Build Python bindings, true/false (default false)"                             OFF     )
option                      (eigen_thread   "Use OpenMP for Eigen threading, true/false (default false)"                    OFF     )
option                      (KOKKOS_INCLUDE_DIRS    "Path of include directories for Kokkos (optional)"                             )
option                      (KOKKOS_LIBRARY "Kokkos library (optional)"                                                             )
option                      (OPENMP_LIBRARY "OpenMP library if not found automatically (optional)"                                  )

# C++17
set                         (CMAKE_CXX_STANDARD 17)
set                         (CMAKE_CXX_STANDARD_REQUIRED ON)

set                         (CMAKE_POSITION_INDEPENDENT_CODE ON)

# ignore variadic args warning (no other good solution)
set                         (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-varargs")

# suppress null pointer warnings in GNU
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options     (-Wno-nonnull)
endif()

# Default to Release
if                          (NOT CMAKE_BUILD_TYPE)
    set                     (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property            (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif                       (NOT CMAKE_BUILD_TYPE)

# MPI
if                          (NOT wrapped_mpi)
    find_package            (MPI REQUIRED)
    set                     (libraries MPI::MPI_C)
endif                       ()

# Threads
find_package                (Threads)

# MFA
if                          (NOT DEFINED MFA_PATH AND DEFINED ENV{MFA_PATH})
    set                     (MFA_PATH $ENV{MFA_PATH})
endif                       ()
add_definitions             (-DMFA_NO_WEIGHTS)

# MFA threading model
if                          (mfa_thread MATCHES "tbb")
    message                 ("Using TBB threading")
    add_definitions         (-DMFA_TBB)
    find_package            (TBB QUIET)
    find_path               (TBB_INCLUDE_DIR        tbb/tbb.h)
    find_library            (TBB_LIBRARY NAMES      tbb)
    if                      (TBB_INCLUDE_DIR AND TBB_LIBRARY)
        include_directories ("${TBB_INCLUDE_DIR}")
        set                 (libraries ${libraries} ${TBB_LIBRARY})
        message             ("Found TBB in ${TBB_INCLUDE_DIR} and in ${TBB_LIBRARY}")
    else                    (TBB_INCLUDE_DIR AND TBB_LIBRARY)
        message             (FATAL_ERROR "Could not find TBB")
    endif                   ()
else                        ()
    message                 ("Using no threading (serial)")
    add_definitions         (-DMFA_SERIAL)
endif                       ()

# Eigen threading (requires OpenMP to be available)
if                          (eigen_thread)
    find_package            (OpenMP)
    message                 ("Using OpenMP for Eigen")
    add_definitions         (-DEIGEN_OPENMP)
    if                      (APPLE)
        set                 (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
        if                  (${OPENMP_CXX_FOUND})
            set             (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
        elseif              (DEFINED OPENMP_INCLUDE_DIRS AND DEFINED OPENMP_LIBRARY)
            include_directories (${OPENMP_INCLUDE_DIRS})
            set             (libraries ${libraries} ${OPENMP_LIBRARY})
            message         ("Using user-provided OpenMP headers and library: ${OPENMP_INCLUDE_DIRS} ${OPENMP_LIBRARY}")
        else                ()
            message         ("Error: cannot find OpenMP and OPENMP_INCLUDE_DIRS or OPENMP_LIBRARY is not set")
            message         (FATAL_ERROR "Either provide OPENMP_INCLUDE_DIRS and OPENMP_LIBRARY explicitly or disable eigen_thread")
        endif               ()
    else                    ()
        set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif                   ()
endif                       ()

# Include dirs
set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (${MFA_PATH}/include
                            ${MFA_PATH}/include/cppoptlib/include
                            ${MFA_PATH}/include/diy/include
                            ${MFA_PATH}/include/eigen3
                            SYSTEM ${MPI_C_INCLUDE_PATH})

# Libraries
set                         (libraries
                             ${libraries}
                             ${CMAKE_DL_LIBS}
                             ${CMAKE_THREAD_LIBS_INIT} )

# Subdirectories
add_subdirectory            (mfa)

