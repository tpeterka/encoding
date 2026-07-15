include_directories         ($ENV{MFA_PATH}/examples)          # needed for block.hpp and opts.h

add_executable              (gridded_3d                             gridded_3d.cpp)

target_link_libraries       (gridded_3d                             ${libraries})

set_property                (TARGET gridded_3d                      PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install                     (TARGETS gridded_3d
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/mfa
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)
