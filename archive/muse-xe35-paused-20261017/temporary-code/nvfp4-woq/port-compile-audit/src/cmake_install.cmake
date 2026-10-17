# Install script for directory: /root/worktrees/sgl-kernel-xpu-up/pr/muse-nvfp4-woq-xe35/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE MODULE FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/common_ops.abi3.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/common_ops.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/common_ops.abi3.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/common_ops.abi3.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-BiasedTopK.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-BiasedTopK.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-BiasedTopK.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-BiasedTopK.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-ChunkedSgmvLoraExpandFwd.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-ChunkedSgmvLoraExpandFwd.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-ChunkedSgmvLoraExpandFwd.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-ChunkedSgmvLoraExpandFwd.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-ChunkedSgmvLoraShrinkFwd.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-ChunkedSgmvLoraShrinkFwd.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-ChunkedSgmvLoraShrinkFwd.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-ChunkedSgmvLoraShrinkFwd.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-CompressPlan.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-CompressPlan.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-CompressPlan.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-CompressPlan.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-Device.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-Device.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-Device.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-Device.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-EmbeddingLoraAFwd.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-EmbeddingLoraAFwd.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-EmbeddingLoraAFwd.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-EmbeddingLoraAFwd.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-FlashCompress128.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FlashCompress128.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FlashCompress128.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FlashCompress128.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-FlashCompress4.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FlashCompress4.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FlashCompress4.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FlashCompress4.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-FusedNormRopeStore.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedNormRopeStore.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedNormRopeStore.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedNormRopeStore.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-FusedQIndexerRopeHadamardQuant.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedQIndexerRopeHadamardQuant.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedQIndexerRopeHadamardQuant.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedQIndexerRopeHadamardQuant.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-FusedQKNormRope.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-FusedQKRope.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedQKRope.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedQKRope.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-FusedQKRope.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GateUpLoraBFwd.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GateUpLoraBFwd.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GateUpLoraBFwd.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GateUpLoraBFwd.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-HCHead.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCHead.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCHead.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCHead.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-HCPost.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCPost.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCPost.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCPost.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-HCPreFuse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCPreFuse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCPreFuse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCPreFuse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-HCPreGemmSqrSum.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCPreGemmSqrSum.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCPreGemmSqrSum.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCPreGemmSqrSum.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-HCSplitSinkhorn.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCSplitSinkhorn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCSplitSinkhorn.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HCSplitSinkhorn.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-Hadamard.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-Hadamard.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-Hadamard.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-Hadamard.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-HashTopK.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HashTopK.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HashTopK.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HashTopK.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-HiSparse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HiSparse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HiSparse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-HiSparse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-InklingAttnPrologue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-InklingAttnPrologue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-InklingAttnPrologue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-InklingAttnPrologue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-InklingSconv.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-InklingSconv.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-InklingSconv.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-InklingSconv.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-KVCache.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-KVCache.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-KVCache.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-KVCache.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-KVCacheIO.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-KVCacheIO.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-KVCacheIO.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-KVCacheIO.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-MHCFusedPostPre.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MHCFusedPostPre.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MHCFusedPostPre.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MHCFusedPostPre.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-MRope.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MRope.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MRope.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MRope.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-MinPSamplingFromProbs.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MinPSamplingFromProbs.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MinPSamplingFromProbs.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MinPSamplingFromProbs.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-MinimaxDecodeTopK.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MinimaxDecodeTopK.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MinimaxDecodeTopK.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MinimaxDecodeTopK.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-MoEAlign.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoEAlign.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoEAlign.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoEAlign.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-MoEPrepareInputs.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoEPrepareInputs.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoEPrepareInputs.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoEPrepareInputs.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-MoESum.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoESum.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoESum.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoESum.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-MoE_fused_gate.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoE_fused_gate.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoE_fused_gate.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoE_fused_gate.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-MoE_sum_reduce.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoE_sum_reduce.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoE_sum_reduce.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-MoE_sum_reduce.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-QKVLoraBFwd.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-QKVLoraBFwd.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-QKVLoraBFwd.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-QKVLoraBFwd.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-RMSNorm.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-RMSNorm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-RMSNorm.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-RMSNorm.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-Rope.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-Rope.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-Rope.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-Rope.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-SGEMMLoraAFwd.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SGEMMLoraAFwd.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SGEMMLoraAFwd.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SGEMMLoraAFwd.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-SGEMMLoraBFwd.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SGEMMLoraBFwd.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SGEMMLoraBFwd.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SGEMMLoraBFwd.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-ScaleResidualNormScaleShift.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-ScaleResidualNormScaleShift.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-ScaleResidualNormScaleShift.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-ScaleResidualNormScaleShift.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-SiluAndMulClamp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SiluAndMulClamp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SiluAndMulClamp.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SiluAndMulClamp.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-SpecBuildTree.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SpecBuildTree.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SpecBuildTree.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SpecBuildTree.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-SpecReconstructTree.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SpecReconstructTree.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SpecReconstructTree.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SpecReconstructTree.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-SpecVerifyTree.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SpecVerifyTree.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SpecVerifyTree.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SpecVerifyTree.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-SwigluAlphaLimit.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SwigluAlphaLimit.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SwigluAlphaLimit.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-SwigluAlphaLimit.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-TopKRenormProbs.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKRenormProbs.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKRenormProbs.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKRenormProbs.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-TopKSigMoid.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKSigMoid.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKSigMoid.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKSigMoid.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-TopKSoftMax.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKSoftMax.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKSoftMax.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKSoftMax.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-TopKTopPSamplingFromProbs.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKTopPSamplingFromProbs.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKTopPSamplingFromProbs.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKTopPSamplingFromProbs.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-TopKTransform.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKTransform.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKTransform.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopKTransform.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-TopPRenormProbs.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopPRenormProbs.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopPRenormProbs.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TopPRenormProbs.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-TripleOps.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TripleOps.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TripleOps.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-TripleOps.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-awq_dequantize.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-awq_dequantize.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-awq_dequantize.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-awq_dequantize.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-causal_conv1d.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-causal_conv1d.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-causal_conv1d.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-causal_conv1d.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-flash_attention.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-flash_attention.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-flash_attention.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-flash_attention.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-gdn_attention.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gdn_attention.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gdn_attention.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gdn_attention.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-chunk_gated_delta_rule.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunk_gated_delta_rule.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunk_gated_delta_rule.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunk_gated_delta_rule.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-gdn_attn_workspace.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gdn_attn_workspace.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gdn_attn_workspace.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gdn_attn_workspace.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-merge_states.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-merge_states.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-merge_states.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-merge_states.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_prefill.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-per_tensor_quant_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_tensor_quant_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_tensor_quant_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_tensor_quant_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-per_token_group_quant_8bit.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_group_quant_8bit.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_group_quant_8bit.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_group_quant_8bit.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-per_token_group_quant_8bit_v2.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_group_quant_8bit_v2.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_group_quant_8bit_v2.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_group_quant_8bit_v2.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-per_token_group_quant_fp4.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_group_quant_fp4.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_group_quant_fp4.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_group_quant_fp4.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-per_token_quant_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_quant_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_quant_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-per_token_quant_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-sgemm_lora_a_fwd_kernel_half_large.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_a_fwd_kernel_half_large.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_a_fwd_kernel_half_large.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_a_fwd_kernel_half_large.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-sgemm_lora_a_fwd_kernel_bf16_large.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_a_fwd_kernel_bf16_large.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_a_fwd_kernel_bf16_large.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_a_fwd_kernel_bf16_large.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_half_large.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_half_large.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_half_large.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_half_large.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_bf16_large.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_bf16_large.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_bf16_large.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_bf16_large.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-qkv_lora_b_fwd_kernel_half_tall.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-qkv_lora_b_fwd_kernel_half_tall.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-qkv_lora_b_fwd_kernel_half_tall.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-qkv_lora_b_fwd_kernel_half_tall.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-qkv_lora_b_fwd_kernel_bf16_tall.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-qkv_lora_b_fwd_kernel_bf16_tall.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-qkv_lora_b_fwd_kernel_bf16_tall.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-qkv_lora_b_fwd_kernel_bf16_tall.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-gate_up_lora_b_fwd_kernel_half_tall.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gate_up_lora_b_fwd_kernel_half_tall.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gate_up_lora_b_fwd_kernel_half_tall.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gate_up_lora_b_fwd_kernel_half_tall.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-gate_up_lora_b_fwd_kernel_bf16_tall.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gate_up_lora_b_fwd_kernel_bf16_tall.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gate_up_lora_b_fwd_kernel_bf16_tall.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-gate_up_lora_b_fwd_kernel_bf16_tall.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-chunked_sgmv_lora_shrink_fwd_kernel_half_small.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_shrink_fwd_kernel_half_small.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_shrink_fwd_kernel_half_small.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_shrink_fwd_kernel_half_small.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-chunked_sgmv_lora_shrink_fwd_kernel_bf16_small.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_shrink_fwd_kernel_bf16_small.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_shrink_fwd_kernel_bf16_small.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_shrink_fwd_kernel_bf16_small.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-chunked_sgmv_lora_expand_fwd_kernel_half_tall.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_expand_fwd_kernel_half_tall.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_expand_fwd_kernel_half_tall.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_expand_fwd_kernel_half_tall.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-chunked_sgmv_lora_expand_fwd_kernel_bf16_tall.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_expand_fwd_kernel_bf16_tall.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_expand_fwd_kernel_bf16_tall.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-chunked_sgmv_lora_expand_fwd_kernel_bf16_tall.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemm.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemm.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemm.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-blockwise_moe_mxfp4.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-blockwise_moe_mxfp4.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-blockwise_moe_mxfp4.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-blockwise_moe_mxfp4.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-blockwise_moe_mxfp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-blockwise_moe_mxfp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-blockwise_moe_mxfp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-blockwise_moe_mxfp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-blockwise_moe_nvfp4_woq_port_payload.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-blockwise_moe_nvfp4_woq_port_payload.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-blockwise_moe_nvfp4_woq_port_payload.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-blockwise_moe_nvfp4_woq_port_payload.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-dsv3_fused_a_gemm.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-dsv3_fused_a_gemm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-dsv3_fused_a_gemm.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-dsv3_fused_a_gemm.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-dsv3_router_gemm.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-dsv3_router_gemm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-dsv3_router_gemm.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-dsv3_router_gemm.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-fp8_blockwise_gemm.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-fp8_blockwise_gemm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-fp8_blockwise_gemm.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-fp8_blockwise_gemm.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-fp8_scaled_mm.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-fp8_scaled_mm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-fp8_scaled_mm.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-fp8_scaled_mm.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-nvfp4_woq_port_payload.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-nvfp4_woq_port_payload.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-nvfp4_woq_port_payload.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-nvfp4_woq_port_payload.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-nvfp4_woq_scale_layout.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-nvfp4_woq_scale_layout.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-nvfp4_woq_scale_layout.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-nvfp4_woq_scale_layout.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_64_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_64_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_96_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_96_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_128_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_128_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_192_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_192_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_256_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_256_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_1_512_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_1_512_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_72_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_72_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_72_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_72_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_72_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_72_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_72_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_72_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_80_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_80_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_80_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_80_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_80_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_80_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_80_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_80_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_96_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_96_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_96_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_96_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_96_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_96_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_96_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_96_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_192_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_192_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_192_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_192_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_192_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_192_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_192_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_192_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_256_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_256_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_256_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_256_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_256_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_256_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_256_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_256_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_512_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_512_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_512_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_512_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_512_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_512_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_512_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_1_512_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_64_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_64_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_96_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_96_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_128_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_128_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_192_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_192_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_256_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_256_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_2_512_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_2_512_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_72_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_72_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_72_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_72_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_72_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_72_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_72_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_72_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_80_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_80_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_80_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_80_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_80_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_80_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_80_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_80_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_96_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_96_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_96_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_96_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_96_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_96_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_96_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_96_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_192_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_192_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_192_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_192_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_192_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_192_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_192_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_192_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_256_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_256_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_256_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_256_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_256_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_256_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_256_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_256_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_512_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_512_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_512_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_512_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_512_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_512_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_512_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_2_512_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_64_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_64_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_96_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_96_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_128_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_128_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_192_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_192_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_256_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_256_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_4_512_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_4_512_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_72_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_72_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_72_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_72_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_72_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_72_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_72_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_72_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_80_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_80_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_80_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_80_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_80_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_80_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_80_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_80_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_96_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_96_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_96_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_96_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_96_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_96_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_96_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_96_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_192_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_192_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_192_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_192_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_192_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_192_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_192_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_192_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_256_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_256_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_256_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_256_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_256_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_256_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_256_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_256_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_512_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_512_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_512_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_512_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_512_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_512_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_512_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_4_512_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_64_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_64_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_96_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_96_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_128_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_128_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_192_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_192_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_256_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_256_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_8_512_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_8_512_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_72_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_72_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_72_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_72_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_72_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_72_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_72_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_72_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_80_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_80_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_80_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_80_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_80_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_80_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_80_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_80_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_96_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_96_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_96_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_96_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_96_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_96_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_96_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_96_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_192_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_192_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_192_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_192_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_192_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_192_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_192_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_192_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_256_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_256_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_256_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_256_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_256_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_256_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_256_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_256_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_512_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_512_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_512_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_512_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_512_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_512_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_512_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_8_512_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_64_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_64_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_96_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_96_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_128_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_128_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_192_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_192_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_256_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_256_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_page_16_512_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_split_decode_page_16_512_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_72_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_72_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_72_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_72_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_72_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_72_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_72_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_72_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_80_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_80_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_80_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_80_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_80_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_80_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_80_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_80_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_96_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_96_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_96_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_96_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_96_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_96_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_96_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_96_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_192_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_192_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_192_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_192_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_192_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_192_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_192_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_192_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_256_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_256_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_256_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_256_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_256_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_256_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_256_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_256_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_512_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_512_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_512_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_512_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_512_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_512_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_512_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_decode_nopage_16_512_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_96_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_192_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_256_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_bf16_fp8.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_bf16_fp8.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_bf16_fp8.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_bf16_fp8.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_page_512_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_64_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_64_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_64_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_64_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_64_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_64_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_64_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_64_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_72_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_72_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_72_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_72_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_72_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_72_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_72_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_72_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_80_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_80_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_80_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_80_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_80_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_80_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_80_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_80_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_96_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_96_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_96_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_96_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_96_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_96_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_96_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_96_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_128_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_128_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_128_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_128_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_128_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_128_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_128_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_128_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_192_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_192_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_192_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_192_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_192_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_192_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_192_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_192_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_256_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_256_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_256_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_256_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_256_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_256_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_256_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_256_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_512_bf16_bf16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_512_bf16_bf16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_512_bf16_bf16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_512_bf16_bf16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_512_fp16_fp16.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_512_fp16_fp16.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_512_fp16_fp16.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-xe_fmha_fwd_prefill_nopage_512_fp16_fp16.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_half_16_q1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_16_q1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_16_q1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_16_q1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_half_16_q8_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_16_q8_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_16_q8_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_16_q8_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_half_32_q1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_32_q1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_32_q1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_32_q1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_half_32_q8_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_32_q8_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_32_q8_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_32_q8_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_half_64_q1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_64_q1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_64_q1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_64_q1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_half_64_q8_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_64_q8_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_64_q8_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_64_q8_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_half_128_q1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_128_q1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_128_q1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_128_q1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_half_128_q8_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_128_q8_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_128_q8_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_half_128_q8_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_bf16_16_q1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_16_q1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_16_q1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_16_q1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_bf16_16_q8_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_16_q8_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_16_q8_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_16_q8_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_bf16_32_q1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_32_q1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_32_q1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_32_q1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_bf16_32_q8_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_32_q8_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_32_q8_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_32_q8_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_bf16_64_q1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_64_q1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_64_q1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_64_q1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_bf16_64_q8_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_64_q8_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_64_q8_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_64_q8_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_bf16_128_q1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_128_q1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_128_q1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_128_q1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_decode_kernel_bf16_128_q8_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_128_q8_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_128_q8_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_decode_kernel_bf16_128_q8_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_prefill_kernel_half_16_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_16_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_16_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_16_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_prefill_kernel_half_32_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_32_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_32_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_32_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_prefill_kernel_half_64_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_64_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_64_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_64_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_prefill_kernel_half_128_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_128_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_128_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_half_128_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_prefill_kernel_bf16_16_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_16_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_16_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_16_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_prefill_kernel_bf16_32_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_32_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_32_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_32_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_prefill_kernel_bf16_64_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_64_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_64_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_64_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_prefill_kernel_bf16_128_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_128_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_128_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_prefill_kernel_bf16_128_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_8_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_8_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_8_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_8_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_8_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_8_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_8_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_8_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_16_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_16_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_16_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_16_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_16_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_16_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_16_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_16_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_32_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_32_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_32_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_32_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_32_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_32_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_32_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_32_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_64_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_64_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_64_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_64_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_64_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_64_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_64_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_512_64_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_8_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_8_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_8_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_8_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_8_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_8_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_8_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_8_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_16_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_16_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_16_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_16_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_16_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_16_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_16_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_16_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_32_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_32_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_32_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_32_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_32_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_32_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_32_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_32_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_64_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_64_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_64_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_64_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_64_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_64_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_64_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_decode_2stage_kernel_bf16_576_64_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_8_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_8_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_8_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_8_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_8_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_8_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_8_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_8_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_32_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_32_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_32_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_32_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_32_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_32_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_32_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_32_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_64_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_64_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_64_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_64_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_64_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_64_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_64_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_512_64_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_8_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_8_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_8_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_8_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_8_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_8_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_8_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_8_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_32_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_32_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_32_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_32_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_32_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_32_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_32_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_32_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_64_0_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_64_0_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_64_0_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_64_0_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_64_1_xe35.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_64_1_xe35.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_64_1_xe35.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-mla_sparse_prefill_2stage_kernel_bf16_576_64_1_xe35.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a0_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a0_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a0_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a0_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a0_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a0_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a0_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a0_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a0_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a0_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a0_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a0_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a0_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a0_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a0_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a0_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a0_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a0_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a0_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a0_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a0_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a1_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a1_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a1_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a1_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a1_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a1_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a1_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a1_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a1_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a1_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a1_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a1_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a1_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a1_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a1_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a1_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a1_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a1_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a1_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a1_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a1_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ftrue_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ftrue_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ftrue_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ftrue_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a2_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a2_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a2_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a2_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a2_ffalse_btrue.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a2_ffalse_btrue.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a2_ffalse_btrue.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a2_ffalse_btrue.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a2_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a2_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a2_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a2_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a2_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a2_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a2_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a2_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a2_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a2_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a2_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a2_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a2_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a3_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a3_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a3_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a3_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a3_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a3_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a3_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a3_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a3_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a3_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a3_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a3_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a3_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a3_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a3_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__8__64__32_a3_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a3_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a3_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a3_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__16__64__32_a3_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a3_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a3_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a3_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__32__64__32_a3_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a3_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a3_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a3_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a3_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a3_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a3_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a3_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a3_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a3_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a3_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a3_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__64__32_a3_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a3_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a3_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a3_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__128__128__32_a3_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a3_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a3_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a3_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__64__32_a3_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a3_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a3_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a3_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmXe35_inst__256__256__32_a3_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a0_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a0_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a0_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a0_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a0_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a0_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a0_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a0_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__64__32_a0_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__64__32_a0_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__64__32_a0_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__64__32_a0_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__128__32_a0_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__128__32_a0_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__128__32_a0_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__128__32_a0_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__64__32_a0_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__64__32_a0_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__64__32_a0_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__64__32_a0_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__256__32_a0_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__256__32_a0_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__256__32_a0_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__256__32_a0_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a4_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a4_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a4_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a4_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a4_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a4_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a4_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__8__64__32_a4_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__64__32_a4_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__64__32_a4_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__64__32_a4_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__64__32_a4_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__128__32_a4_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__128__32_a4_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__128__32_a4_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__128__128__32_a4_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__64__32_a4_ftrue_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__64__32_a4_ftrue_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__64__32_a4_ftrue_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__64__32_a4_ftrue_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sgl_kernel" TYPE SHARED_LIBRARY FILES "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__256__32_a4_ffalse_bfalse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__256__32_a4_ffalse_bfalse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__256__32_a4_ffalse_bfalse.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/sgl_kernel/libsgl-ops-sycl-GroupGemmMxfp4W4A16_inst_xe35__256__256__32_a4_ffalse_bfalse.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/port-compile-audit/src/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
