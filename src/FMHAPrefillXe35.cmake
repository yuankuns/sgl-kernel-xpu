set(FMHA_XE35_HEAD_DIMS 64 128)

set(FMHA_XE35_PREFILL_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/flash_attentionXe35_prefill_kernel.cpp.in")
set(FMHA_XE35_PREFILL_NOPAGE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/flash_attentionXe35_prefill_nopage_kernel.cpp.in")

foreach(HEAD_DIM ${FMHA_XE35_HEAD_DIMS})
    set(GENERATED_PREFILL_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/flash_attentionXe35_prefill_${HEAD_DIM}.cpp")
    configure_file(${FMHA_XE35_PREFILL_TEMPLATE} ${GENERATED_PREFILL_FILE} @ONLY)
    list(APPEND device_cpp_xe35 ${GENERATED_PREFILL_FILE})

    # Non-paged (contiguous ragged KV) variant compiled as its own translation
    # unit so each object only instantiates a single KV-layout to bound memory.
    set(GENERATED_PREFILL_NOPAGE_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/flash_attentionXe35_prefill_nopage_${HEAD_DIM}.cpp")
    configure_file(${FMHA_XE35_PREFILL_NOPAGE_TEMPLATE} ${GENERATED_PREFILL_NOPAGE_FILE} @ONLY)
    list(APPEND device_cpp_xe35 ${GENERATED_PREFILL_NOPAGE_FILE})
endforeach()
