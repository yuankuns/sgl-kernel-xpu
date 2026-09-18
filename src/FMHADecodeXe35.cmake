set(FMHA_XE35_HEAD_DIMS 64 128)
set(FMHA_XE35_ELEM_TAGS bf16 fp8_e4m3 fp8_e5m2)

set(FMHA_XE35_DECODE_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/flash_attentionXe35_decode_kernel.cpp.in")
set(FMHA_XE35_DECODE_NOPAGE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/flash_attentionXe35_decode_nopage_kernel.cpp.in")

foreach(HEAD_DIM ${FMHA_XE35_HEAD_DIMS})
    foreach(ELEM_TAG ${FMHA_XE35_ELEM_TAGS})
        if(ELEM_TAG STREQUAL "bf16")
            set(ELEM_TYPE "cutlass::bfloat16_t")
        elseif(ELEM_TAG STREQUAL "fp8_e4m3")
            set(ELEM_TYPE "cutlass::float_e4m3_t")
        else()
            set(ELEM_TYPE "cutlass::float_e5m2_t")
        endif()
        set(GENERATED_DECODE_FILE
            "${CMAKE_CURRENT_BINARY_DIR}/sycl/flash_attentionXe35_decode_${HEAD_DIM}_${ELEM_TAG}.cpp")
        configure_file(${FMHA_XE35_DECODE_TEMPLATE} ${GENERATED_DECODE_FILE} @ONLY)
        list(APPEND device_cpp_xe35 ${GENERATED_DECODE_FILE})
    endforeach()

    # Non-paged (contiguous ragged KV) variant compiled as its own translation
    # unit so each object only instantiates a single KV-layout to bound memory.
    set(GENERATED_DECODE_NOPAGE_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/flash_attentionXe35_decode_nopage_${HEAD_DIM}.cpp")
    configure_file(${FMHA_XE35_DECODE_NOPAGE_TEMPLATE} ${GENERATED_DECODE_NOPAGE_FILE} @ONLY)
    list(APPEND device_cpp_xe35 ${GENERATED_DECODE_NOPAGE_FILE})
endforeach()
