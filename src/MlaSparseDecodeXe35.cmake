# Generate Xe3 sparse MLA decode kernel instantiations.
set(MLA_SPARSE_DECODE_ELEM_TAGS bf16)
set(MLA_SPARSE_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/mla_sparseXe35_decode_kernel.cpp.in")
set(MLA_SPARSE_DECODE_D_QK 512)
set(MLA_SPARSE_DECODE_B_H 8 16 32 64)
set(MLA_SPARSE_DECODE_HAS_ATTN_SINK 0 1)

foreach(ELEM_TAG ${MLA_SPARSE_DECODE_ELEM_TAGS})
    if(ELEM_TAG STREQUAL "bf16")
        set(ELEM_SYCL_TYPE "sycl::ext::oneapi::bfloat16")
    else()
        message(FATAL_ERROR "Unknown sparse MLA decode element type: ${ELEM_TAG}")
    endif()

    foreach(D_QK ${MLA_SPARSE_DECODE_D_QK})
        foreach(B_H ${MLA_SPARSE_DECODE_B_H})
            foreach(HAS_ATTN_SINK ${MLA_SPARSE_DECODE_HAS_ATTN_SINK})
                set(GENERATED_FILE
                    "${CMAKE_CURRENT_BINARY_DIR}/sycl/mla_sparseXe35_decode_kernel_${ELEM_TAG}_${D_QK}_${B_H}_${HAS_ATTN_SINK}.cpp")
                configure_file(${MLA_SPARSE_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
                list(APPEND device_cpp_xe35 ${GENERATED_FILE})
            endforeach()
        endforeach()
    endforeach()
endforeach()
