CI_SGL_KERNEL_XPU_TESTS = [
    "tests/test_activation.py::test_fused_silu_mul[1-1-128]",
    "tests/test_mxfp4_blockwise_moe.py::TestMXFP4BlockwiseScaledGroupedMM::test_kernel_vs_reference[2-64-64-64]",
    "tests/test_mxfp4_blockwise_moe.py::TestMXFP4BlockwiseScaledGroupedMM::test_sanity_check_small_values",
    "tests/test_mxfp8_blockwise_moe.py::TestMXFP8BlockwiseScaledGroupedMM::test_kernel_vs_reference[2-128-128-128]",
    "tests/test_mxfp8_blockwise_moe.py::TestMXFP8BlockwiseScaledGroupedMM::test_single_expert",
    "tests/test_mxfp4_moe_w4a4.py::TestMxfp4MoeScaleToMnMajor::test_matches_reference[4-3x5]",
    "tests/test_mxfp4_moe_w4a4.py::TestMoeGroupedMmNtXe35Mxfp4W4A4::test_kernel_vs_reference[False]",
    "tests/test_mxfp4_moe_w4a4.py::TestMoeGroupedMmNtXe35Mxfp4W4A4::test_end_to_end_quant_relayout_gemm",
]
