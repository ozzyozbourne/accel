BOOK.register('index', {
 "add/default": {
  "chapter": 1,
  "kernels": [
   "E_8192_32_4"
  ],
  "time_us": 21.57,
  "ok": true,
  "passes": 52
 },
 "add/torch": {
  "chapter": null,
  "eager": [
   "void at::native::vectorized_elementwise_kernel<4, at::native"
  ],
  "triton": [
   "triton_poi_fused_add_0",
   "triton_poi_fused_add_0"
  ]
 },
 "add_ptx/cuda": {
  "chapter": 9,
  "kernels": [
   "E_8192_32_4"
  ],
  "time_us": 22.46,
  "ok": true,
  "passes": 52
 },
 "add_ptx/ptx": {
  "chapter": 9,
  "kernels": [
   "E_8192_32_4"
  ],
  "time_us": 22.53,
  "ok": true,
  "passes": 52
 },
 "broadcast_add/default": {
  "chapter": 3,
  "kernels": [
   "E_8_64_32_4_4_4"
  ],
  "time_us": 32.77,
  "ok": true,
  "passes": 77
 },
 "broadcast_add/torch": {
  "chapter": null,
  "eager": [
   "void at::native::elementwise_kernel<128, 2, at::native::gpu_"
  ],
  "triton": [
   "triton_poi_fused_add_unsqueeze_0",
   "triton_poi_fused_add_unsqueeze_0"
  ]
 },
 "exp/default": {
  "chapter": 8,
  "kernels": [
   "E_8192_32_4"
  ],
  "time_us": 24.58,
  "ok": true,
  "passes": 52
 },
 "exp/torch": {
  "chapter": null,
  "eager": [
   "void at::native::vectorized_elementwise_kernel<4, at::native"
  ],
  "triton": [
   "triton_poi_fused_exp_0",
   "triton_poi_fused_exp_0"
  ]
 },
 "exp/transcendental2": {
  "chapter": 8,
  "kernels": [
   "E_8192_32_4"
  ],
  "time_us": 23.55,
  "ok": true,
  "passes": 52
 },
 "fused_chain/default": {
  "chapter": 2,
  "kernels": [
   "E_8192_32_4"
  ],
  "time_us": 24.54,
  "ok": true,
  "passes": 54
 },
 "fused_chain/torch": {
  "chapter": null,
  "eager": [
   "void at::native::vectorized_elementwise_kernel<4, at::native",
   "void at::native::vectorized_elementwise_kernel<4, at::native",
   "void at::native::vectorized_elementwise_kernel<4, at::native",
   "void at::native::vectorized_elementwise_kernel<4, at::native"
  ],
  "triton": [
   "triton_poi_fused_add_mul_relu_0",
   "triton_poi_fused_add_mul_relu_0"
  ]
 },
 "gelu/default": {
  "chapter": 2,
  "kernels": [
   "E_8192_32_4"
  ],
  "time_us": 22.53,
  "ok": true,
  "passes": 54
 },
 "gelu/torch": {
  "chapter": null,
  "eager": [
   "void at::native::vectorized_elementwise_kernel<4, at::native"
  ],
  "triton": [
   "triton_poi_fused_gelu_0",
   "triton_poi_fused_gelu_0"
  ]
 },
 "int_divmod/default": {
  "chapter": 8,
  "kernels": [
   "E_8192_32_4"
  ],
  "time_us": 22.53,
  "ok": true,
  "passes": 54
 },
 "int_divmod/torch": {
  "chapter": null,
  "eager": [
   "void at::native::vectorized_elementwise_kernel<4, at::native",
   "void at::native::vectorized_elementwise_kernel<4, at::native",
   "void at::native::vectorized_elementwise_kernel<4, at::native"
  ],
  "triton": [
   "triton_poi_fused_add_floor_divide_remainder_0",
   "triton_poi_fused_add_floor_divide_remainder_0"
  ]
 },
 "jit/default": {
  "chapter": 10
 },
 "layernorm/default": {
  "chapter": 5,
  "kernels": [
   "r_1024_16_64",
   "r_1024_16_64",
   "E_128_16_8_16_4"
  ],
  "time_us": 70.66,
  "ok": true,
  "passes": 117
 },
 "layernorm/torch": {
  "chapter": null,
  "eager": [
   "void at::native::(anonymous namespace)::vectorized_layer_nor"
  ],
  "triton": [
   "triton_per_fused_native_layer_norm_0"
  ]
 },
 "matmul/beam2": {
  "chapter": 6,
  "kernels": [
   "r_16_8_16_8_4_4_4_256_4"
  ],
  "time_us": 224.26,
  "ok": true,
  "passes": 519
 },
 "matmul/default": {
  "chapter": 6,
  "kernels": [
   "r_32_16_8_16_4_4_256_4"
  ],
  "time_us": 686.08,
  "ok": true,
  "passes": 170
 },
 "matmul/noopt": {
  "chapter": 6,
  "kernels": [
   "r_1024_1024_1024"
  ],
  "time_us": 37240.0,
  "ok": true,
  "passes": 54
 },
 "matmul/torch": {
  "chapter": null,
  "eager": [
   "Memset (Device)",
   "ampere_sgemm_128x64_nn"
  ],
  "triton": []
 },
 "matmul_fp16/default": {
  "chapter": 7,
  "kernels": [
   "r_32_16_32_4_2_2_4_4_128_2"
  ],
  "time_us": 527.36,
  "ok": true,
  "passes": 466
 },
 "matmul_fp16/no_tc": {
  "chapter": 7,
  "kernels": [
   "r_64_32_8_16_4_4_512_4"
  ],
  "time_us": 2841.6,
  "ok": true,
  "passes": 170
 },
 "matmul_fp16/torch": {
  "chapter": null,
  "eager": [
   "Memset (Device)",
   "ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn",
   "void at::native::unrolled_elementwise_kernel<at::native::dir"
  ],
  "triton": [
   "triton_poi_fused__to_copy_0",
   "triton_poi_fused__to_copy_0"
  ]
 },
 "matmul_ptx/cuda": {
  "chapter": 9,
  "kernels": [
   "r_32_16_8_16_4_4_256_4"
  ],
  "time_us": 684.03,
  "ok": true,
  "passes": 170
 },
 "matmul_ptx/ptx": {
  "chapter": 9,
  "kernels": [
   "r_32_16_8_16_4_4_256_4"
  ],
  "time_us": 690.18,
  "ok": true,
  "passes": 170
 },
 "max_cols/default": {
  "chapter": 4,
  "kernels": [
   "r_1024_16_64"
  ],
  "time_us": 33.79,
  "ok": true,
  "passes": 54
 },
 "max_cols/torch": {
  "chapter": null,
  "eager": [
   "Memset (Device)",
   "void at::native::reduce_kernel<128, 4, at::native::ReduceOp<"
  ],
  "triton": [
   "triton_red_fused_amax_0",
   "triton_per_fused_amax_1"
  ]
 },
 "mlp_backward/default": {
  "chapter": 10,
  "kernels": [
   "r_8_4_8_16_4_4_128_4",
   "r_8_8_8_16_4_4_64_4"
  ],
  "time_us": 108.55,
  "ok": true,
  "passes": 204
 },
 "mlp_backward/torch": {
  "chapter": null,
  "eager": [
   "void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al",
   "void cublasLt::splitKreduce_kernel<32, 16, int, float, float",
   "void at::native::vectorized_elementwise_kernel<4, at::native",
   "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<",
   "void at::native::vectorized_elementwise_kernel<4, at::native",
   "void at::native::elementwise_kernel<128, 2, at::native::gpu_",
   "ampere_sgemm_32x32_sliced1x4_nt"
  ],
  "triton": [
   "triton_poi_fused_expand_threshold_backward_0",
   "triton_poi_fused_expand_threshold_backward_0",
   "triton_per_fused_relu_sum_1",
   "triton_red_fused_relu_sum_threshold_backward_0"
  ]
 },
 "pad_slice/default": {
  "chapter": 3,
  "kernels": [
   "E_251_337_2"
  ],
  "time_us": 67.58,
  "ok": true,
  "passes": 63
 },
 "pad_slice/torch": {
  "chapter": null,
  "eager": [
   "void at::native::vectorized_elementwise_kernel<4, at::native",
   "void at::native::elementwise_kernel<128, 2, at::native::gpu_"
  ],
  "triton": [
   "triton_poi_fused_constant_pad_nd_slice_0",
   "triton_poi_fused_constant_pad_nd_slice_0"
  ]
 },
 "permute4d/default": {
  "chapter": 3,
  "kernels": [
   "E_32_32_2_16_8_4"
  ],
  "time_us": 22.53,
  "ok": true,
  "passes": 54
 },
 "permute4d/torch": {
  "chapter": null,
  "eager": [
   "void at::native::elementwise_kernel<128, 2, at::native::gpu_"
  ],
  "triton": [
   "triton_poi_fused_clone_permute_0",
   "triton_poi_fused_clone_permute_0"
  ]
 },
 "sin/default": {
  "chapter": 8,
  "kernels": [
   "E_8192_32_4"
  ],
  "time_us": 23.55,
  "ok": true,
  "passes": 52
 },
 "sin/torch": {
  "chapter": null,
  "eager": [
   "void at::native::vectorized_elementwise_kernel<4, at::native"
  ],
  "triton": [
   "triton_poi_fused_sin_0",
   "triton_poi_fused_sin_0"
  ]
 },
 "sin/transcendental2": {
  "chapter": 8,
  "kernels": [
   "E_8192_32_4"
  ],
  "time_us": 69.63,
  "ok": true,
  "passes": 52
 },
 "softmax/default": {
  "chapter": 5,
  "kernels": [
   "r_1024_16_64",
   "r_1024_16_64",
   "E_128_16_8_16_4"
  ],
  "time_us": 67.59,
  "ok": true,
  "passes": 117
 },
 "softmax/torch": {
  "chapter": null,
  "eager": [
   "void (anonymous namespace)::softmax_warp_forward<float, floa"
  ],
  "triton": [
   "triton_per_fused__softmax_prepare_softmax_online_0"
  ]
 },
 "sum_all/default": {
  "chapter": 4,
  "kernels": [
   "r_32_32_4_1024_4",
   "r_256_16",
   "r_16_16"
  ],
  "time_us": 505.77,
  "ok": true,
  "passes": 148
 },
 "sum_all/torch": {
  "chapter": null,
  "eager": [
   "Memset (Device)",
   "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<"
  ],
  "triton": [
   "triton_red_fused_sum_0",
   "triton_per_fused_sum_1"
  ]
 },
 "sum_rows/default": {
  "chapter": 4,
  "kernels": [
   "r_1024_16_64"
  ],
  "time_us": 25.6,
  "ok": true,
  "passes": 54
 },
 "sum_rows/torch": {
  "chapter": null,
  "eager": [
   "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<"
  ],
  "triton": [
   "triton_per_fused_sum_0"
  ]
 },
 "sum_rows_ptx/cuda": {
  "chapter": 9,
  "kernels": [
   "r_1024_16_64"
  ],
  "time_us": 25.54,
  "ok": true,
  "passes": 54
 },
 "sum_rows_ptx/ptx": {
  "chapter": 9,
  "kernels": [
   "r_1024_16_64"
  ],
  "time_us": 27.65,
  "ok": true,
  "passes": 54
 },
 "transpose/default": {
  "chapter": 3,
  "kernels": [
   "E_128_16_8_16_4"
  ],
  "time_us": 23.55,
  "ok": true,
  "passes": 54
 },
 "transpose/torch": {
  "chapter": null,
  "eager": [
   "void at::native::elementwise_kernel<128, 2, at::native::gpu_"
  ],
  "triton": [
   "triton_poi_fused_clone_t_0"
  ]
 }
});
