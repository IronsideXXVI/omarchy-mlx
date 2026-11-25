// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/coopmat_select.h"

#include <algorithm>

namespace mlx::core::omarchy {

bool cooperative_matrix_advertises(
    const std::vector<CoopmatShape>& advertised, const CoopmatShape& want) {
  return std::find(advertised.begin(), advertised.end(), want) !=
      advertised.end();
}

CoopmatShape qmm_prefill_coopmat_shape() {
  CoopmatShape shape;
  shape.m = 8u;
  shape.n = 8u;
  shape.k = 8u;
  // Both output dtypes widen the activations to an fp32 A matrix before the
  // matmad (fp16 x is loaded as fp16 and converted), so the shape is fp32xfp32.
  shape.a = CoopmatType::Float32;
  shape.b = CoopmatType::Float32;
  shape.c = CoopmatType::Float32;
  shape.result = CoopmatType::Float32;
  shape.scope = CoopmatScope::Subgroup;
  shape.saturating = false;
  return shape;
}

CoopmatShape sdpa_causal_flash_coopmat16_bf16_shape() {
  return qmm_prefill_coopmat16_bf16_shape();
}

bool sdpa_causal_flash_coopmat16_enabled(
    const char* env,
    bool shape_advertised,
    int q_len,
    bool full_query_tiles) {
  if (env != nullptr && env[0] == '0') {
    return false;
  }
  if (!shape_advertised || !full_query_tiles) {
    return false;
  }
  return q_len >= 64 || (env != nullptr && env[0] == '1');
}

CoopmatShape qmm_prefill_coopmat16_bf16_shape() {
  CoopmatShape shape;
  shape.m = 16u;
  shape.n = 16u;
  shape.k = 16u;
  shape.a = CoopmatType::BFloat16;
  shape.b = CoopmatType::BFloat16;
  shape.c = CoopmatType::Float32;
  shape.result = CoopmatType::Float32;
  shape.scope = CoopmatScope::Subgroup;
  shape.saturating = false;
  return shape;
}

} // namespace mlx::core::omarchy
