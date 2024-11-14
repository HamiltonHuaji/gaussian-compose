import utils;

[BackwardDifferentiable]
float4 wxyz_to_xyzw(float4 q) {
    return float4(q.y, q.z, q.w, q.x);
}

[BackwardDifferentiable]
float4 xyzw_to_wxyz(float4 q) {
    return float4(q.w, q.x, q.y, q.z);
}

[BackwardDifferentiable]
float4 normalize(float4 q) {
    return q / sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
}

/**
 * Convert a quaternion to a rotation matrix. Assumes the quaternion is normalized.
 * @param q The quaternion to convert.
 * @return The rotation matrix.
 */
[BackwardDifferentiable]
float3x3 normalized_quaternion_to_matrix(float4 q) {
    float3x3 R;
    R[0][0] = 1 - 2 * (q.y * q.y + q.z * q.z);
    R[0][1] = 2 * (q.x * q.y - q.w * q.z);
    R[0][2] = 2 * (q.x * q.z + q.w * q.y);
    R[1][0] = 2 * (q.x * q.y + q.w * q.z);
    R[1][1] = 1 - 2 * (q.x * q.x + q.z * q.z);
    R[1][2] = 2 * (q.y * q.z - q.w * q.x);
    R[2][0] = 2 * (q.x * q.z - q.w * q.y);
    R[2][1] = 2 * (q.y * q.z + q.w * q.x);
    R[2][2] = 1 - 2 * (q.x * q.x + q.y * q.y);
    return R;
}

/**
 * Multiply two quaternions.
 * @param a The first quaternion.
 * @param b The second quaternion.
 * @return The product of a and b.
 */
[BackwardDifferentiable]
float4 quaternion_raw_multiply(float4 a, float4 b) {
    float4 result;
    result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    return result;
}

/**
 * Inverts a quaternion. Assumes the quaternion is normalized.
 * @param q The quaternion to invert.
 * @return The inverted quaternion.
 */
[BackwardDifferentiable]
float4 normalized_quaternion_invert(float4 q) {
    return float4(-q.x, -q.y, -q.z, q.w);
}

/**
 * Inverts a quaternion. The quaternion is not assumed to be normalized.
 * @param q The quaternion to invert.
 * @return The inverted quaternion.
 */
[BackwardDifferentiable]
float4 quaternion_invert(float4 q) {
    return float4(-q.x, -q.y, -q.z, q.w) / dot(q, q);
}

/**
 * Conjugates a quaternion.
 * @param q The quaternion to conjugate.
 * @return The conjugated quaternion.
 */
[BackwardDifferentiable]
float4 quaternion_conjugate(float4 q) {
    return float4(-q.x, -q.y, -q.z, q.w);
}

/**
 * Rotates a vector by a quaternion.
 * @param q The quaternion to rotate by. Assumes the quaternion is normalized.
 * @param v The vector to rotate.
 * @return The rotated vector.
 */
[BackwardDifferentiable]
float3 rotate_vector_with_normalized_quaternion(float4 q, float3 v) {
    float4 qv     = float4(v.x, v.y, v.z, 0);
    float4 q_inv  = normalized_quaternion_invert(q);
    float4 result = quaternion_raw_multiply(quaternion_raw_multiply(q, qv), q_inv);
    return float3(result.x, result.y, result.z);
}

/**
 * The real implementation of the transform_wxyz kernel
 */
[BackwardDifferentiable]
Tuple<float3, float4> transform_wxyz(float3 trans, float4 rotor, float3 mean, float4 quat) {
    float4 rotor_ = normalize(wxyz_to_xyzw(rotor));
    float4 quat_  = normalize(wxyz_to_xyzw(quat));

    float3 transformed_mean = rotate_vector_with_normalized_quaternion(rotor_, mean) + trans;
    float4 transformed_quat = quaternion_raw_multiply(rotor_, quat_);
    return Tuple<float3, float4>(transformed_mean, xyzw_to_wxyz(transformed_quat));
}

[CudaKernel]
void indexed_transform_wxyz_fwd_kernel(
    TensorView<float3> trans,  // (n, 3)
    TensorView<float4> rotors, // (n, 4)

    TensorView<float3>  means,   // (m, 3)
    TensorView<float4>  quats,   // (m, 4)
    TensorView<int32_t> indices, // (m, 1)

    TensorView<float3> result_means, // (m, 3)
    TensorView<float4> result_quats  // (m, 4)
) {
    int32_t m = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (m >= means.size(0)) {
        return;
    }
    uint32_t n = uint32_t(indices[m]);
    if (n >= trans.size(0)) {
        // intentionally set to NaN to indicate error
        result_means[m] = float3(0. / 0., 0. / 0., 0. / 0.);
        result_quats[m] = float4(0. / 0., 0. / 0., 0. / 0., 0. / 0.);
        return;
    }

    Tuple<float3, float4> result = transform_wxyz(trans[n], rotors[n], means[m], quats[m]);

    result_means[m] = result._0;
    result_quats[m] = result._1;
}

[CudaKernel]
void indexed_transform_wxyz_bwd_kernel(
    TensorView<float3> trans,  // (n, 3)
    TensorView<float4> rotors, // (n, 4)

    TensorView<float3>  means,   // (m, 3)
    TensorView<float4>  quats,   // (m, 4)
    TensorView<int32_t> indices, // (m, 1)

    TensorView<float3> trans_grad,  // (n, 3)
    TensorView<float4> rotors_grad, // (n, 4)

    TensorView<float3> means_grad, // (m, 3)
    TensorView<float4> quats_grad, // (m, 4)

    TensorView<float3> result_means_grad, // (m, 3)
    TensorView<float4> result_quats_grad  // (m, 4)
) {
    int32_t m = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (m >= means.size(0)) {
        return;
    }
    uint32_t n = uint32_t(indices[m]);
    // if (n >= trans.size(0)) {
    //     // intentionally set to NaN to indicate error
    //     means_grad[m] = float3(0., 0., 0.);
    //     quats_grad[m] = float4(0., 0., 0., 0.);
    //     return;
    // }

    float3 trans_value;
    float4 rotor_value;

    if (WaveActiveAllEqual(n)) {
        if (WaveIsFirstLane()) {
            trans_value = trans[n];
            rotor_value = rotors[n];
        }
        trans_value = WaveReadLaneFirst(trans_value);
        rotor_value = WaveReadLaneFirst(rotor_value);
    } else {
        trans_value = trans[n];
        rotor_value = rotors[n];
    }

    var dp_trans = diffPair(trans_value);
    var dp_rotor = diffPair(rotor_value);
    var dp_mean = diffPair(means[m]);
    var dp_quat = diffPair(quats[m]);

    // Tuple<float3, float4> result = transform_wxyz(dp_trans.p, dp_rotor.p, dp_mean.p, dp_quat.p);
    // var dp_result = diffPair(result);
    // dp_result.dadd(result_means_grad[m], result_quats_grad[m]);
    __bwd_diff(transform_wxyz)(dp_trans, dp_rotor, dp_mean, dp_quat, makeTuple(result_means_grad[m], result_quats_grad[m]));

    means_grad[m] = dp_mean.d;
    quats_grad[m] = dp_quat.d;

    // if (m == 0 && n == 0) {
    //     if (WaveActiveAllEqual(n)) {
    //         printf("WaveActiveAllEqual(n)=true\n");
    //     } else {
    //         printf("WaveActiveAllEqual(n)=false\n");
    //     }
    // }

    GroupMemoryBarrierWithWaveSync();

    if (WaveActiveAllEqual(n)) {
        // all n are the same
        var trans_grad_sum = WaveActiveSum(dp_trans.d);
        var rotors_grad_sum = WaveActiveSum(dp_rotor.d);
        if (WaveIsFirstLane()) {
            float3 trans_ignored;
            float4 rotors_ignored;
            trans_grad.InterlockedAdd(n, trans_grad_sum, trans_ignored);
            rotors_grad.InterlockedAdd(n, rotors_grad_sum, rotors_ignored);
        }
    } else {
        // naive implementation
        float3 trans_ignored;
        float4 rotors_ignored;
        trans_grad.InterlockedAdd(n, dp_trans.d, trans_ignored);
        rotors_grad.InterlockedAdd(n, dp_rotor.d, rotors_ignored);
    }
}

/////////////////////////////////////////////////////////////////////////////////
// Torch entry points
/////////////////////////////////////////////////////////////////////////////////

struct IndexedTransformWXYZForwardReturnType {
    TorchTensor<float3> means; // (m, 3)
    TorchTensor<float4> quats; // (m, 4)
}

[TorchEntryPoint]
IndexedTransformWXYZForwardReturnType indexed_transform_wxyz_fwd(
    // Translations and rotations
    TorchTensor<float3> trans,  // (n, 3)
    TorchTensor<float4> rotors, // (n, 4)

    TorchTensor<float3>  means,  // (m, 3)
    TorchTensor<float4>  quats,  // (m, 4)
    TorchTensor<int32_t> indices // (m, 1)
) {
    uint3 dims        = uint3(means.size(0), 1, 1);
    uint3 block_size  = uint3(256, 1, 1);
    uint3 block_count = getLaunchGridSize(block_size, dims);

    IndexedTransformWXYZForwardReturnType result;
    result.means = TorchTensor<float3>.emptyLike(means);
    result.quats = TorchTensor<float4>.emptyLike(quats);

    __dispatch_kernel(indexed_transform_wxyz_fwd_kernel, block_count, block_size)(trans, rotors, means, quats, indices, result.means, result.quats);

    return result;
}

struct IndexedTransformWXYZBackwardReturnType {
    TorchTensor<float3> trans_grad;  // (n, 3)
    TorchTensor<float4> rotors_grad; // (n, 4)

    TorchTensor<float3> means_grad; // (m, 3)
    TorchTensor<float4> quats_grad; // (m, 4)
    // No gradients for indices
}

[TorchEntryPoint]
IndexedTransformWXYZBackwardReturnType indexed_transform_wxyz_bwd(
    TorchTensor<float3> trans,  // (n, 3)
    TorchTensor<float4> rotors, // (n, 4)

    TorchTensor<float3>  means,   // (m, 3)
    TorchTensor<float4>  quats,   // (m, 4)
    TorchTensor<int32_t> indices, // (m, 1)

    TorchTensor<float3> result_means_grad, // (m, 3)
    TorchTensor<float4> result_quats_grad  // (m, 4)
) {
    IndexedTransformWXYZBackwardReturnType result;

    result.trans_grad  = TorchTensor<float3>.zerosLike(trans);
    result.rotors_grad = TorchTensor<float4>.zerosLike(rotors);

    result.means_grad = TorchTensor<float3>.zerosLike(means);
    result.quats_grad = TorchTensor<float4>.zerosLike(quats);

    uint3 dims        = uint3(means.size(0), 1, 1);
    uint3 block_size  = uint3(256, 1, 1);
    uint3 block_count = getLaunchGridSize(block_size, dims);

    __dispatch_kernel(indexed_transform_wxyz_bwd_kernel, block_count, block_size)(
        trans, rotors, means, quats, indices,
        result.trans_grad, result.rotors_grad, result.means_grad, result.quats_grad,
        result_means_grad, result_quats_grad);

    return result;
}
