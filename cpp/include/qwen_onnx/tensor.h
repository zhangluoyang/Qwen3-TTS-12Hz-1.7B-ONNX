#pragma once

// 这个头文件定义 C++ runtime 内部使用的最小 Tensor 容器。
//
// 项目没有引入 Eigen/xtensor 之类的大型张量库，原因是 ONNX Runtime
// 本身已经负责真正的模型计算；C++ 侧只需要保存 shape + 连续内存，
// 并做少量 prompt 拼接、slice、sum 这类轻量操作。

#include <cstdint>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen::onnx {

// Tensor<T> 是“带 shape 的 std::vector<T>”。
//
// shape 使用 ONNX/Numpy 风格，例如：
//   [1, T]      文本 token ids
//   [T, 16]     12Hz codec codes
//   [1, T, D]   talker 输入 embedding / hidden state
//
// 构造时会校验 shape 的元素数量和 values 数量一致，这能很早发现
// prompt 拼接或 ONNX 输出 reshape 的错误。
template <typename T>
class Tensor {
 public:
  Tensor() = default;
  Tensor(std::vector<int64_t> shape, std::vector<T> values)
      : shape_(std::move(shape)), values_(std::move(values)) {
    if (NumElements(shape_) != values_.size()) {
      std::ostringstream ss;
      ss << "Tensor shape does not match value count: shape=[";
      for (size_t i = 0; i < shape_.size(); ++i) {
        if (i) ss << ",";
        ss << shape_[i];
      }
      ss << "] expected=" << NumElements(shape_) << " values=" << values_.size();
      throw std::invalid_argument(ss.str());
    }
  }

  explicit Tensor(std::vector<int64_t> shape) : shape_(std::move(shape)), values_(NumElements(shape_)) {}

  const std::vector<int64_t>& shape() const { return shape_; }
  std::vector<int64_t>& shape() { return shape_; }
  const std::vector<T>& values() const { return values_; }
  std::vector<T>& values() { return values_; }
  const T* data() const { return values_.data(); }
  T* data() { return values_.data(); }
  size_t size() const { return values_.size(); }
  bool empty() const { return values_.empty(); }

  static size_t NumElements(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1;
    size_t n = 1;
    for (int64_t d : shape) {
      if (d < 0) throw std::invalid_argument("Runtime tensor shape cannot contain negative dims");
      n *= static_cast<size_t>(d);
    }
    return n;
  }

 private:
  std::vector<int64_t> shape_;
  std::vector<T> values_;
};

using FloatTensor = Tensor<float>;
using Int64Tensor = Tensor<int64_t>;

// 下面这些 helper 只覆盖本项目需要的形状操作，主要用于构造 talker prompt：
// - Add:        两个 [B,T,D] embedding 相加
// - ConcatAxis1:沿时间轴 T 拼接
// - RepeatAxis1:把 [B,1,D] 的 pad embedding 复制成 [B,N,D]
// - SliceAxis1: 沿时间轴取切片
// - SumAxis1KeepDims:把一帧内多个 codebook embedding 求和为 [B,1,D]
FloatTensor Add(const FloatTensor& a, const FloatTensor& b);
FloatTensor ConcatAxis1(const std::vector<FloatTensor>& tensors);
FloatTensor RepeatAxis1(const FloatTensor& tensor, int64_t repeats);
FloatTensor SliceAxis1(const FloatTensor& tensor, int64_t begin, int64_t end);
FloatTensor SumAxis1KeepDims(const FloatTensor& tensor);
Int64Tensor MakeInt64Tensor(std::vector<int64_t> shape, std::initializer_list<int64_t> values);

}  // 命名空间 qwen::onnx
