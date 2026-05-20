#include "qwen_onnx/tensor.h"

#include <algorithm>

namespace qwen::onnx {

// 所有 FloatTensor helper 都假设数据是 row-major 连续布局。
// 这里没有做广播：shape 不一致通常代表 prompt 构造逻辑有 bug，直接抛错更容易定位。
FloatTensor Add(const FloatTensor& a, const FloatTensor& b) {
  if (a.shape() != b.shape()) throw std::invalid_argument("Add expects equal shapes");
  FloatTensor out(a.shape());
  for (size_t i = 0; i < a.size(); ++i) out.values()[i] = a.values()[i] + b.values()[i];
  return out;
}

FloatTensor ConcatAxis1(const std::vector<FloatTensor>& tensors) {
  if (tensors.empty()) return {};
  const auto& first = tensors.front().shape();
  if (first.size() != 3) throw std::invalid_argument("ConcatAxis1 expects rank-3 tensors");
  int64_t total = 0;
  for (const auto& t : tensors) {
    if (t.shape().size() != 3 || t.shape()[0] != first[0] || t.shape()[2] != first[2]) {
      throw std::invalid_argument("ConcatAxis1 shape mismatch");
    }
    total += t.shape()[1];
  }
  FloatTensor out({first[0], total, first[2]});
  size_t offset = 0;
  for (const auto& t : tensors) {
    // [B,T,D] 在内存里已经按时间轴连续排列，所以 axis=1 拼接可以整段 copy。
    std::copy(t.values().begin(), t.values().end(), out.values().begin() + static_cast<long>(offset));
    offset += t.size();
  }
  return out;
}

FloatTensor RepeatAxis1(const FloatTensor& tensor, int64_t repeats) {
  if (tensor.shape().size() != 3 || tensor.shape()[1] != 1) {
    throw std::invalid_argument("RepeatAxis1 expects shape [B,1,D]");
  }
  FloatTensor out({tensor.shape()[0], repeats, tensor.shape()[2]});
  const int64_t d = tensor.shape()[2];
  for (int64_t r = 0; r < repeats; ++r) {
    // 当前项目里 RepeatAxis1 主要用于把 tts_pad embedding 扩展到 codec/text 对齐长度。
    std::copy(tensor.values().begin(), tensor.values().end(), out.values().begin() + r * d);
  }
  return out;
}

FloatTensor SliceAxis1(const FloatTensor& tensor, int64_t begin, int64_t end) {
  if (tensor.shape().size() != 3) throw std::invalid_argument("SliceAxis1 expects rank-3 tensor");
  const int64_t b = tensor.shape()[0];
  const int64_t t = tensor.shape()[1];
  const int64_t d = tensor.shape()[2];
  if (begin < 0) begin += t;
  if (end < 0) end += t;
  begin = std::max<int64_t>(0, begin);
  end = std::min<int64_t>(t, end);
  if (end < begin) end = begin;
  FloatTensor out({b, end - begin, d});
  for (int64_t bi = 0; bi < b; ++bi) {
    // 每个 batch 单独 copy [begin,end) 的时间片。
    auto src = tensor.values().begin() + (bi * t + begin) * d;
    auto dst = out.values().begin() + bi * (end - begin) * d;
    std::copy(src, src + (end - begin) * d, dst);
  }
  return out;
}

FloatTensor SumAxis1KeepDims(const FloatTensor& tensor) {
  if (tensor.shape().size() != 3) throw std::invalid_argument("SumAxis1KeepDims expects rank-3 tensor");
  const int64_t b = tensor.shape()[0];
  const int64_t t = tensor.shape()[1];
  const int64_t d = tensor.shape()[2];
  FloatTensor out({b, 1, d});
  for (int64_t bi = 0; bi < b; ++bi) {
    for (int64_t ti = 0; ti < t; ++ti) {
      for (int64_t di = 0; di < d; ++di) {
        out.values()[bi * d + di] += tensor.values()[(bi * t + ti) * d + di];
      }
    }
  }
  return out;
}

Int64Tensor MakeInt64Tensor(std::vector<int64_t> shape, std::initializer_list<int64_t> values) {
  // 小型常量张量的便利函数，例如 scalar layer_idx 或一行 special token ids。
  return Int64Tensor(std::move(shape), std::vector<int64_t>(values));
}

}  // 命名空间 qwen::onnx
