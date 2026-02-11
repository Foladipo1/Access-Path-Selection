#pragma once

#include <vector>
#include <cstdint>
#include <type_traits>
#include <algorithm>
#include <stdexcept>
#include <immintrin.h>
#include <iostream>
#include <unordered_set>
#include "duckdb/common/types/selection_vector.hpp"

namespace duckdb{
template<typename BaseType, typename CodeType>
class ColumnSketch {
    static_assert(std::is_same<BaseType, uint64_t>::value || std::is_same<BaseType, uint32_t>::value,
                  "BaseType must be uint64_t or uint32_t");
    static_assert(std::is_same<CodeType, uint8_t>::value || std::is_same<CodeType, uint16_t>::value,
                  "CodeType must be uint16_t or uint8_t");

private:
    void build_order_preserving_map();
    void build_sketch_column();

    std::vector<BaseType> base_data_;
    std::vector<CodeType> sketch_;
    std::vector<BaseType> endpoints_;
    size_t num_codes_;
    bool need_check_base = true;
    BaseType min_data_;
    BaseType max_data_;

public:
    ColumnSketch(const std::vector<BaseType>& base_data);
    
    const std::vector<CodeType> &GetSketchColumn() const { return sketch_; }
    const std::vector<BaseType> &GetBaseData() const { return base_data_; }
    const std::vector<BaseType> &GetEndpoints() const { return endpoints_; }
    size_t GetNumCodes() const { return num_codes_; }

    std::vector<uint32_t> evaluate_less_than_CPU(BaseType x) const;
    std::vector<uint32_t> evaluate_less_than_AVX2(BaseType x) const;
    idx_t evaluate_less_than_AVX512(BaseType x, ManagedSelection &sel) const;
    idx_t evaluate_lessthan_orequal_avx512(BaseType x, ManagedSelection &sel) const;

    std::vector<uint32_t> evaluate_greater_than_CPU(BaseType x) const;
    std::vector<uint32_t> evaluate_greater_than_AVX2(BaseType x) const;
    idx_t evaluate_greater_than_AVX512(BaseType x, ManagedSelection &sel) const;
    idx_t evaluate_greaterthan_orequal_avx512(BaseType x, ManagedSelection &sel) const;

    std::vector<uint32_t> evaluate_equal(BaseType x) const;
    std::vector<uint32_t> evaluate_between(BaseType x1, BaseType x2) const;
};

template<typename BaseType, typename CodeType>
ColumnSketch<BaseType, CodeType>::ColumnSketch(const std::vector<BaseType>& base_data)
    : base_data_(base_data) {

    num_codes_ = std::numeric_limits<CodeType>::max() + 1;

    build_order_preserving_map();
    build_sketch_column();

    // std::cout << "Build sketches for array of size " << base_data_.size() << " using " << num_codes_ << 
    //         " codes." << std::endl;
}

template<typename BaseType, typename CodeType>
void ColumnSketch<BaseType, CodeType>::build_order_preserving_map() {
    std::vector<BaseType> sample = base_data_;
    std::sort(sample.begin(), sample.end());

    num_codes_ = std::min(sample.size(), num_codes_);
    endpoints_.resize(num_codes_);

    std::unordered_set<BaseType> unique_values(sample.begin(), sample.end());
    size_t unique_count = unique_values.size();
    if (unique_count<=num_codes_) need_check_base = false;
    min_data_ = sample.front();
    max_data_ = sample.back();

    size_t n = sample.size();
    for (size_t i = 0; i < num_codes_; ++i) {
        size_t idx = (i + 1) * n / num_codes_ - 1;
        endpoints_[i] = sample[std::min(idx, n - 1)];
    }
}

template<typename BaseType, typename CodeType>
void ColumnSketch<BaseType, CodeType>::build_sketch_column() {
    sketch_.resize(base_data_.size());
    for (size_t i = 0; i < base_data_.size(); ++i) {
        BaseType val = base_data_[i];
        auto it = std::lower_bound(endpoints_.begin(), endpoints_.end(), val);
        sketch_[i] = static_cast<CodeType>(std::distance(endpoints_.begin(), it));
    }
}

template<typename BaseType, typename CodeType>
std::vector<uint32_t> ColumnSketch<BaseType, CodeType>::evaluate_less_than_CPU(BaseType x) const {
    std::vector<uint32_t> result;

    if (!endpoints_.empty() && x > max_data_) {
        result.reserve(base_data_.size());
        for (size_t i = 0; i < sketch_.size(); ++i) {
            result.push_back(static_cast<uint32_t>(i));
        }
        return result;
    }

    CodeType code_x = static_cast<CodeType>(
        std::distance(endpoints_.begin(), std::lower_bound(endpoints_.begin(), endpoints_.end(), x))
    );
    for (size_t i = 0; i < sketch_.size(); ++i) {
        if (sketch_[i] < code_x || (sketch_[i] == code_x && base_data_[i] < x))
            result.push_back(static_cast<uint32_t>(i));
    }
    return result;
}

template<typename BaseType, typename CodeType>
std::vector<uint32_t> ColumnSketch<BaseType, CodeType>::evaluate_less_than_AVX2(BaseType x) const {
    static_assert(std::is_same<CodeType, uint8_t>::value, "Only uint8_t code SIMD supported");

    std::vector<uint32_t> result;
    CodeType code_x = static_cast<CodeType>(
        std::distance(endpoints_.begin(), std::lower_bound(endpoints_.begin(), endpoints_.end(), x))
    );

    __m256i cmp = _mm256_set1_epi8(static_cast<char>(code_x));
    size_t i = 0, n = sketch_.size();
    for (; i + 32 <= n; i += 32) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&sketch_[i]));
        __m256i lt = _mm256_cmpgt_epi8(cmp, data);
        int mask = _mm256_movemask_epi8(lt);
        for (int j = 0; j < 32; ++j)
            if (mask & (1 << j)) result.push_back(i + j);
    }
    for (; i < n; ++i) {
        if (sketch_[i] < code_x || (sketch_[i] == code_x && base_data_[i] < x))
            result.push_back(static_cast<uint32_t>(i));
    }
    return result;
}

template<typename BaseType, typename CodeType>
idx_t ColumnSketch<BaseType, CodeType>::evaluate_less_than_AVX512(BaseType x, ManagedSelection &msel) const {
#if defined(__AVX512F__)
    static_assert(std::is_same<CodeType, uint8_t>::value, "Only uint8_t code AVX-512 supported");

    if (!endpoints_.empty() && x <= min_data_) {
        return 0;
    }

    msel.bitmask.clear();
    SelectionVector &sel = msel.Selection();
    sel.Initialize(base_data_.size());
    
    
    if (!endpoints_.empty() && x > max_data_) {
        sel_t *data = sel.data();
        size_t m = base_data_.size();
        int16_t k = 0;
        for (; k + 16 <= m; k += 16) {
            __m512i idx = _mm512_set_epi32(k+15, k+14, k+13, k+12, k+11, k+10, k+9, k+8,
                                           k+7, k+6, k+5, k+4, k+3, k+2, k+1, k);
            _mm512_storeu_si512(reinterpret_cast<void*>(data + k), idx);
        }
        for (; k < m; ++k) {
            data[k] = k;
        }

        size_t mask_count = (m + 63) / 64;
        msel.bitmask.resize(mask_count, 0xFFFFFFFFFFFFFFFFULL);
        size_t remain = m % 64;
        if (remain != 0) {
            msel.bitmask[mask_count - 1] = (1ULL << remain) - 1;
        }

        msel.SetCount(m);
        return static_cast<int32_t>(m);
    }

    CodeType code_x = static_cast<CodeType>(
        std::distance(endpoints_.begin(), std::lower_bound(endpoints_.begin(), endpoints_.end(), x))
    );

    __m512i cmp = _mm512_set1_epi8(static_cast<char>(code_x));
    size_t i = 0, n = sketch_.size();
    idx_t cnt = 0;
    uint32_t offset;
    if(!need_check_base) {
        for (; i + 64 <= n; i += 64) {
            __m512i data = _mm512_loadu_si512(reinterpret_cast<const void*>(&sketch_[i]));
            
            __mmask64 lt_mask = _mm512_cmplt_epu8_mask(data, cmp);
            msel.bitmask.push_back(static_cast<uint64_t>(lt_mask));
            while (lt_mask != 0) {
                uint32_t pos = i + __builtin_ctzll(lt_mask);
                sel.set_index(cnt++,pos);
                lt_mask = lt_mask & (lt_mask - 1);
            }
        }
    }
    else {
        for (; i + 64 <= n; i += 64) {
            __m512i data = _mm512_loadu_si512(reinterpret_cast<const void*>(&sketch_[i]));
            
            __mmask64 lt_mask = _mm512_cmplt_epu8_mask(data, cmp);
            __mmask64 eq_mask = _mm512_cmpeq_epu8_mask(data, cmp);
            __mmask64 temp_mask = eq_mask;
            while (temp_mask != 0) {
                offset = __builtin_ctzll(temp_mask);
                uint32_t pos = i + offset;
                if (base_data_[pos] >= x) {
                    eq_mask &= ~(1ULL << offset);
                }
                temp_mask = temp_mask & (temp_mask - 1);
            }
            __mmask64 mask = lt_mask | eq_mask;
            msel.bitmask.push_back(static_cast<uint64_t>(mask));
            while (mask != 0) {
                uint32_t pos = i + __builtin_ctzll(mask);
                sel.set_index(cnt++,pos);
                mask = mask & (mask - 1);
            }
        }
    }
    
    uint64_t tail_mask = 0;
    int tail_bits = 0;
    for (; i < n; ++i, ++tail_bits) {
        if (sketch_[i] < code_x || (sketch_[i] == code_x && base_data_[i] < x)) {
            sel.set_index(cnt++,i);
            tail_mask |= (1ULL << tail_bits);
        }   
    }
   
    msel.bitmask.push_back(tail_mask);

    msel.SetCount(cnt);
    return static_cast<int32_t>(cnt);
#else
    throw std::runtime_error("AVX-512 not supported");
#endif
}

template<typename BaseType, typename CodeType>
idx_t ColumnSketch<BaseType, CodeType>::evaluate_lessthan_orequal_avx512(BaseType x, ManagedSelection &msel) const {
#if defined(__AVX512F__)
    static_assert(std::is_same<CodeType, uint8_t>::value, "Only uint8_t code AVX-512 supported");

    if (!endpoints_.empty() && x < min_data_) {
        return 0;
    }

    msel.bitmask.clear();
    SelectionVector &sel = msel.Selection();
    sel.Initialize(base_data_.size());

    if (!endpoints_.empty() && (x > max_data_ || (!need_check_base && x >= max_data_))) {
        sel_t *data = sel.data();
        size_t m = base_data_.size();
        int16_t k = 0;
        for (; k + 16 <= m; k += 16) {
            __m512i idx = _mm512_set_epi32(k+15, k+14, k+13, k+12, k+11, k+10, k+9, k+8,
                                           k+7, k+6, k+5, k+4, k+3, k+2, k+1, k);
            _mm512_storeu_si512(reinterpret_cast<void*>(data + k), idx);
        }
        for (; k < m; ++k) {
            data[k] = k;
        }

        size_t mask_count = (m + 63) / 64;
        msel.bitmask.resize(mask_count, 0xFFFFFFFFFFFFFFFFULL);
        size_t remain = m % 64;
        if (remain != 0) {
            msel.bitmask[mask_count - 1] = (1ULL << remain) - 1;
        }

        msel.SetCount(m);
        return static_cast<int32_t>(m);
    }

    CodeType code_x = static_cast<CodeType>(
        std::distance(endpoints_.begin(), std::lower_bound(endpoints_.begin(), endpoints_.end(), x))
    );

    __m512i cmp = _mm512_set1_epi8(static_cast<char>(code_x));
    size_t i = 0, n = sketch_.size();
    idx_t cnt = 0;
    uint32_t offset;
    if(!need_check_base) {
        for (; i + 64 <= n; i += 64) {
            __m512i data = _mm512_loadu_si512(reinterpret_cast<const void*>(&sketch_[i]));
            
            __mmask64 lt_mask = _mm512_cmple_epu8_mask(data, cmp);

            msel.bitmask.push_back(static_cast<uint64_t>(lt_mask));
            while (lt_mask != 0) {
                uint32_t pos = i + __builtin_ctzll(lt_mask);
                sel.set_index(cnt++,pos);
                lt_mask = lt_mask & (lt_mask - 1);
            }
        }
    }
    else {
        for (; i + 64 <= n; i += 64) {
            __m512i data = _mm512_loadu_si512(reinterpret_cast<const void*>(&sketch_[i]));
            
            __mmask64 lt_mask = _mm512_cmplt_epu8_mask(data, cmp);
            __mmask64 eq_mask = _mm512_cmpeq_epu8_mask(data, cmp);
            __mmask64 temp_mask = eq_mask;
            while (temp_mask != 0) {
                offset = __builtin_ctzll(temp_mask);
                uint32_t pos = i + offset;
                if (base_data_[pos] > x) {
                    eq_mask &= ~(1ULL << offset);
                }
                temp_mask = temp_mask & (temp_mask - 1);
            }
            __mmask64 mask = lt_mask | eq_mask;
            msel.bitmask.push_back(static_cast<uint64_t>(mask));
            while (mask != 0) {
                uint32_t pos = i + __builtin_ctzll(mask);
                sel.set_index(cnt++,pos);
                mask = mask & (mask - 1);
            }
        }
    }
    uint64_t tail_mask = 0;
    int tail_bits = 0;
    for (; i < n; ++i, ++tail_bits) {
        if (sketch_[i] <= code_x || (sketch_[i] == code_x && base_data_[i] <= x)) {
            sel.set_index(cnt++,i);
            tail_mask |= (1ULL << tail_bits);
        }   
    }

    msel.bitmask.push_back(tail_mask);

    msel.SetCount(cnt);
    return static_cast<int32_t>(cnt);
#else
    throw std::runtime_error("AVX-512 not supported");
#endif
}

template<typename BaseType, typename CodeType>
std::vector<uint32_t> ColumnSketch<BaseType, CodeType>::evaluate_greater_than_CPU(BaseType x) const {
    std::vector<uint32_t> result;

    CodeType code_x = static_cast<CodeType>(
        std::distance(endpoints_.begin(), std::lower_bound(endpoints_.begin(), endpoints_.end(), x))
    );
    for (size_t i = 0; i < sketch_.size(); ++i) {
        if (sketch_[i] > code_x || (sketch_[i] == code_x && base_data_[i] > x))
            result.push_back(static_cast<uint32_t>(i));
    }
    return result;
}

template<typename BaseType, typename CodeType>
idx_t ColumnSketch<BaseType, CodeType>::evaluate_greater_than_AVX512(BaseType x, ManagedSelection &msel) const {
#if defined(__AVX512F__)
    static_assert(std::is_same<CodeType, uint8_t>::value, "Only uint8_t code AVX-512 supported");

    if (!endpoints_.empty() && x >= max_data_) {
        return 0;
    }

    msel.bitmask.clear();
    SelectionVector &sel = msel.Selection();
    sel.Initialize(base_data_.size());
    
    
    if (!endpoints_.empty() && x < min_data_) {
        sel_t *data = sel.data();
        size_t m = base_data_.size();
        int16_t k = 0;
        for (; k + 16 <= m; k += 16) {
            __m512i idx = _mm512_set_epi32(k+15, k+14, k+13, k+12, k+11, k+10, k+9, k+8,
                                           k+7, k+6, k+5, k+4, k+3, k+2, k+1, k);
            _mm512_storeu_si512(reinterpret_cast<void*>(data + k), idx);
        }
        for (; k < m; ++k) {
            data[k] = k;
        }

        size_t mask_count = (m + 63) / 64;
        msel.bitmask.resize(mask_count, 0xFFFFFFFFFFFFFFFFULL);
        size_t remain = m % 64;
        if (remain != 0) {
            msel.bitmask[mask_count - 1] = (1ULL << remain) - 1;
        }

        msel.SetCount(m);
        return static_cast<int32_t>(m);
    }

    CodeType code_x = static_cast<CodeType>(
        std::distance(endpoints_.begin(), std::lower_bound(endpoints_.begin(), endpoints_.end(), x))
    );

    __m512i cmp = _mm512_set1_epi8(static_cast<char>(code_x));
    size_t i = 0, n = sketch_.size();
    idx_t cnt = 0;
    uint32_t offset;
    if(!need_check_base) {
        for (; i + 64 <= n; i += 64) {
            __m512i data = _mm512_loadu_si512(reinterpret_cast<const void*>(&sketch_[i]));
            
            __mmask64 gt_mask = _mm512_cmpgt_epu8_mask(data, cmp); 
            msel.bitmask.push_back(static_cast<uint64_t>(gt_mask));
            while (gt_mask != 0) {
                uint32_t pos = i + __builtin_ctzll(gt_mask);
                sel.set_index(cnt++,pos);
                gt_mask = gt_mask & (gt_mask - 1);
            }
        }
    }
    else {
        for (; i + 64 <= n; i += 64) {
            __m512i data = _mm512_loadu_si512(reinterpret_cast<const void*>(&sketch_[i]));
            
            __mmask64 gt_mask = _mm512_cmpgt_epu8_mask(data, cmp);
            __mmask64 eq_mask = _mm512_cmpeq_epu8_mask(data, cmp);
            __mmask64 temp_mask = eq_mask;
            while (temp_mask != 0) {
                offset = __builtin_ctzll(temp_mask);
                uint32_t pos = i + offset;
                if (base_data_[pos] <= x) {
                    eq_mask &= ~(1ULL << offset);
                }
                temp_mask = temp_mask & (temp_mask - 1);
            }
            __mmask64 mask = gt_mask | eq_mask;
            msel.bitmask.push_back(static_cast<uint64_t>(mask));
            while (mask != 0) {
                uint32_t pos = i + __builtin_ctzll(mask);
                sel.set_index(cnt++,pos);
                mask = mask & (mask - 1);
            }
        }
    }
    
    uint64_t tail_mask = 0;
    int tail_bits = 0;
    for (; i < n; ++i, ++tail_bits) {
        if (sketch_[i] > code_x || (sketch_[i] == code_x && base_data_[i] > x)) {
            sel.set_index(cnt++,i);
            tail_mask |= (1ULL << tail_bits);
        }   
    }
   
    msel.bitmask.push_back(tail_mask);

    msel.SetCount(cnt);
    return static_cast<int32_t>(cnt);
#else
    throw std::runtime_error("AVX-512 not supported");
#endif
}

template<typename BaseType, typename CodeType>
idx_t ColumnSketch<BaseType, CodeType>::evaluate_greaterthan_orequal_avx512(BaseType x, ManagedSelection &msel) const {
#if defined(__AVX512F__)
    static_assert(std::is_same<CodeType, uint8_t>::value, "Only uint8_t code AVX-512 supported");

    if (!endpoints_.empty() && x > max_data_) {
        return 0;
    }

    msel.bitmask.clear();
    SelectionVector &sel = msel.Selection();
    sel.Initialize(base_data_.size());

    if (!endpoints_.empty() && (x < min_data_ || (!need_check_base && x <= min_data_))) {
        sel_t *data = sel.data();
        size_t m = base_data_.size();
        int16_t k = 0;
        for (; k + 16 <= m; k += 16) {
            __m512i idx = _mm512_set_epi32(k+15, k+14, k+13, k+12, k+11, k+10, k+9, k+8,
                                           k+7, k+6, k+5, k+4, k+3, k+2, k+1, k);
            _mm512_storeu_si512(reinterpret_cast<void*>(data + k), idx);
        }
        for (; k < m; ++k) {
            data[k] = k;
        }

        size_t mask_count = (m + 63) / 64;
        msel.bitmask.resize(mask_count, 0xFFFFFFFFFFFFFFFFULL);
        size_t remain = m % 64;
        if (remain != 0) {
            msel.bitmask[mask_count - 1] = (1ULL << remain) - 1;
        }

        msel.SetCount(m);
        return static_cast<int32_t>(m);
    }

    CodeType code_x = static_cast<CodeType>(
        std::distance(endpoints_.begin(), std::lower_bound(endpoints_.begin(), endpoints_.end(), x))
    );

    __m512i cmp = _mm512_set1_epi8(static_cast<char>(code_x));
    size_t i = 0, n = sketch_.size();
    idx_t cnt = 0;
    uint32_t offset;
    if(!need_check_base) {
        for (; i + 64 <= n; i += 64) {
            __m512i data = _mm512_loadu_si512(reinterpret_cast<const void*>(&sketch_[i]));
            
            __mmask64 lt_mask = _mm512_cmpge_epu8_mask(data, cmp);
            msel.bitmask.push_back(static_cast<uint64_t>(lt_mask));
            while (lt_mask != 0) {
                uint32_t pos = i + __builtin_ctzll(lt_mask);
                sel.set_index(cnt++,pos);
                lt_mask = lt_mask & (lt_mask - 1);
            }
        }
    }
    else {
        for (; i + 64 <= n; i += 64) {
            __m512i data = _mm512_loadu_si512(reinterpret_cast<const void*>(&sketch_[i]));
            
            __mmask64 lt_mask = _mm512_cmpgt_epu8_mask(data, cmp);
            __mmask64 eq_mask = _mm512_cmpeq_epu8_mask(data, cmp);
            __mmask64 temp_mask = eq_mask;
            while (temp_mask != 0) {
                offset = __builtin_ctzll(temp_mask);
                uint32_t pos = i + offset;
                if (base_data_[pos] < x) {
                    eq_mask &= ~(1ULL << offset);
                }
                temp_mask = temp_mask & (temp_mask - 1);
            }
            __mmask64 mask = lt_mask | eq_mask;
            msel.bitmask.push_back(static_cast<uint64_t>(mask));
            while (mask != 0) {
                uint32_t pos = i + __builtin_ctzll(mask);
                sel.set_index(cnt++,pos);
                mask = mask & (mask - 1);
            }
        }
    }

    uint64_t tail_mask = 0;
    int tail_bits = 0;
    for (; i < n; ++i, ++tail_bits) {
        if (sketch_[i] >= code_x || (sketch_[i] == code_x && base_data_[i] >= x)) {
            sel.set_index(cnt++,i);
            tail_mask |= (1ULL << tail_bits);
        }   
    }
    msel.bitmask.push_back(tail_mask);


    msel.SetCount(cnt);
    return static_cast<int32_t>(cnt);
#else
    throw std::runtime_error("AVX-512 not supported");
#endif
}

template<typename BaseType, typename CodeType>
std::vector<uint32_t> ColumnSketch<BaseType, CodeType>::evaluate_equal(BaseType x) const {
    std::vector<uint32_t> result;
    CodeType code_x = static_cast<CodeType>(
        std::distance(endpoints_.begin(), std::lower_bound(endpoints_.begin(), endpoints_.end(), x))
    );
    for (size_t i = 0; i < sketch_.size(); ++i) {
        if (sketch_[i] == code_x && base_data_[i] == x)
            result.push_back(static_cast<uint32_t>(i));
    }
    return result;
}

template<typename BaseType, typename CodeType>
std::vector<uint32_t> ColumnSketch<BaseType, CodeType>::evaluate_between(BaseType x1, BaseType x2) const {
    std::vector<uint32_t> result;
    for (size_t i = 0; i < base_data_.size(); ++i) {
        if (base_data_[i] >= x1 && base_data_[i] <= x2)
            result.push_back(static_cast<uint32_t>(i));
    }
    return result;
}



struct BaseColumnSketch {
    virtual ~BaseColumnSketch() = default;
    virtual std::shared_ptr<BaseColumnSketch> Copy() const = 0;
};

template<typename BaseType, typename CodeType>
struct ColumnSketchWrapper : public BaseColumnSketch {
    ColumnSketch<BaseType, CodeType> impl;
    explicit ColumnSketchWrapper(const std::vector<BaseType>& data) : impl(data) {}

    std::shared_ptr<BaseColumnSketch> Copy() const override {
        return std::make_shared<ColumnSketchWrapper<BaseType, CodeType>>(*this);
    }
};

}