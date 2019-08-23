#include <benchmark/benchmark.h>
#include <numeric>
#include <algorithm>
#include <random>
size_t n = 10000000ul;

template<typename T>
class matrix_t {
    typedef T value_type;
    std::vector<T> data;
    size_t rows_;
    size_t cols_;

public:
    matrix_t(size_t rows, size_t cols) : data(rows * cols), rows_(rows), cols_(cols) {}
    matrix_t(const matrix_t& x) = default;
    matrix_t(matrix_t&& x) = default;

    matrix_t& operator=(const matrix_t& x) {
        data = x.data;
        rows_ = x.rows_;
        cols_ = x.cols_;
        return *this;
    }

    auto begin() {
        return data.begin();
    }

    auto begin() const {
        return data.begin();
    }

    size_t rows() const {
        return rows_;
    }

    size_t cols() const {
        return cols_;
    }


    matrix_t& operator=(matrix_t&& x) {
        data = std::move(x.data);
        rows_ = x.rows_;
        cols_ = x.cols_;
        return *this;
    }

    T& operator()(size_t i, size_t j) {
        return  data[i * cols_ + j];
    }

    const T& operator()(size_t i, size_t j) const {
        return  data[i * cols_ + j];
    }


    friend
    inline
    matrix_t operator*(const matrix_t& x, const matrix_t& y) {
        matrix_t z(x.rows(), y.cols());
        assert(x.cols() == y.rows());
        for (size_t i = 0; i < x.rows(); ++i) {
            for (size_t j = 0; j < y.cols(); ++j) {
               for (size_t k = 0; k < y.rows(); ++k) {
                   z(i, j) += x(i, k) * y(j, k);
               }
            }
        }
        return z;
    }
};

inline
matrix_t<double> matrix_multiply1(const matrix_t<double>& x, const matrix_t<double>& y) {
    matrix_t<double> z(x.rows(), y.cols());
    assert(x.cols() == y.rows());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t k = 0; k < y.rows(); ++k) {
            for (size_t j = 0; j < y.cols(); ++j) {
               z(i, j) += x(i, k) * y(j, k);
           }
        }
    }
    return z;
}
template<typename T>
T transpose(const T& x) {
    T y(x.cols(), x.rows());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            y(j, i) = x(i, j);
        }
    }
    return y;
}

inline
matrix_t<double> matrix_multiply_by_transposed(const matrix_t<double>& x, const matrix_t<double>& y) {
        matrix_t<double> z(x.rows(), y.cols());
        assert(x.cols() == y.rows());
        for (size_t i = 0; i < x.rows(); ++i) {
            for (size_t j = 0; j < y.cols(); ++j) {
               auto ith_row_of_x = x.begin() + i * x.cols();
               auto jth_row_of_y = y.begin() + j * y.cols();
               z(i, j) = std::inner_product(ith_row_of_x, ith_row_of_x + x.cols(),
                                            jth_row_of_y, double(0.0));
            }
        }
        return z;
}




static void BM_inner_product_double(benchmark::State& state) {
  // Perform setup here
  std::vector<double> a(n);
  std::random_device rd;
  std::generate(a.begin(), a.end(), [&]() { return double(rd()); });
  std::vector<double> b(n);
  std::generate(b.begin(), b.end(), [&]() { return double(rd()); });

  for (auto _ : state) {
    volatile double result = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
  }
}

static void BM_inner_product_float(benchmark::State& state) {
  // Perform setup here
  size_t n = 10000000ul;
  std::vector<float> a(n);
  std::random_device rd;
  std::generate(a.begin(), a.end(), [&]() { return float(rd()); });
  std::vector<float> b(n);
  std::generate(b.begin(), b.end(), [&]() { return float(rd()); });


  for (auto _ : state) {
    volatile float result = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
  }
}

static void BM_innert_reduce_sum(benchmark::State& state) {
  size_t n = 10000000ul;
  std::vector<int> a(n);
  std::vector<int> b(n);
  std::vector<int> c(n);

  for (auto _ : state) {
      // c = [f(x, y) for x, y in zip(a, b)]
      volatile auto _abcd = std::transform(a.begin(), a.end(), b.begin(), c.begin(), [](auto x, auto y) {
        return x + y;
     });
  }
}

static void BM_innert_reduce_product(benchmark::State& state) {
  size_t n = 10000000ul;
  std::vector<int> a(n);
  std::vector<int> b(n);
  std::vector<int> c(n);

  for (auto _ : state) {
      // c = [f(x, y) for x, y in zip(a, b)]
      volatile auto _1 = std::transform(a.begin(), a.end(), b.begin(), c.begin(), [](auto x, auto y) {
        return x * y;
     });
  }
}

static void BM_mm0(benchmark::State& state) {
  size_t n = 4000ul;
  matrix_t<double> x(n, n);
  matrix_t<double> y(n, n);

  for (auto _ : state) {
      volatile auto result = x * y;
  }
}
static void BM_mm1(benchmark::State& state) {
  size_t n = 4000ul;
  matrix_t<double> x(n, n);
  matrix_t<double> y(n, n);

  for (auto _ : state) {
      volatile auto result = matrix_multiply1(x, y);
  }
}

static void BM_mm2(benchmark::State& state) {
  size_t n = 4000ul;
  matrix_t<double> x(n, n);
  matrix_t<double> y(n, n);

auto transposed = transpose(y);
  for (auto _ : state) {
      volatile auto result = matrix_multiply_by_transposed(x, transposed);
  }
}


// Register the function as a benchmark
//BENCHMARK(BM_inner_product_float);
//BENCHMARK(BM_inner_product_double);
//BENCHMARK(BM_innert_reduce_sum);
//BENCHMARK(BM_innert_reduce_product);
BENCHMARK(BM_mm0);
BENCHMARK(BM_mm1);
BENCHMARK(BM_mm2);
// Run the benchmark
BENCHMARK_MAIN();
