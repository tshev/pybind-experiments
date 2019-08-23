#include <benchmark/benchmark.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
const size_t n = 10000000ul;
const size_t matrix_size = 1000ul;

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

    size_t size() const {
        return data.size();
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
                   z(i, j) += x(i, k) * y(k, j);
               }
            }
        }
        return z;
    }

    template<typename O>
    friend
    inline
    O& operator<<(O& out, const matrix_t& x) {
        for (size_t i = 0; i < x.rows(); ++i) {
            for (size_t j = 0; j < x.cols(); ++j) {
                out << x(i, j) << " ";
            }
            out << std::endl;
        }
        return out;
    }
};

inline
matrix_t<double> matrix_multiply1(const matrix_t<double>& x, const matrix_t<double>& y) {
    matrix_t<double> z(x.rows(), y.cols());
    assert(x.cols() == y.rows());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t k = 0; k < y.rows(); ++k) {
            for (size_t j = 0; j < y.cols(); ++j) {
               z(i, j) += x(i, k) * y(k, j);
           }
        }
    }
    return z;
}
/*
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
*/
template<typename T>
T transpose(const T& x) {
    T y(x.cols(), x.rows());
    auto y_first = y.begin();
    auto y_last = y_first + y.size();


    auto x_first = x.begin();
    auto x_last = x_first + x.size();

    while (y_first != y_last) {
        auto x_current = x_first;
        while (x_current != x_last) {
            *y_first = *x_current;
            x_current += x.cols();
            ++y_first;
        }
        ++x_first;
        ++x_last;
    }
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            y(j, i) = x(i, j);
        }
    }
    return y;
}

inline
matrix_t<double> matrix_multiply_by_transposed(const matrix_t<double>& x, const matrix_t<double>& y) {
        matrix_t<double> z(x.rows(), y.rows());
        auto x_row_last = x.begin() + x.size();
        auto y_row_last = y.begin() + y.size();
        auto z_first = z.begin();
        for(auto x_row = x.begin(); x_row != x_row_last; x_row += x.cols()) {
            for (auto y_row = y.begin(); y_row != y_row_last; y_row += y.cols()) {
               *z_first = std::inner_product(x_row, x_row + x.cols(), y_row, double(0.0));
            }
            ++z_first;
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
  std::vector<float> a(n);
  std::random_device rd;
  std::generate(a.begin(), a.end(), [&]() { return float(rd()); });
  std::vector<float> b(n);
  std::generate(b.begin(), b.end(), [&]() { return float(rd()); });


  for (auto _ : state) {
    volatile float result = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
  }
}

static void BM_reduce_sum(benchmark::State& state) {
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

static void BM_reduce_product(benchmark::State& state) {
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
  size_t n = matrix_size;
  matrix_t<double> x(n, n);
  matrix_t<double> y(n, n);

  for (auto _ : state) {
      volatile auto result = x * y;
  }
}
static void BM_mm1(benchmark::State& state) {
  size_t n = matrix_size;
  matrix_t<double> x(n, n);
  matrix_t<double> y(n, n);

  for (auto _ : state) {
      volatile auto result = matrix_multiply1(x, y);
  }
}

static void BM_mm2(benchmark::State& state) {
  size_t n = matrix_size;
  matrix_t<double> x(n, n);
  matrix_t<double> y(n, n);

  for (auto _ : state) {
      auto transposed = transpose(x);
      volatile auto result = matrix_multiply_by_transposed(x, transposed);
  }
}


// Register the function as a benchmark
BENCHMARK(BM_inner_product_float);
BENCHMARK(BM_inner_product_double);
BENCHMARK(BM_reduce_sum);
BENCHMARK(BM_reduce_product);
BENCHMARK(BM_mm2);
BENCHMARK(BM_mm0);
BENCHMARK(BM_mm1);
// Run the benchmark
BENCHMARK_MAIN();
