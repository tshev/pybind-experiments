#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

int add(int i, int j) {
    return i + j;
}

double dot_product(const pybind11::array_t<double>& x, const pybind11::array_t<double>& y) {
    if (x.ndim() != 1 || y.ndim() != 1) {
        throw std::runtime_error("x and y must be 1d arrays");
    }
    size_t x_length = x.shape(0);
    size_t y_length = y.shape(0);

    if (x_length != y_length) {
        throw std::runtime_error("size should be the same");
    }

    const double* y_data = y.data();
    const double* x_data = x.data();

    return std::inner_product(x_data, x_data + x_length, y_data, 0.0);
}

namespace py = pybind11;

PYBIND11_MODULE(python_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: python_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("mod", [](int x, int y) { return x % y; }, "my mod");
    m.def("dot_product", [](const pybind11::array_t<double>& x, const pybind11::array_t<double>& y) { return dot_product(x, y); }, "dot product");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
