#include <iostream>
#include <cmath>
#include <complex>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace py::literals;

const long int m = 1e6; // Frequency parameter
const double m2m1_rt = sqrt(m*m - 1.0);

double w (double x) { return m2m1_rt/(x*x + 1.0); };

double g (double x) { return 0; };

std::vector<std::complex<double> > bursty (std::vector<double> x) {
  double temp = 0;
  std::vector<std::complex<double> > res;
  for(auto it = std::begin(x); it != std::end(x); ++it) { temp += (*it)*(*it); }
  temp += 1.0;
  temp = sqrt(temp)/m;
  for(auto it = std::begin(x); it != std::end(x); ++it)
  {
    double matanx = m*atan(*it);
    std::complex<double> num = std::complex<double>(cos(matanx), sin(matanx));
    res.push_back(temp*num);
  }
  return res;
};

std::complex<double> bursty (double x)
{
  std::vector<double> x_vec = {x}; 
  return bursty(x_vec)[0];
};

std::vector<std::complex<double> > burstdy (std::vector<double> x) {
  double temp = 0;
  std::vector<std::complex<double> > res;
  for(auto it = std::begin(x); it != std::end(x); ++it) { temp += (*it)*(*it); }
  temp += 1.0;
  temp = 1.0/(m*sqrt(temp));
  for(auto it = std::begin(x); it != std::end(x); ++it)
  {
    std::complex<double> matanx = m*atan(*it);
    std::complex<double> num = std::complex<double>(*it, m) * cos(matanx) + std::complex<double>(-m, *it) * sin(matanx);
    res.push_back(temp*num);
  }
  return res;
};

std::complex<double> burstdy (double x)
{
  std::vector<double> x_vec = {x}; 
  return burstdy(x_vec)[0];
};

PYBIND11_EMBEDDED_MODULE(embeded, m)
{
  m.def("w", py::vectorize(w));
  m.def("g", py::vectorize(g));
};


int main()
{
  std::cout << "Demo for the burst example solved by Ricatti solver, thanks to the Pybind extension!" << std::endl;
  const double eps = 1e-8;
  const double epsh = 1e-12;
  double xi = -m;
  double xf = m;
  std::complex<double> yi = bursty(xi);
  std::complex<double> dyi = burstdy(xi);
  
  py::scoped_interpreter guard{}; 
  // py::module sys = py::module::import("sys");
  // py::print(sys.path);
  // sys.path.append(BPHF_DIR);
  py::module riccati = py::module::import("riccati");
  py::module em = py::module::import("embeded");
  // py::object riccati_solve = riccati.attr("solve"); // could use this instead of bound/explicit call below
  py::object sol = riccati.attr("solve")(em.attr("w"), em.attr("g"), xi, xf, yi, dyi, "eps"_a = eps, "epsh"_a = epsh, "n"_a = 32);

  auto sol_cpp = sol.cast<std::tuple<std::vector<double>, std::vector<std::complex<double> >, std::vector<std::complex<double> >,
                                     std::vector<bool>, std::vector<double>, std::vector<int> > >();

  std::cout << "Solution at x = " << std::get<0>(sol_cpp).back() << " is " << std::get<1>(sol_cpp).back() << "." << std::endl;
}