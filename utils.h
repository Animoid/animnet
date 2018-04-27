#ifndef UTILS_H
#define UTILS_H

#include <memory>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace animnet {

typedef unsigned int uint;

template <typename T>
using ptr = std::shared_ptr<T>;

// Rectified Linear Unit function
inline double relu(double x) {
  return std::max(0.0,x);
}
  
// standard logistic function
inline double logistic(double x)
{
  return 1.0/(1.0+std::exp(-x));
}
  

}

#endif
