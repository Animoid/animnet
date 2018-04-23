#ifndef UTILS_H
#define UTILS_H

#include <memory>
#include <algorithm>
#include <iostream>

namespace animnet {

typedef unsigned int uint;

template <typename T>
using ptr = std::shared_ptr<T>;

// Rectified Linear Unit function
inline double relu(double v) {
  return std::max(0.0,v);
}

}

#endif
