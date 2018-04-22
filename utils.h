#ifndef UTILS_H
#define UTILS_H

#include <memory>
#include <iostream>

namespace animnet {

typedef unsigned int uint;

template <typename T>
using ptr = std::shared_ptr<T>;


}

#endif
