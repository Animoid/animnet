#include <connection.h>

#include <iostream>

#include <sheet.h>

using namespace Eigen;

using namespace animnet;

Connection::Connection(ptr<Sheet> src, ptr<Sheet> target, 
                       Type type, Activation activation)
  : src(src), target(target), type(type), activation(activation)
{
    uint rows;
    uint cols;
    switch (type) {
      case Type::Dense:
        rows = src->size();
        cols = target->size();
        break;
      default:
        throw std::runtime_error("unsupported Type");
    }
    w.resize(rows,cols);
    b.resize(cols);
}

Connection::~Connection()
{
  
}

void Connection::forward() const
{
  const auto& sam { src->activations() };
  auto& tam { target->activations() };
  
  // convert to row vectors
  const Map<const RowVectorXd> sa(sam.data(), sam.size());
  Map<RowVectorXd> ta(tam.data(), tam.size());
 
  // standard ANN layer weight param application
  auto lin = sa*w + b.transpose();
  
  // followed by element-wise activation function
  switch (activation) {
    case Activation::Tanh:
      ta = tanh(lin.array());
      break;
    default:
      throw std::runtime_error("unimplemented activation function");
  }
    
}
