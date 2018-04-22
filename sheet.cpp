#include <sheet.h>

using namespace Eigen;

using namespace animnet;

Sheet::Sheet(uint rows, uint cols, Activation activation)
  : a(rows,cols), activation(activation)
{
  
}

Sheet::~Sheet()
{
  
}

uint Sheet::size() const
{
  return a.rows()*a.cols();
}

const MatrixXd& Sheet::activations() const
{
  return a;
}

MatrixXd& Sheet::activations()
{
  return a;
}


void Sheet::activate()
{
  switch (activation) {
    case Activation::Identity: 
      break;
    case Activation::Tanh:
      a = tanh(a.array());
      break;
    default:
      throw std::runtime_error("unimplemented activation function");
  }
}
