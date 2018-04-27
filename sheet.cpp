#include <sheet.h>

using namespace Eigen;

using namespace animnet;

Sheet::Sheet(Index rows, Index cols, Activation activation)
  : a(rows,cols), activation(activation)
{
  
}

Sheet::~Sheet()
{
  
}

Index Sheet::size() const
{
  return a.rows()*a.cols();
}

Index Sheet::rows() const
{
  return a.rows();
}

Index Sheet::cols() const
{
  return a.cols();
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
      a = a.array().tanh();
      break;
    case Activation::Logistic:
      a = a.unaryExpr(std::ptr_fun(&logistic));
      break;
    case Activation::ReLU:
      a = a.unaryExpr(std::ptr_fun(&relu));
      break;
    case Activation::Softmax: {
      auto exps = a.array().exp();
      auto sum = exps.sum();
      a = exps / sum;
    } break;
    default:
      throw std::runtime_error("unimplemented activation function");
  }
}
