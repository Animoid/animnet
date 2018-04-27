#include <connection.h>

#include <iostream>

#include <sheet.h>

using namespace Eigen;

using namespace animnet;

Connection::Connection(ptr<Sheet> source, ptr<Sheet> target, 
                       Type type)
  : _source(source), _target(target), type(type)
{
    Index rows;
    Index cols;
    switch (type) {
      case Type::Dense:
        rows = source->size();
        cols = target->size();
        break;
      case Type::Convolution:
        rows = cols = 2; // default
        break;
      default:
        throw std::runtime_error("unsupported Type");
    }
    w = MatrixXd::Zero(rows,cols);
    b = VectorXd::Zero(cols);
}

Connection::~Connection()
{
  
}

void Connection::setConvolutionParams(Index width, Index height, uint numFilters, std::string padding)
{
  assert(type == Type::Convolution);
  assert(numFilters == 1); // only 1 supported for now
  
  if (padding != "valid")
    throw std::runtime_error("only valid padding supported"); // TODO: support "same" (cf TensorFlow)

  if (width % 2 != 1)
    throw std::runtime_error("Convolution kernel width must be odd");

  if (height % 2 != 1)
    throw std::runtime_error("Convolution kernel height must be odd");

  // must be compatible with the source & target dimensions
  if (target()->cols() != source()->cols() - (width-1))
    throw std::runtime_error("Convolution width "+std::to_string(width)+" incompatible with source width "+std::to_string(source()->cols())+" and target width "+std::to_string(target()->cols()));

  if (target()->rows() != source()->rows() - (height-1))
    throw std::runtime_error("Convolution height "+std::to_string(height)+" incompatible with source height "+std::to_string(source()->rows())+" and target height "+std::to_string(target()->rows()));
  
  w = MatrixXd::Zero(width, height);
  b = VectorXd::Zero(0); // unused
}

ptr<Sheet> Connection::source() const
{
  return _source;
}

ptr<Sheet> Connection::target() const
{
  return _target;
}

Eigen::MatrixXd& Connection::weight()
{
  return w;
}
const Eigen::MatrixXd& Connection::weight() const
{
  return w;
}
  
Eigen::VectorXd& Connection::bias()
{
  return b;
}
const Eigen::VectorXd& Connection::bias() const
{
  return b;
}

void Connection::forward() const
{
  const auto& sam { source()->activations() };
  auto& tam { target()->activations() };
 
  if (type == Type::Dense) {
    // convert to row vectors
    const Map<const RowVectorXd> sa(sam.data(), sam.size());
    Map<RowVectorXd> ta(tam.data(), tam.size());

    // standard ANN layer weight param application
    ta = sa*w + b.transpose();
    
  } else if (type == Type::Convolution) {
    Index sw { source()->cols() }; // source dims
    Index sh { source()->rows() };
    Index tw { target()->cols() }; // target dims
    Index th { target()->rows() };
    Index kw { w.cols() };  // kernel dims
    Index kh { w.rows() };
    
    assert(tw == sw - kw + 1);
    assert(th == sh - kh + 1);
    
    const auto& s { source()->activations() };
    auto& t { target()->activations() };

    // scan kernel over source, without overhang (valid padding)
    // (kernel centered at x,y in source)
    Index padw = (kw-1)/2;
    Index padh = (kh-1)/2;
    for(Index y=padh; y<sh - padh; y++) {
      for(Index x=padw; x<sw - padw; x++) {
        // pick out kernel shaped block from source
        auto csrc { s.block(x-padw, y-padh, kw, kh) };
        // element-wise multiply with kernel & sum to scalar
        t(x-padw, y-padh) = (csrc.array() * w.array()).sum();
      }
    }
    
  } // convolution
}
