#include <connection.h>

#include <iostream>

#include <sheet.h>

using namespace Eigen;

using namespace animnet;

Connection::Connection(ptr<Sheet> source, ptr<Sheet> target, 
                       Type type)
  : _source(source), _target(target), type(type)
{
    uint rows;
    uint cols;
    switch (type) {
      case Type::Dense:
        rows = source->size();
        cols = target->size();
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

ptr<Sheet> Connection::source() const
{
  return _source;
}

ptr<Sheet> Connection::target() const
{
  return _target;
}

void Connection::forward() const
{
  const auto& sam { source()->activations() };
  auto& tam { target()->activations() };
  
  // convert to row vectors
  const Map<const RowVectorXd> sa(sam.data(), sam.size());
  Map<RowVectorXd> ta(tam.data(), tam.size());
 
  // standard ANN layer weight param application
  ta = sa*w + b.transpose();
}
