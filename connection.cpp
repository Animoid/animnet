#include <connection.h>

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
    b.resize(rows);
}

Connection::~Connection()
{
  
}

void Connection::forward() const
{
  const auto& sa { src->activations() };
  auto& ta { target->activations() };
  
  
}
