#include <sheet.h>

using namespace Eigen;

using namespace animnet;

Sheet::Sheet(uint rows, uint cols)
  : a(rows,cols)
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
