#ifndef SHEET_H
#define SHEET_H

#include <utils.h>

#include <vector>
#include <Eigen/Dense>


namespace animnet {

class Sheet {

public:
  Sheet(uint rows, uint cols=1);
  virtual ~Sheet();
  
  uint size() const;
  
  const Eigen::MatrixXd& activations() const;
  Eigen::MatrixXd& activations();
  
protected:
  Eigen::MatrixXd a; // activations

};



}

#endif
