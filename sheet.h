#ifndef SHEET_H
#define SHEET_H

#include <utils.h>

#include <vector>
#include <Eigen/Dense>


namespace animnet {

class Sheet 
{
public:
  enum class Activation {
    Identity,
    Tanh,
    Softmax
  };

  Sheet(uint rows, uint cols, Activation activation = Activation::Identity);
  virtual ~Sheet();
    
  uint size() const;
  
  void activate();
  
  const Eigen::MatrixXd& activations() const;
  Eigen::MatrixXd& activations();
  
protected:
  Activation activation;
  Eigen::MatrixXd a; // activations

};



}

#endif
