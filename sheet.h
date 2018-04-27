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
    Logistic,
    ReLU,
    Softmax
  };

  Sheet(Eigen::Index rows, Eigen::Index cols, Activation activation = Activation::Identity);
  virtual ~Sheet();
    
  Eigen::Index rows() const;
  Eigen::Index cols() const;
  Eigen::Index size() const;
  
  void activate();
  
  const Eigen::MatrixXd& activations() const;
  Eigen::MatrixXd& activations();
  
protected:
  Activation activation;
  Eigen::MatrixXd a; // activations

};



}

#endif
