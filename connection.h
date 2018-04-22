#ifndef CONNECTION_H
#define CONNECTION_H

#include <utils.h>

#include <Eigen/Dense>


namespace animnet {

class Sheet;  
  
class Connection
{
public:

  enum class Type {
    Dense
  };
  
  enum class Activation {
    Tanh
  };

  Connection(ptr<Sheet> src, ptr<Sheet> target, 
             Type type=Type::Dense, Activation activation=Activation::Tanh);
  virtual ~Connection();
  
  void forward() const;
  
protected:
  ptr<Sheet> src;
  ptr<Sheet> target;
  
  Type type;
  Activation activation;
  
  Eigen::MatrixXd w; // weight 
  Eigen::VectorXd b; // bias

};



}

#endif
