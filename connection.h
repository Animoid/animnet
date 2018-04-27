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
    Dense,
    Convolution
  };
  
  Connection(ptr<Sheet> source, ptr<Sheet> target, 
             Type type=Type::Dense);
  virtual ~Connection();
  
  ptr<Sheet> source() const;
  ptr<Sheet> target() const;
  
  Eigen::MatrixXd& weight();
  const Eigen::MatrixXd& weight() const;
  
  Eigen::VectorXd& bias();
  const Eigen::VectorXd& bias() const;
  
  void setConvolutionParams(Eigen::Index width, Eigen::Index height, uint numFilters=1, std::string padding="valid");
  
  void forward() const;
  
protected:
  ptr<Sheet> _source;
  ptr<Sheet> _target;
  
  Type type;
  
  Eigen::MatrixXd w; // weight 
  Eigen::VectorXd b; // bias

};



}

#endif
