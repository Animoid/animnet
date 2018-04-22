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
  
  Connection(ptr<Sheet> source, ptr<Sheet> target, 
             Type type=Type::Dense);
  virtual ~Connection();
  
  ptr<Sheet> source() const;
  ptr<Sheet> target() const;
  
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
