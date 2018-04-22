#ifndef NETWORK_H
#define NETWORK_H

#include <utils.h>

#include <list>

namespace animnet {

class Connection;  
  
class Network {

public:
  Network();
  virtual ~Network();
  
  void add(ptr<Connection> connection);
  
protected:
  std::list<ptr<Connection>> connections;

};



}

#endif
