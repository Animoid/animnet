#include <network.h>

#include <connection.h>
#include <sheet.h>

using namespace animnet;

Network::Network()
{
  
}

Network::~Network()
{
  
}

void Network::add(ptr<Connection> connection)
{
  connections.push_back(connection);
}

void Network::forward() const
{
  for(auto connection : connections) {
    connection->forward();
    connection->target()->activate();
  }
}
