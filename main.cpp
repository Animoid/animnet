#include <utils.h>
#include <sheet.h>
#include <connection.h>
#include <network.h>

#include <cstdint>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>

using namespace animnet;

int main(int argc, char** argv)
{
  auto readFile = [](auto name) -> auto {
    std::ifstream ifs(name, std::ios::binary);
    return std::string( (std::istreambuf_iterator<char>(ifs) ),
                        (std::istreambuf_iterator<char>()    ) );
        
  };
  
  auto stream_uint32 = [](uint8_t* data) -> uint32_t {
    return uint(data[3]<<0u) | uint(data[2]<<8u) | uint(data[1]<<16u) | uint(data[0]<<24u);
  };
  
  // [offset] [type]          [value]          [description] 
  // 0000     32 bit integer  0x00000803(2051) magic number 
  // 0004     32 bit integer  60000            number of images 
  // 0008     32 bit integer  28               number of rows 
  // 0012     32 bit integer  28               number of columns 
  auto images = readFile("data/mnist/train-images-idx3-ubyte");
  uint magic { stream_uint32(reinterpret_cast<uint8_t*>(&images.data()[0])) };
  assert(magic == 2051);

  uint count  { stream_uint32(reinterpret_cast<uint8_t*>(&images.data()[4])) };
  uint width  { stream_uint32(reinterpret_cast<uint8_t*>(&images.data()[8])) };
  uint height { stream_uint32(reinterpret_cast<uint8_t*>(&images.data()[12])) };
  
  std::cout << std::to_string(count) << " images " << std::to_string(width) << 'x' << std::to_string(height) << std::endl;
  assert(count==60000);
  assert(width==28);
  assert(height==28);
  
  // [offset] [type]          [value]          [description] 
  // 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
  // 0004     32 bit integer  60000            number of items 
  auto labels = readFile("data/mnist/train-labels-idx1-ubyte");

  auto showImage = [=](uint index) {
    uint offset = 16+(width*height*index);
    for(uint x=0; x<width; x++) {
      for(uint y=0; y<height; y++)
        std::cout << (uint8_t(images[offset+x*width+y]) > 128 ? "X" : " ");
      std::cout << std::endl;
    }
    std::cout << "label:" << std::to_string(int(labels[8+index])) << std::endl;
  };
  
  // Architecture
  // 1. Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
  // 2. Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
  // 3. Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
  // 4. Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
  // 5. Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element will be dropped during training)
  // 6. Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9).
  
  
  auto input = std::make_shared<Sheet>(width,height);
  
  auto conv1 = std::make_shared<Sheet>(24,24, Sheet::Activation::ReLU);
  auto wconv1 = std::make_shared<Connection>(input, conv1, Connection::Type::Convolution);
  wconv1->setConvolutionParams(5,5, "valid"); // TODO: switch to same

  auto output = std::make_shared<Sheet>(10,1, Sheet::Activation::Softmax);

  auto w2 = std::make_shared<Connection>(conv1, output, Connection::Type::Dense);

  auto network = std::make_shared<Network>();
  
  network->add(wconv1);
  network->add(w2);
  
  auto setInputImage = [=](uint index) {
    Eigen::MatrixXd& indata { input->activations() };
    uint offset = 16+(width*height*index);
    for(uint x=0; x<width; x++)
      for(uint y=0; y<height; y++) 
        indata(x,y) = images[offset+x*width+y] / 255.0;    
  };
  
  constexpr uint imageIndex = 12;

  showImage(imageIndex);
  
  // set image as input
  setInputImage(imageIndex);
  
  // classify image
  network->forward();
  
  std::cout << "Classifier:\n" << output->activations() << std::endl;
  
  exit(0);
}