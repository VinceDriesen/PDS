#pragma once
#include "encrypt.h"
//#define NDEBUG
#include <stdint.h>
#include <vector>
#include <string>

class Encrypt_cuda : public Encrypt {

public:
	Encrypt_cuda(std::string key);
	virtual std::vector<uint8_t> encrypt(std::vector<uint8_t> plaintext);
	virtual std::vector<uint8_t> decrypt(std::vector<uint8_t> cipher);
	virtual bool unittest(std::string plaintext);
};
