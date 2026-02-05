#pragma once
#include "encrypt.h"
#include <stdint.h>
#include <vector>
#include <string>

class Encrypt {

protected:
	void precalculateKeys(std::vector<uint8_t> key, unsigned int rounds);
public:
	Encrypt(std::string key);
	virtual std::vector<uint8_t> encryptString(std::string plaintext);
	virtual std::vector<uint8_t> encrypt(std::vector<uint8_t> plaintext) = 0;
	virtual std::vector<uint8_t> decrypt(std::vector<uint8_t> cipher) = 0;
	virtual std::string decryptToString(std::vector<uint8_t> cipher);

	virtual bool unittest(std::string plaintext) = 0;

protected:
	std::vector<std::vector<uint8_t> > roundkeys;
};
