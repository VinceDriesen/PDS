#include <algorithm>
#include <iostream>
//#define NDEBUG
#include <assert.h>
#include <stdexcept>
#include "parameters.h"
#include "encrypt_cuda.h"

std::vector<uint8_t> encrypt_cuda(std::vector<uint8_t> plaintext, std::vector<std::vector<uint8_t> > roundkeys, bool decrypt = false);

Encrypt_cuda::Encrypt_cuda(std::string key) : Encrypt(key) {
}


std::vector<uint8_t> Encrypt_cuda::encrypt(std::vector<uint8_t> plaintext) {
	if (plaintext.size() % 1024 != 0)
		plaintext.resize((plaintext.size() / 1024 + 1) * 1024, 0);
	assert(plaintext.size() % 1024 == 0);
	return encrypt_cuda(plaintext, roundkeys);
}

std::vector<uint8_t> Encrypt_cuda::decrypt(std::vector<uint8_t> cipher) {
	assert(cipher.size() % 1024 == 0);

	if (cipher.size() == 0)
		throw std::invalid_argument("Cipher must have some data");

	std::vector<uint8_t> totalresult = encrypt_cuda(cipher, roundkeys, true);
	
	return totalresult;
}

bool Encrypt_cuda::unittest(std::string plaintext) {
	return true;
}
