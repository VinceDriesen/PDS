#include "encrypt.h"
#include <algorithm>
#include "parameters.h"
#include <stdexcept>

Encrypt::Encrypt(std::string key) : roundkeys(totalrounds) {
	if (key.size() == 0)
		throw std::invalid_argument("Key cannot be empty");
	std::vector<uint8_t> keyvec(key.length());
	std::transform(key.begin(), key.end(), keyvec.begin(), [](char c) { return static_cast<uint8_t>(c); });
	precalculateKeys(keyvec, totalrounds);
}

std::vector<uint8_t> Encrypt::encryptString(std::string plaintext) {
	if (plaintext.size() == 0)
		throw std::invalid_argument("Plaintext must have some data");

	size_t plaintextlength = plaintext.length();
	if (plaintextlength % 1024 != 0)
		plaintextlength += 1024 - (plaintextlength % 1024);
	std::vector<uint8_t> plaintextvec(plaintextlength, 0);
	std::transform(plaintext.begin(), plaintext.end(), plaintextvec.begin(), [](char c) { return static_cast<uint8_t>(c); });

	return encrypt(plaintextvec);
}

std::string Encrypt::decryptToString(std::vector<uint8_t> cipher) {
	std::vector<uint8_t> decr = decrypt(cipher);
	return std::string(decr.begin(), decr.end());
}

void Encrypt::precalculateKeys(std::vector<uint8_t> key, unsigned int rounds)
{
	const unsigned int whiteningRounds = 1278;

	uint32_t shiftRegister = 0x4DF6ACE1u;
	uint32_t bit;

	//Preshift
	for (unsigned int period = 0; period < key.size() + (whiteningRounds) * 512 * 8; ++period) {
		if (period < key.size()) {
			uint32_t currentKeyByte = key[period];
			shiftRegister ^= currentKeyByte;
		}
		//taps: 32,22,2,1
		bit = ((shiftRegister >> 0) ^ (shiftRegister >> 10) ^ (shiftRegister >> 30) ^ (shiftRegister >> 31)) & 1;
		shiftRegister = (shiftRegister >> 1) | (bit << 31);
	}

	//Derive key
	for (unsigned int round = 0; round < rounds; ++round) {
		std::vector<uint8_t> derivedKey(512);
		for (unsigned int period = 0; period < 512 * 8; ++period) {
			bit = ((shiftRegister >> 0) ^ (shiftRegister >> 10) ^ (shiftRegister >> 30) ^ (shiftRegister >> 31)) & 1;
			shiftRegister = (shiftRegister >> 1) | (bit << 31);
			if (period % 8 == 0)
				derivedKey[period / 8] = (uint8_t)(shiftRegister & 0x000000FFu);
		}
		roundkeys[round] = derivedKey;
	}
}
