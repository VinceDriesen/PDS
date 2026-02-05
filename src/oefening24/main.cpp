#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <fstream>
#include "encrypt_cuda.h"

void write_vector_to_file(const std::vector<uint8_t>& data, std::string filename)
{
	std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
	if (!ofs)
		throw std::runtime_error("error opening file");
	std::ostream_iterator<char> osi{ ofs };
	std::copy(data.begin(), data.end(), osi);
}

std::vector<uint8_t> read_vector_from_file(std::string filename)
{
	std::vector<uint8_t> data;
	std::ifstream ifs(filename, std::ios::in | std::ifstream::binary);
	if (!ifs)
		throw std::runtime_error("error opening file");
	std::istreambuf_iterator<char> iter(ifs);
	std::istreambuf_iterator<char> end;
	std::copy(iter, end, std::back_inserter(data));
	return data;
}

int main(int argc, char** argv) {

	if (argc != 5) {
		std::cout << "Usage: " << argv[0] << " inputfile outputfile key <decrypt/encrypt>" << std::endl;
		return 1;
	}

	std::vector<uint8_t> plaintext = read_vector_from_file(argv[1]);
	Encrypt_cuda encrypter(argv[3]);
	std::string mode = argv[4];
	std::vector<uint8_t> encrypted;
	if(mode == "encrypt")
		encrypted = encrypter.encrypt(plaintext);
	else if(mode == "decrypt")
		encrypted = encrypter.decrypt(plaintext);
	else
	{
		std::cout << "mode needs to be 'encrypt' or 'decrypt' but is " << mode << std::endl;
		return 1;
	}

	write_vector_to_file(encrypted, argv[2]);

	return 0;
}
