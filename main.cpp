#include <stdint.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <random>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#define MAP_SIZE 32
#define POP_SIZE 100
#define STEPS 300
#define GENOME_SIZE 4
#define MUTATION_RATE 0.01
#define INNER_NEURONS 4
#define INPUT_NEURONS 21
#define OUTPUT_NEURONS 11
#define SCALE 10000

typedef std::pair<uint8_t, uint8_t> Coord;
typedef uint32_t gene;

struct Connection {
	unsigned int source_type : 1;
	unsigned int source : 7;
	unsigned int sink_type : 1;
	unsigned int sink : 7;
	int weight : 16;

	void from_gene(gene g) {
		source_type = (g >> 31) & 1;
		source = (g >> 24) & 127;
		sink_type = (g >> 23) & 1;
		sink = (g >> 16) & 127;
		weight = g & 65535;
	}

	gene to_gene() {
		gene g = 0;
		g |= (source_type & 1) << 31;
		g |= (source & 127) << 24;
		g |= (sink_type & 1) << 23;
		g |= (sink & 127) << 16;
		g |= weight & 65535;
		return g;
	}

	bool valid() {
		bool valid = true;
		valid &= source_type == 0 || source_type == 1;
		valid &= source < INPUT_NEURONS + INNER_NEURONS;
		valid &= sink_type == 0 || sink_type == 1;
		valid &= sink < INNER_NEURONS + OUTPUT_NEURONS;
		return valid;
	}
};

class NeuralNet {
   private:
	std::array<Connection, GENOME_SIZE> connections;
	std::vector<std::pair<uint8_t, float>> output_neurons;
	float current_state[INPUT_NEURONS];

	float compute(uint8_t neuron_id) {
		float sum = 0;
		std::vector<int16_t> self_connections;
		for (auto &c : connections) {
			if (c.sink_type == 0 && c.sink == neuron_id) {
				if (c.source_type == 0) {
					sum += current_state[c.source] * c.weight / SCALE;
				} else {
					if (c.source == neuron_id) {
						self_connections.push_back(c.weight);
					} else {
						sum += compute(c.source) * c.weight / SCALE;
					}
				}
			}
		}
		for (auto &c : self_connections) {
			sum = c * sum / SCALE;
		}
		return sum;
	}

   public:
	NeuralNet(std::array<gene, GENOME_SIZE> genome,
			  float initial_state[INPUT_NEURONS]) {
		for (int i = 0; i < GENOME_SIZE; i++) {
			connections[i].from_gene(genome[i]);
			if (connections[i].sink_type == 0) {
				output_neurons.push_back(
					std::make_pair((uint8_t)connections[i].sink, 0));
			}
		}
		for (int i = 0; i < INPUT_NEURONS; i++) {
			this->current_state[i] = initial_state[i];
		}
	}

	std::vector<std::pair<uint8_t, float>> step() {
		for (auto &neuron : output_neurons) {
			uint8_t neuron_id = neuron.first;
			auto res = [&](Connection c) {
				if (neuron.second == 0 & c.sink_type == 0 &&
					c.sink == neuron_id) {
					neuron.second = compute(neuron_id);
				}
			};
			std::for_each(connections.cbegin(), connections.cend(), res);
		}
		return output_neurons;
	}

	void update_state(float new_state[INPUT_NEURONS]) {
		for (int i = 0; i < INPUT_NEURONS; i++) {
			current_state[i] = new_state[i];
		}
	}
};

class Cell {
   private:
	Coord loc;
	std::array<gene, GENOME_SIZE> genome;
	NeuralNet *brain;

   public:
	Cell(Coord loc, std::array<gene, GENOME_SIZE> genome,
		 float state[INPUT_NEURONS]) {
		this->loc = loc;
		for (int i = 0; i < GENOME_SIZE; i++) {
			this->genome[i] = genome[i];
		}
		this->brain = new NeuralNet(genome, state);
	}

	void step() {
		auto res = brain->step();
		for (auto &neuron : res) {
			if (neuron.first == 0) {
				// move
			} else if (neuron.first == 1) {
				// eat
			} else if (neuron.first == 2) {
				// attack
			} else if (neuron.first == 3) {
				// mate
			}
		}
	}

	void update_state(float new_state[INPUT_NEURONS]) {
		brain->update_state(new_state);
	}
};

void print_map(int map[MAP_SIZE][MAP_SIZE]) {
	for (int i = 0; i < MAP_SIZE; i++) {
		for (int j = 0; j < MAP_SIZE; j++) {
			if (map[i][j] == 1) {
				std::cout << "X ";
			} else {
				std::cout << "  ";
			}
		}
		std::cout << std::endl;
	}
}

bool valid_gene(gene g) {
	Connection c;
	c.from_gene(g);
	return c.valid();
}

std::array<gene, GENOME_SIZE> genome(
	std::uniform_int_distribution<uint32_t> &dist, std::mt19937_64 &eng) {
	std::array<gene, GENOME_SIZE> genome;
	for (int i = 0; i < GENOME_SIZE; i++) {
		gene g;
		do {
			g = dist(eng);
		} while (valid_gene(g));
		genome[i] = g;
	}
	return genome;
}

float state[INPUT_NEURONS] = {0.0f};

int main() {
	// Get a random seed from the OS entropy device, or whatever
	std::random_device rd;
	// Use the 64-bit Mersenne Twister 19937 generator and seed it with
	// entropy.
	std::mt19937_64 eng(rd());
	// Define the distribution, by default it goes from 0 to MAX(unsigned
	// long long) or what have you.
	std::uniform_int_distribution<uint32_t> dist;

	int map[MAP_SIZE][MAP_SIZE];
	for (int i = 0; i < MAP_SIZE; i++) {
		for (int j = 0; j < MAP_SIZE; j++) {
			map[i][j] = 0;
		}
	}

	std::array<Cell *, POP_SIZE> cells;
	for (auto i : std::views::iota(1, POP_SIZE)) {
		uint8_t x, y;
		do {
			x = dist(eng) % MAP_SIZE;
			y = dist(eng) % MAP_SIZE;
		} while (map[x][y] == 1);
		cells[i] = new Cell(Coord(x, y), genome(dist, eng), state);
		map[x][y] = 1;
	}

	print_map(map);

	for (auto i : std::views::iota(1, POP_SIZE)) {
		float state[INPUT_NEURONS] = {0.0f};
		cells[i]->update_state(state);
	}

	for (auto i : std::views::iota(1, POP_SIZE)) {
		delete cells[i];
	}
	return 0;
}
