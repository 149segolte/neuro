#include <stdint.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#define MAP_SIZE 32
#define POP_SIZE 1000
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

	void step(float state[INPUT_NEURONS]) {
		brain->update_state(state);
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
};

void print(int map[MAP_SIZE][MAP_SIZE]) {
	for (int i = 0; i < MAP_SIZE; i++) {
		for (int j = 0; j < MAP_SIZE; j++) {
			std::cout << map[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

std::array<gene, GENOME_SIZE> genome();
float state[INPUT_NEURONS] = {0.0f};

int main() {
	int map[MAP_SIZE][MAP_SIZE];
	for (int i = 0; i < MAP_SIZE; i++) {
		for (int j = 0; j < MAP_SIZE; j++) {
			map[i][j] = 0;
			if (i == 0 || j == 0 || i == MAP_SIZE - 1 || j == MAP_SIZE - 1) {
				map[i][j] = 1;
			}
		}
	}

	std::array<Cell *, POP_SIZE> cells;
	for (auto i : std::views::iota(1, POP_SIZE)) {
		int x = rand() % (MAP_SIZE - 2) + 1;
		int y = rand() % (MAP_SIZE - 2) + 1;
		cells[i] = new Cell(Coord(x, y), genome(), state);
	}

	return 0;
}
