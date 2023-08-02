#include <stdint.h>

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;
#define MAP_SIZE 32
#define POP_SIZE 1000
#define STEPS 300
#define GENOME_SIZE 4
#define MUTATION_RATE 0.01
#define INNER_NEURONS 4
#define INPUT_NEURONS 21
#define OUTPUT_NEURONS 11
#define SCALE 10000

typedef pair<uint8_t, uint8_t> Coord;
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
	Connection connections[GENOME_SIZE];
	vector<pair<uint8_t, float>> output_neurons;
	float current_state[INPUT_NEURONS];

	NeuralNet(gene genome[GENOME_SIZE], float current_state[INPUT_NEURONS]) {
		for (int i = 0; i < GENOME_SIZE; i++) {
			connections[i].from_gene(genome[i]);
			if (connections[i].sink_type == 0) {
				output_neurons.push_back(
					make_pair((uint8_t)connections[i].sink, 0));
			}
		}
		for (int i = 0; i < INPUT_NEURONS; i++) {
			this->current_state[i] = current_state[i];
		}
	}

	float compute(uint8_t neuron_id) {
		float sum = 0;
		for (auto &c : connections) {
			if (c.sink_type == 0 && c.sink == neuron_id) {
				if (c.source_type == 0) {
					sum += current_state[c.source] * c.weight / SCALE;
				} else {
					sum += compute(c.source) * c.weight / SCALE;
				}
			}
		}
		return sum;
	}

	vector<pair<uint8_t, float>> step() {
		for (auto &neuron : output_neurons) {
			uint8_t neuron_id = neuron.first;
			auto compute = [&](Connection c) {
				if (c.sink_type == 0 && c.sink == neuron_id) {
					neuron.second = c.source_type == 0 ? current_state[c.source]
													   : compute(c.source);
				}
			};
			ranges::for_each(connections, compute);
		}
		return output_neurons;
	}
};

class Cell {
	Coord loc;
	gene genome[GENOME_SIZE];
};

void print(int map[MAP_SIZE][MAP_SIZE]) {
	for (int i = 0; i < MAP_SIZE; i++) {
		for (int j = 0; j < MAP_SIZE; j++) {
			cout << map[i][j] << " ";
		}
		cout << endl;
	}
}

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

	print(map);

	return 0;
}
