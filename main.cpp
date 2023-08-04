#include <stdint.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <ranges>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define MAP_SIZE 32
#define POP_SIZE 100
#define STEPS 300
#define GENOME_SIZE 4
#define MUTATION_RATE 0.01
#define INNER_NEURONS 4
#define INPUT_NEURONS 11
#define OUTPUT_NEURONS 4
#define SCALE 10000

enum input_type {
	RANDOM,
	OSCILLATOR,
	AGE,
	BLOCK_LR,
	BLOCK_FORWARD,
	POP_DENSITY,
	POP_GRADIENT_LR,
	POP_GRADIENT_FORWARD,
	LOC_X,
	LOC_Y,
	LOC_WALL_NS,
	LOC_WALL_EW
};

enum output_type { FORWARD, BACKWARD, LEFT, RIGHT };

enum Direction { NORTH, EAST, SOUTH, WEST };

typedef uint32_t gene;

int map[MAP_SIZE][MAP_SIZE];
int TIME = 0;

struct Coord {
	int x;
	int y;
	Direction dir;
};

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
				if (neuron.second == 0 & c.sink_type == 0 && c.sink == neuron_id) {
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

	Coord get_loc() { return loc; }

	void move(Direction dir) {
		Coord *new_loc;
		if (dir == Direction::NORTH) {
			new_loc = new Coord(loc.x, loc.y + 1, Direction::NORTH);
		} else if (dir == Direction::SOUTH) {
			new_loc = new Coord(loc.x, loc.y - 1, Direction::SOUTH);
		} else if (dir == Direction::EAST) {
			new_loc = new Coord(loc.x + 1, loc.y, Direction::EAST);
		} else {
			new_loc = new Coord(loc.x - 1, loc.y, Direction::WEST);
		}
		bool valid = new_loc->x >= 0 && new_loc->x < MAP_SIZE && new_loc->y >= 0 &&
								 new_loc->y < MAP_SIZE && map[new_loc->x][new_loc->y] == 0;
		if (valid) {
			map[loc.x][loc.y] = 0;
			loc = *new_loc;
			map[new_loc->x][new_loc->y] = 1;
		}
	}

	Direction step() {
		std::cout << "Cell at " << (int)loc.x << ", " << (int)loc.y << std::endl;
		auto res = brain->step();
		std::array<float, OUTPUT_NEURONS> output;
		for (auto &neuron : res) {
			output[neuron.first] = neuron.second;
		}
		int max_index = 0;
		for (int i = 0; i < OUTPUT_NEURONS; i++) {
			if (output[i] > output[max_index]) {
				max_index = i;
			}
		}
		switch (static_cast<output_type>(max_index)) {
			case output_type::LEFT:
				return (Direction)((int)loc.dir - 1);
			case output_type::RIGHT:
				return (Direction)((int)loc.dir + 1);
			case output_type::FORWARD:
				return loc.dir;
			case output_type::BACKWARD:
				return (Direction)((int)loc.dir + 2);
		}
	}

	void update_state(float new_state[INPUT_NEURONS]) {
		brain->update_state(new_state);
	}
};

void print_map(int map[MAP_SIZE][MAP_SIZE]) {
	system("clear||cls");
	for (int i = MAP_SIZE - 1; i >= 0; i--) {
		for (int j = 0; j < MAP_SIZE; j++) {
			if (map[j][i] == 1) {
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

void update_state(std::array<Cell *, POP_SIZE> &cells,
									std::uniform_int_distribution<uint32_t> &dist,
									std::mt19937_64 &eng) {
	for (auto i : std::views::iota(0, POP_SIZE)) {
		float state[INPUT_NEURONS] = {0.0f};
		for (auto j : std::views::iota(0, INPUT_NEURONS)) {
			input_type input = static_cast<input_type>(j);
			Coord loc = cells[i]->get_loc();
			int left = 0, right = 0, front = 0, back = 0;
			int sum = 0;
			switch (input) {
				case input_type::RANDOM:
					state[j] = (int32_t)(dist(eng) - UINT16_MAX) / (float)INT32_MAX;
					break;
				case input_type::OSCILLATOR:
					state[j] = sin(TIME / 1.0f);
					break;
				case input_type::AGE:
					state[j] = TIME / (float)STEPS;
					break;
				case input_type::LOC_X:
					state[j] = loc.x / (float)MAP_SIZE;
				case input_type::LOC_Y:
					state[j] = loc.y / (float)MAP_SIZE;
				case input_type::BLOCK_LR:
					sum = 0;
					if (loc.dir == Direction::NORTH || loc.dir == Direction::SOUTH) {
						if (loc.x > 0) {
							sum += map[loc.x - 1][loc.y];
						}
						if (loc.x < MAP_SIZE - 1) {
							sum += map[loc.x + 1][loc.y];
						}
						state[j] = sum / 2.0f;
					} else {
						if (loc.y > 0) {
							sum += map[loc.x][loc.y - 1];
						}
						if (loc.y < MAP_SIZE - 1) {
							sum += map[loc.x][loc.y + 1];
						}
						state[j] = sum / 2.0f;
					}
					break;
				case input_type::BLOCK_FORWARD:
					if (loc.dir == Direction::NORTH && loc.y < MAP_SIZE - 1) {
						state[j] = map[loc.x][loc.y + 1];
					} else if (loc.dir == Direction::SOUTH && loc.y > 0) {
						state[j] = map[loc.x][loc.y - 1];
					} else if (loc.dir == Direction::EAST && loc.x < MAP_SIZE - 1) {
						state[j] = map[loc.x + 1][loc.y];
					} else if (loc.dir == Direction::WEST && loc.x > 0) {
						state[j] = map[loc.x - 1][loc.y];
					} else {
						state[j] = 1;
					}
					break;
				case input_type::LOC_WALL_EW:
					if (loc.x > MAP_SIZE / 2) {
						state[j] = 2 - ((loc.x * 2) / (float)MAP_SIZE);
					} else {
						state[j] = ((loc.x * 2) / (float)MAP_SIZE);
					}
					break;
				case input_type::LOC_WALL_NS:
					if (loc.y > MAP_SIZE / 2) {
						state[j] = 2 - ((loc.y * 2) / (float)MAP_SIZE);
					} else {
						state[j] = ((loc.y * 2) / (float)MAP_SIZE);
					}
					break;
				case input_type::POP_DENSITY:
					sum = 0;
					if (loc.x - 1 >= 0 && loc.y - 1 >= 0 && loc.x + 1 < MAP_SIZE &&
							loc.y + 1 < MAP_SIZE) {
						sum += map[loc.x - 1][loc.y - 1];
						sum += map[loc.x][loc.y - 1];
						sum += map[loc.x + 1][loc.y - 1];
						sum += map[loc.x + 1][loc.y];
						sum += map[loc.x + 1][loc.y + 1];
						sum += map[loc.x][loc.y + 1];
						sum += map[loc.x - 1][loc.y + 1];
						sum += map[loc.x - 1][loc.y];
					} else {
						if (loc.x == 0) {
							if (loc.y == 0) {
								sum += map[loc.x + 1][loc.y];
								sum += map[loc.x + 1][loc.y + 1];
								sum += map[loc.x][loc.y + 1];
							} else if (loc.y == MAP_SIZE - 1) {
								sum += map[loc.x][loc.y - 1];
								sum += map[loc.x + 1][loc.y - 1];
								sum += map[loc.x + 1][loc.y];
							} else {
								sum += map[loc.x][loc.y - 1];
								sum += map[loc.x + 1][loc.y - 1];
								sum += map[loc.x + 1][loc.y];
								sum += map[loc.x + 1][loc.y + 1];
								sum += map[loc.x][loc.y + 1];
							}
						} else if (loc.x == MAP_SIZE - 1) {
							if (loc.y == 0) {
								sum += map[loc.x - 1][loc.y];
								sum += map[loc.x - 1][loc.y + 1];
								sum += map[loc.x][loc.y + 1];
							} else if (loc.y == MAP_SIZE - 1) {
								sum += map[loc.x][loc.y - 1];
								sum += map[loc.x - 1][loc.y - 1];
								sum += map[loc.x - 1][loc.y];
							} else {
								sum += map[loc.x][loc.y - 1];
								sum += map[loc.x - 1][loc.y - 1];
								sum += map[loc.x - 1][loc.y];
								sum += map[loc.x - 1][loc.y + 1];
								sum += map[loc.x][loc.y + 1];
							}
						} else {
							if (loc.y == 0) {
								sum += map[loc.x - 1][loc.y];
								sum += map[loc.x - 1][loc.y + 1];
								sum += map[loc.x][loc.y + 1];
								sum += map[loc.x + 1][loc.y + 1];
								sum += map[loc.x + 1][loc.y];
							} else if (loc.y == MAP_SIZE - 1) {
								sum += map[loc.x - 1][loc.y];
								sum += map[loc.x - 1][loc.y - 1];
								sum += map[loc.x][loc.y - 1];
								sum += map[loc.x + 1][loc.y - 1];
								sum += map[loc.x + 1][loc.y];
							}
						}
					}
					state[j] = sum / 8.0f;
					break;
				case input_type::POP_GRADIENT_LR:
					if (loc.dir == Direction::NORTH) {
						for (int i = loc.x - 1; i >= 0; i--) {
							for (int j = 0; j < MAP_SIZE; j++) {
								left += map[i][j];
							}
						}
						for (int i = loc.x + 1; i < MAP_SIZE; i++) {
							for (int j = 0; j < MAP_SIZE; j++) {
								right += map[i][j];
							}
						}
					} else if (loc.dir == Direction::SOUTH) {
						for (int i = loc.x + 1; i >= 0; i--) {
							for (int j = 0; j < MAP_SIZE; j++) {
								left += map[i][j];
							}
						}
						for (int i = loc.x - 1; i < MAP_SIZE; i++) {
							for (int j = 0; j < MAP_SIZE; j++) {
								right += map[i][j];
							}
						}
					} else if (loc.dir == Direction::EAST) {
						for (int i = loc.y + 1; i >= 0; i--) {
							for (int j = 0; j < MAP_SIZE; j++) {
								left += map[j][i];
							}
						}
						for (int i = loc.y - 1; i < MAP_SIZE; i++) {
							for (int j = 0; j < MAP_SIZE; j++) {
								right += map[j][i];
							}
						}
					} else if (loc.dir == Direction::WEST) {
						for (int i = loc.y - 1; i >= 0; i--) {
							for (int j = 0; j < MAP_SIZE; j++) {
								left += map[j][i];
							}
						}
						for (int i = loc.y + 1; i < MAP_SIZE; i++) {
							for (int j = 0; j < MAP_SIZE; j++) {
								right += map[j][i];
							}
						}
					}
					state[j] = (float)(left - right) / POP_SIZE;
					break;
				case input_type::POP_GRADIENT_FORWARD:
					if (loc.dir == Direction::NORTH) {
						for (int i = loc.y + 1; i >= 0; i--) {
							for (int j = 0; j < MAP_SIZE; j++) {
								front += map[i][j];
							}
						}
						for (int i = loc.y - 1; i < MAP_SIZE; i++) {
							for (int j = 0; j < MAP_SIZE; j++) {
								back += map[i][j];
							}
						}
					} else if (loc.dir == Direction::SOUTH) {
						for (int i = loc.y - 1; i >= 0; i--) {
							for (int j = 0; j < MAP_SIZE; j++) {
								front += map[i][j];
							}
						}
						for (int i = loc.y + 1; i < MAP_SIZE; i++) {
							for (int j = 0; j < MAP_SIZE; j++) {
								back += map[i][j];
							}
						}
					} else if (loc.dir == Direction::EAST) {
						for (int i = loc.x + 1; i >= 0; i--) {
							for (int j = 0; j < MAP_SIZE; j++) {
								front += map[j][i];
							}
						}
						for (int i = loc.x - 1; i < MAP_SIZE; i++) {
							for (int j = 0; j < MAP_SIZE; j++) {
								back += map[j][i];
							}
						}
					} else if (loc.dir == Direction::WEST) {
						for (int i = loc.x - 1; i >= 0; i--) {
							for (int j = 0; j < MAP_SIZE; j++) {
								front += map[j][i];
							}
						}
						for (int i = loc.x + 1; i < MAP_SIZE; i++) {
							for (int j = 0; j < MAP_SIZE; j++) {
								back += map[j][i];
							}
						}
					}
					state[j] = (float)(back - front) / POP_SIZE;
					break;
			}
		}
		cells[i]->update_state(state);
	}
}

int main() {
	// Get a random seed from the OS entropy device, or whatever
	std::random_device rd;
	// Use the 64-bit Mersenne Twister 19937 generator and seed it with
	// entropy.
	std::mt19937_64 eng(rd());
	// Define the distribution, by default it goes from 0 to MAX(unsigned
	// long long) or what have you.
	std::uniform_int_distribution<uint32_t> dist;

	for (int i = 0; i < MAP_SIZE; i++) {
		for (int j = 0; j < MAP_SIZE; j++) {
			map[i][j] = 0;
		}
	}

	std::array<Cell *, POP_SIZE> cells;
	for (auto i : std::views::iota(0, POP_SIZE)) {
		uint8_t x, y;
		do {
			x = dist(eng) % MAP_SIZE;
			y = dist(eng) % MAP_SIZE;
		} while (map[x][y] == 1);
		Coord c(x, y, (Direction)(dist(eng) % 4));
		cells[i] = new Cell(c, genome(dist, eng), state);
		map[x][y] = 1;
	}

	print_map(map);

	do {
		std::this_thread::sleep_for(std::chrono::seconds(1));

		std::cout << "TIME: " << TIME << std::endl;
		update_state(cells, dist, eng);
		for (auto i : std::views::iota(0, POP_SIZE)) {
			Direction d = cells[i]->step();
			std::cout << "Cell " << i << std::endl;
			cells[i]->move(d);
		}

		print_map(map);
		TIME++;
	} while (TIME < STEPS);

	for (auto i : std::views::iota(0, POP_SIZE)) {
		delete cells[i];
	}
	return 0;
}
