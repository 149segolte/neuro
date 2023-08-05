#include <stdint.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <execution>
#include <iostream>
#include <limits>
#include <random>
#include <ranges>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define MAP_SIZE 64
#define COORD_SIZE 6
#define POP_SIZE 512
#define STEPS 256
#define GENOME_SIZE 6
#define MUTATION_RATE 0.005
#define INNER_NEURONS 4
#define INPUT_NEURONS 11
#define OUTPUT_NEURONS 4
#define SCALE 8000

enum input_type {
	RANDOM = 0,
	OSCILLATOR = 1,
	AGE = 2,
	BLOCK_LR = 3,
	BLOCK_FORWARD = 4,
	POP_DENSITY = 5,
	POP_GRADIENT_LR = 6,
	POP_GRADIENT_FORWARD = 7,
	LOC_X = 8,
	LOC_Y = 9,
	LOC_WALL_NS = 10,
	LOC_WALL_EW = 11
};

enum output_type { FORWARD = 0, BACKWARD = 1, LEFT = 2, RIGHT = 3 };

typedef uint32_t gene;

int map[MAP_SIZE][MAP_SIZE];
int TIME = 0;
// Get a random seed from the OS entropy device, or whatever
std::random_device rd;
// Use the 64-bit Mersenne Twister 19937 generator and seed it with
// entropy.
std::mt19937_64 eng(rd());
// Define the distribution, by default it goes from 0 to MAX(unsigned
// long long) or what have you.
std::uniform_int_distribution<uint32_t> dist;

float initial_state[INPUT_NEURONS] = {0.0f};

struct Coord {
	unsigned int x : COORD_SIZE = 0;
	unsigned int y : COORD_SIZE = 0;
	unsigned int dir : 2;
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
		if (source_type == 0) {
			valid &= source < INPUT_NEURONS;
		} else {
			valid &= source < INNER_NEURONS;
		}
		valid &= sink_type == 0 || sink_type == 1;
		if (sink_type == 0) {
			valid &= sink < OUTPUT_NEURONS;
		} else {
			valid &= sink < INNER_NEURONS;
		}
		return valid;
	}
};

class NeuralNet {
 private:
	std::array<Connection, GENOME_SIZE> connections;
	std::array<float, OUTPUT_NEURONS> output;
	std::array<float, INPUT_NEURONS> current_state;

	float inner_compute(uint8_t id, bool reset = false) {
		static std::array<float, INNER_NEURONS> inner_state;
		if (reset) {
			inner_state.fill(-2);
		}
		float sum = 0;
		std::ranges::sort(connections, [](Connection &a, Connection &b) {
			return a.source_type <= b.source_type;
		});
		for (auto &c : connections) {
			if (c.sink_type == 1 && c.sink == id) {
				if (c.source_type == 0) {
					sum += current_state[c.source] * c.weight / SCALE;
				} else {
					if (inner_state[id] == -2) {
						inner_state[id] = tanh(sum);
					}
					if (inner_state[c.source] == -2) {
						sum += inner_compute(c.source) * c.weight / SCALE;
					}
				}
			}
		}
		return tanh(sum);
	}

	float compute(uint8_t id) {
		float sum = 0;
		for (auto &c : connections) {
			if (c.sink_type == 0 && c.sink == id) {
				if (c.source_type == 0) {
					sum += current_state[c.source] * c.weight / SCALE;
				} else {
					sum += inner_compute(c.source, true) * c.weight / SCALE;
				}
			}
		}
		return std::tanh(sum);
	}

 public:
	std::array<float, OUTPUT_NEURONS> get_output() { return output; }

	std::array<float, INPUT_NEURONS> get_state() { return current_state; }

	void set_genome(std::array<gene, GENOME_SIZE> genome) {
		for (int i = 0; i < GENOME_SIZE; i++) {
			connections[i].from_gene(genome[i]);
			if (connections[i].sink_type == 0) {
				output[connections[i].sink] = -2;
			}
		}
		for (int i = 0; i < INPUT_NEURONS; i++) {
			this->current_state[i] = initial_state[i];
		}
	}

	std::array<float, OUTPUT_NEURONS> step() {
		for (int i = 0; i < OUTPUT_NEURONS; i++) {
			if (output[i] == -2) {
				output[i] = compute(i);
			}
		}
		return output;
	}

	void update_state(float new_state[INPUT_NEURONS]) {
		for (int i = 0; i < INPUT_NEURONS; i++) {
			current_state[i] = new_state[i];
		}
	}
};

bool valid_gene(gene g) {
	Connection c;
	c.from_gene(g);
	return c.valid();
}

std::array<gene, GENOME_SIZE> new_genome() {
	std::array<gene, GENOME_SIZE> genome;
	for (int i = 0; i < GENOME_SIZE; i++) {
		gene g;
		do {
			g = dist(eng);
		} while (!valid_gene(g));
		genome[i] = g;
	}
	return genome;
}

class Cell {
 private:
	Coord loc;
	std::array<gene, GENOME_SIZE> genome;
	NeuralNet brain;

 public:
	Cell() { set_genome(new_genome()); }

	void debug() {
		std::cout << "Input: \n";
		auto input = brain.get_state();
		for (int i = 0; i < INPUT_NEURONS; i++) {
			std::cout << input[i] << " ";
		}
		std::cout << "\nOutput: \n";
		auto output = brain.get_output();
		for (int i = 0; i < OUTPUT_NEURONS; i++) {
			std::cout << output[i] << " ";
		}
	}

	Coord get_loc() { return loc; }

	void set_loc(uint16_t x, uint16_t y, uint8_t dir) {
		loc.x = x;
		loc.y = y;
		loc.dir = dir;
	}

	std::array<gene, GENOME_SIZE> get_genome() { return genome; }

	void set_genome(std::array<gene, GENOME_SIZE> genome) {
		map[loc.x][loc.y] = 0;
		int x, y;
		do {
			x = dist(eng) % MAP_SIZE;
			y = dist(eng) % MAP_SIZE;
		} while (map[x][y] == 1);
		map[x][y] = 1;
		set_loc(x, y, (dist(eng) % 4));
		this->genome = genome;
		this->brain.set_genome(genome);
	}

	void move(uint8_t dir) {
		Coord new_loc;
		new_loc.dir = dir;
		if (dir == 0) {
			new_loc.x = loc.x;
			new_loc.y = loc.y + 1;
		} else if (dir == 2) {
			new_loc.x = loc.x;
			new_loc.y = loc.y - 1;
		} else if (dir == 1) {
			new_loc.x = loc.x + 1;
			new_loc.y = loc.y;
		} else {
			new_loc.x = loc.x - 1;
			new_loc.y = loc.y;
		}
		bool valid = new_loc.x >= 0 && new_loc.x < MAP_SIZE && new_loc.y >= 0 &&
								 new_loc.y < MAP_SIZE && map[new_loc.x][new_loc.y] == 0;
		if (valid) {
			/* std::cout << "Move: " << loc.x << ", " << loc.y << " -> " <<
				 new_loc.x
								<< ", " << new_loc.y << std::endl; */
			map[loc.x][loc.y] = 0;
			set_loc(new_loc.x, new_loc.y, new_loc.dir);
			map[loc.x][loc.y] = 1;
		}
	}

	uint8_t step() {
		auto output = brain.step();
		int max_index = 0;
		for (int i = 0; i < OUTPUT_NEURONS; i++) {
			if (output[i] > output[max_index]) {
				max_index = i;
			}
		}
		switch (static_cast<output_type>(max_index)) {
			case output_type::LEFT:
				return (loc.dir + 3) % 4;
			case output_type::RIGHT:
				return (loc.dir + 1) % 4;
			case output_type::FORWARD:
				return loc.dir;
			case output_type::BACKWARD:
				return (loc.dir + 2) % 4;
			default:
				return loc.dir;
		}
	}

	void update_state(float new_state[INPUT_NEURONS]) {
		brain.update_state(new_state);
	}
};

void print_map(int map[MAP_SIZE][MAP_SIZE]) {
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

void update_state(std::array<Cell, POP_SIZE> &cells) {
	std::for_each(std::begin(cells), std::end(cells), [](Cell &cell) {
		float state[INPUT_NEURONS] = {0.0f};
		for (auto j : std::views::iota(0, INPUT_NEURONS)) {
			input_type input = static_cast<input_type>(j);
			Coord loc = cell.get_loc();
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
					if (loc.dir == 0 || loc.dir == 2) {
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
					if (loc.dir == 0 && loc.y < MAP_SIZE - 1) {
						state[j] = map[loc.x][loc.y + 1];
					} else if (loc.dir == 2 && loc.y > 0) {
						state[j] = map[loc.x][loc.y - 1];
					} else if (loc.dir == 1 && loc.x < MAP_SIZE - 1) {
						state[j] = map[loc.x + 1][loc.y];
					} else if (loc.dir == 3 && loc.x > 0) {
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
					if (loc.dir == 0) {
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
					} else if (loc.dir == 2) {
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
					} else if (loc.dir == 1) {
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
					} else if (loc.dir == 3) {
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
					if (loc.dir == 0) {
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
					} else if (loc.dir == 2) {
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
					} else if (loc.dir == 1) {
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
					} else if (loc.dir == 3) {
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
		cell.update_state(state);
	});
}

int main() {
	for (int i = 0; i < MAP_SIZE; i++) {
		for (int j = 0; j < MAP_SIZE; j++) {
			map[i][j] = 0;
		}
	}

	std::array<Cell, POP_SIZE> cells;

	print_map(map);

	float acc = 0, last_10_sum = 0, max = 0;
	std::vector<float> last_10;
	last_10.reserve(10);
	std::chrono::duration<double> elapsed = std::chrono::duration<double>::zero(),
																elapsed_10 =
																		std::chrono::duration<double>::zero();
	std::vector<std::chrono::duration<double>> last_10_elapsed;
	last_10_elapsed.reserve(10);
	int gen = 0;
	do {
		TIME = 0;
		std::chrono::high_resolution_clock::time_point begin =
				std::chrono::high_resolution_clock::now();
		for (auto i : std::views::iota(0, MAP_SIZE)) {
			for (auto j : std::views::iota(0, MAP_SIZE)) {
				map[i][j] = 0;
			}
		}
		do {
			update_state(cells);
			for (auto i : std::views::iota(0, POP_SIZE)) {
				uint8_t d = cells[i].step();
				cells[i].move(d);
			}

			// std::cout << "TIME: " << TIME << "\r" << std::flush;

			TIME++;
		} while (TIME < STEPS);

		const auto [ret, last] = std::ranges::remove_if(
				cells, [](Cell c) { return c.get_loc().x < (2 * MAP_SIZE / 3); });
		const auto valid = std::ranges::distance(cells.begin(), ret);
		acc = (float)valid / POP_SIZE;
		if (acc > max) {
			max = acc;
		}
		if (last_10.size() == 10) {
			last_10_sum -= last_10[0];
			last_10.erase(last_10.begin());
		}
		last_10_sum += acc;
		last_10.push_back(acc);

		std::vector<std::array<gene, GENOME_SIZE>> new_genes;
		for (auto cell = cells.begin(); cell != ret; cell++) {
			new_genes.push_back(cell->get_genome());
		}
		std::ranges::shuffle(new_genes, eng);

		size_t child_num = ceil(POP_SIZE / (acc / 2));
		std::vector<std::array<gene, GENOME_SIZE>> children;
		for (auto i = 0; i < valid; i += 2) {
			auto g1 = new_genes[i], g2 = new_genes[i + 1];
			std::array<gene, GENOME_SIZE * 2> box;
			std::ranges::copy(g1, box.begin());
			std::ranges::copy(g2, box.begin() + GENOME_SIZE);
			for (size_t j = 0; j < child_num; j++) {
				std::ranges::shuffle(box, eng);
				std::array<gene, GENOME_SIZE> child;
				std::ranges::copy(box.begin(), box.begin() + GENOME_SIZE,
													child.begin());
				size_t mut =
						std::uniform_int_distribution<size_t>(0, (1 / MUTATION_RATE))(eng);
				if (mut == 0) {
					size_t i = std::uniform_int_distribution<size_t>(0, GENOME_SIZE)(eng);
					do {
						size_t bit =
								std::uniform_int_distribution<size_t>(0, 8 * sizeof(gene))(eng);
						child[i] ^= 1 << (bit % (8 * sizeof(gene)));
					} while (!valid_gene(child[i]));
				}
				children.push_back(child);
			}
		}

		Cell sample = cells[0];
		std::ranges::shuffle(children, eng);
		std::array<std::array<gene, GENOME_SIZE>, POP_SIZE> new_pop;
		std::ranges::copy(children.begin(), children.begin() + POP_SIZE,
											new_pop.begin());
		for (auto i : std::views::iota(0, POP_SIZE)) {
			cells[i].set_genome(new_pop[i]);
		}

		gen++;
		elapsed = std::chrono::high_resolution_clock::now() - begin;
		if (last_10_elapsed.size() == 10) {
			elapsed_10 -= last_10_elapsed[0];
			last_10_elapsed.erase(last_10_elapsed.begin());
		}
		elapsed_10 += elapsed;
		last_10_elapsed.push_back(elapsed);

		if (gen % 4 == 0) {
			std::cout << "\E[H\E[J";
			std::cout << "GEN: " << gen << "\n";
			std::cout << "ACC: " << acc << " MAX: " << max
								<< " LAST 10 AVG: " << last_10_sum / 10 << "\n";
			std::cout << "CLOCK: " << elapsed.count()
								<< " LAST 10 AVG: " << elapsed_10.count() / 10 << "\n\n";
			Coord loc = sample.get_loc();
			std::cout << "Sample cell:\n"
								<< "X: " << loc.x << " Y: " << loc.y << "\n"
								<< "Dir: " << loc.dir << "\n";
			std::cout << "Genome:\n";
			for (auto i : std::views::iota(0, GENOME_SIZE)) {
				std::cout << std::bitset<8 * sizeof(gene)>(sample.get_genome()[i])
									<< "\n";
			}
			std::cout << "\n";
			sample.debug();
			// print_map(map);
		}
	} while (acc < 0.8);

	return 0;
}
