use leptos::*;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use strum::{EnumCount, FromRepr};
use wasm_bindgen::JsCast;
use web_sys::CanvasRenderingContext2d;

type Inner = u8;
const INNER_STATES: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumCount, FromRepr)]
enum Input {
    Random = 0,
    Oscillator,
    Age,
    BlockLR,
    BlockForward,
    BlockForwardLong,
    LocX,
    LocY,
    LastMoveX,
    LastMoveY,
    LocWallNS,
    LocWallEW,
    NearestWall,
    PopDensity,
    // PopGradientLR,
    // PopGradientForward,
    // PopGradientForwardLong,
    // GeneMatchForward,
    // PheromoneDensity,
    // PheromoneGradientLR,
    // PheromoneGradientForward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumCount, FromRepr)]
enum Output {
    MoveRandom = 0,
    MoveForward,
    MoveReverse,
    MoveLR,
    MoveEW,
    MoveNS,
    // SetProbeDistance,
    // SetOscillator,
    // SetResponsive,
    // EmitPheromone,
    // KillForward,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FromConn {
    Input(Input),
    Inner(Inner),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ToConn {
    Output(Output),
    Inner(Inner),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Connection {
    from: FromConn,
    to: ToConn,
    weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, FromRepr)]
enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    fn reverse(&self) -> Self {
        match self {
            Self::North => Self::South,
            Self::South => Self::North,
            Self::East => Self::West,
            Self::West => Self::East,
        }
    }

    fn left(&self) -> Self {
        match self {
            Self::North => Self::West,
            Self::South => Self::East,
            Self::East => Self::North,
            Self::West => Self::South,
        }
    }

    fn right(&self) -> Self {
        match self {
            Self::North => Self::East,
            Self::South => Self::West,
            Self::East => Self::South,
            Self::West => Self::North,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Intention {
    direction: Direction,
}

impl Intention {
    fn new(direction: Direction) -> Self {
        Self { direction }
    }
}

#[derive(Debug, Clone)]
struct Cell {
    genome: Vec<Connection>,
    state: HashMap<Input, f32>,
    intermidiate: HashMap<Inner, (Vec<(FromConn, f32)>, f32)>,
    result: HashMap<Output, (Vec<(FromConn, f32)>, f32)>,
    facing: Direction,
}

impl Cell {
    fn new(genome: Vec<Connection>) -> Self {
        let mut state: HashMap<Input, f32> = HashMap::new();
        let mut intermidiate: HashMap<Inner, (Vec<(FromConn, f32)>, f32)> = HashMap::new();
        let mut result: HashMap<Output, (Vec<(FromConn, f32)>, f32)> = HashMap::new();

        for gene in genome.iter() {
            match gene.from {
                FromConn::Input(input) => {
                    state.insert(input, 0.0);
                }
                _ => {}
            }

            match gene.to {
                ToConn::Output(output) => {
                    if let Some((connections, value)) = result.get_mut(&output) {
                        connections.push((gene.from, gene.weight));
                    } else {
                        result.insert(output, (vec![(gene.from, gene.weight)], 0.0));
                    }
                }
                ToConn::Inner(inner) => {
                    if let Some((connections, value)) = intermidiate.get_mut(&inner) {
                        connections.push((gene.from, gene.weight));
                    } else {
                        intermidiate.insert(inner, (vec![(gene.from, gene.weight)], 0.0));
                    }
                }
            }
        }

        Self {
            genome,
            state,
            intermidiate,
            result,
            facing: Direction::from_repr(rand::thread_rng().gen_range(0..4)).unwrap(),
        }
    }

    fn set_direction(&mut self, direction: Direction) {
        self.facing = direction;
    }

    fn update(&mut self, grid: &Vec<Vec<bool>>) {
        self.state.iter_mut().for_each(|(input, value)| {
            *value = match input {
                Input::Random => rand::thread_rng().gen::<f32>(),
                Input::Oscillator => todo!(),
                Input::Age => todo!(),
                Input::BlockLR => todo!(),
                Input::BlockForward => todo!(),
                Input::BlockForwardLong => todo!(),
                Input::LocX => todo!(),
                Input::LocY => todo!(),
                Input::LastMoveX => todo!(),
                Input::LastMoveY => todo!(),
                Input::LocWallNS => todo!(),
                Input::LocWallEW => todo!(),
                Input::NearestWall => todo!(),
                Input::PopDensity => todo!(),
            };
        });
    }

    fn calc(&mut self) {
        // Initialize intermidiate states using only input states
        let mut init: HashMap<Inner, f32> = HashMap::new();
        self.intermidiate
            .iter()
            .for_each(|(inner, (connections, value))| {
                init.insert(
                    inner.clone(),
                    connections
                        .iter()
                        .fold(0.0, |acc, (from, weight)| match from {
                            FromConn::Input(input) => acc + self.state[input] * weight,
                            _ => acc,
                        })
                        .tanh(),
                );
            });

        // re-calculate intermidiate states including previous values
        self.intermidiate
            .iter_mut()
            .for_each(|(inner, (connections, value))| {
                *value = connections
                    .iter()
                    .fold(0.0, |acc, (from, weight)| match from {
                        FromConn::Input(input) => acc + self.state[input] * weight,
                        FromConn::Inner(inner) => acc + init[inner] * weight,
                    })
                    .tanh();
            });

        self.result
            .iter_mut()
            .for_each(|(output, (connections, value))| {
                *value = connections
                    .iter()
                    .fold(0.0, |acc, (from, weight)| match from {
                        FromConn::Input(input) => acc + self.state[input] * weight,
                        FromConn::Inner(inner) => acc + self.intermidiate[inner].1 * weight,
                    })
                    .tanh();
            });
    }

    fn intention(&self) -> Intention {
        self.result
            .iter()
            .fold(
                Intention::new(self.facing),
                |acc, (output, (_, value))| match output {
                    Output::MoveRandom => {
                        if value > &0.5 {
                            Intention::new(
                                Direction::from_repr(rand::thread_rng().gen_range(0..4)).unwrap(),
                            )
                        } else {
                            acc
                        }
                    }
                    Output::MoveForward => {
                        if value > &0.5 {
                            Intention::new(self.facing)
                        } else {
                            acc
                        }
                    }
                    Output::MoveReverse => {
                        if value > &0.5 {
                            Intention::new(self.facing.reverse())
                        } else {
                            acc
                        }
                    }
                    Output::MoveLR => {
                        if value > &0.5 {
                            Intention::new(self.facing.left())
                        } else if value < &-0.5 {
                            Intention::new(self.facing.right())
                        } else {
                            acc
                        }
                    }
                    Output::MoveEW => {
                        if value > &0.5 {
                            Intention::new(Direction::East)
                        } else if value < &-0.5 {
                            Intention::new(Direction::West)
                        } else {
                            acc
                        }
                    }
                    Output::MoveNS => {
                        if value > &0.5 {
                            Intention::new(Direction::North)
                        } else if value < &-0.5 {
                            Intention::new(Direction::South)
                        } else {
                            acc
                        }
                    }
                },
            )
    }
}

#[derive(Debug, Clone)]
struct World {
    state: Render,
    population: HashMap<(u16, u16), Cell>,
}

impl World {
    fn new(state: &Render) -> Self {
        let mut rng = rand::thread_rng();
        let size_dist = Uniform::from(0..state.world_size);
        let pop_size = ((state.world_size as u32).pow(2) as f32 * state.pop_percent) as usize;

        debug_warn!("pop_size: {}", pop_size);

        let mut coords = HashSet::with_capacity(pop_size);
        while coords.len() < pop_size {
            coords.insert((size_dist.sample(&mut rng), size_dist.sample(&mut rng)));
        }

        debug_warn!(
            "coords: {:?}, sample: {:?}",
            coords.len(),
            coords.iter().next()
        );

        let mut population = HashMap::with_capacity(pop_size);
        for (x, y) in coords {
            let mut genome = Vec::with_capacity(state.genome_size as usize);
            for _ in 0..state.genome_size {
                genome.push(valid_connection(state));
            }

            population.insert((x, y), Cell::new(genome));
        }

        Self {
            state: state.clone(),
            population,
        }
    }

    fn get_map(&self) -> Vec<(u16, u16)> {
        self.population.keys().copied().collect()
    }

    fn step(&mut self) {
        let mut grid = Vec::with_capacity(self.state.world_size as usize);
        for _ in 0..self.state.world_size {
            grid.push(vec![false; self.state.world_size as usize]);
        }

        for (coord, _) in &self.population {
            grid[coord.0 as usize][coord.1 as usize] = true;
        }

        let slice = self.population.keys().cloned().collect::<Vec<_>>();

        slice.iter().for_each(|coord| {
            let mut cell = self.population.remove(coord).unwrap();
            cell.update(&grid);
            cell.calc();

            let res = cell.intention();
            let new_coord: (i16, i16) = match res.direction {
                Direction::North => (coord.0 as i16, coord.1 as i16 + 1),
                Direction::East => (coord.0 as i16 + 1, coord.1 as i16),
                Direction::South => (coord.0 as i16, coord.1 as i16 - 1),
                Direction::West => (coord.0 as i16 - 1, coord.1 as i16),
            };

            let new_coord = if new_coord.0 < 0
                || new_coord.0 >= self.state.world_size as i16
                || new_coord.1 < 0
                || new_coord.1 >= self.state.world_size as i16
            {
                coord.clone()
            } else {
                (new_coord.0 as u16, new_coord.1 as u16)
            };

            if new_coord == *coord || self.population.contains_key(&new_coord) {
                self.population.insert(coord.clone(), cell);
            } else {
                cell.set_direction(res.direction);
                self.population.insert(new_coord, cell);
            }
        });
    }
}

fn valid_connection(state: &Render) -> Connection {
    let mut rng = rand::thread_rng();
    let in_type: bool = rng.gen();
    let from = if in_type {
        FromConn::Input(Input::from_repr(rng.gen_range(0..Input::COUNT)).unwrap())
    } else {
        FromConn::Inner(rng.gen_range(0..INNER_STATES))
    };
    let out_type: bool = rng.gen();
    let to = if out_type {
        ToConn::Output(Output::from_repr(rng.gen_range(0..Output::COUNT)).unwrap())
    } else {
        ToConn::Inner(rng.gen_range(0..INNER_STATES))
    };
    let weight: f32 = rng.gen_range((0.0 - 1.0)..1.0) * (state.weight_factor as f32);

    Connection { from, to, weight }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Render {
    world_size: u16,
    pop_percent: f32,
    time_steps: u16,
    genome_size: u8,
    mutation_rate: f32,
    weight_factor: u8,
}

impl Default for Render {
    fn default() -> Self {
        Self {
            world_size: 32,
            pop_percent: 0.2,
            time_steps: 32,
            genome_size: 1,
            mutation_rate: 0.01,
            weight_factor: 4,
        }
    }
}

fn display_world(state: &Render, map: Vec<(u16, u16)>, canvas: NodeRef<leptos::html::Canvas>) {
    let canvas = canvas.get().unwrap();
    let ctx = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap();
    ctx.clear_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);

    let pixel = move |x: u16, y: u16| {
        ctx.fill_rect(
            x as f64 * canvas.width() as f64 / state.world_size as f64,
            y as f64 * canvas.height() as f64 / state.world_size as f64,
            canvas.width() as f64 / state.world_size as f64,
            canvas.height() as f64 / state.world_size as f64,
        );
    };

    for coord in map {
        pixel(coord.0, coord.1);
    }
}

fn main() {
    mount_to_body(|cx| {
        let (canvas_size, _set_canvas_size) = create_signal(cx, 720);

        let (state, set_state) = create_signal(cx, Render::default());
        set_state(Render {
            world_size: 128,
            pop_percent: 0.2,
            time_steps: 256,
            genome_size: 4,
            mutation_rate: 0.01,
            weight_factor: 4,
        });

        let (render, set_render) = create_signal(cx, Render::default());

        let compute_state = create_memo(cx, move |_| {
            return render() == state();
        });

        let canvas_ref = create_node_ref::<leptos::html::Canvas>(cx);

        let compute = move |_| {
            set_render(state());
            let state = state();

            debug_warn!("state: {:?}", state);

            let console = document().get_element_by_id("console").unwrap();
            console.class_list().remove_1("hidden").unwrap();
            let logger = document().get_element_by_id("log").unwrap();
            let log = |msg: &str| {
                logger
                    .append_child(&document().create_text_node(msg))
                    .unwrap();
                logger
                    .append_child(&document().create_element("br").unwrap())
                    .unwrap();
            };

            log(format!(
                "Loading a world of size {size}x{size}...",
                size = state.world_size
            )
            .as_str());

            let mut world = World::new(&state);

            log("World loaded!");

            display_world(&state, world.get_map(), canvas_ref);

            log("Starting simulation...");

            console.class_list().add_1("hidden").unwrap();

            for _ in 0..state.time_steps {
                world.step();
                display_world(&state, world.get_map(), canvas_ref);
            }

            log("Simulation complete!");
        };

        view! { cx,
            <div class="w-screen h-screen font-sans flex flex-col items-center justify-center md:justify-normal">
                <p class="pt-8 text-4xl font-sans font-bold">"Welcome to Neuro!"</p>
                <p class="text-2xl font-sans">"This is a generational, artificial neural network experiment."</p>
                <div class="container grid grid-cols-1 lg:grid-cols-2 gap-8 lg:my-auto">
                    <div class="h-auto pt-8 lg:pt-0 flex flex-col gap-4 justify-center items-center">
                        <div class="w-full flex items-center">
                            <label for="size" class="text-2xl whitespace-nowrap">"World Size : "</label>

                            <div class="w-full ml-2 flex flex-col justify-between">
                                <input id="size" type="range" min="32" max="256" step="32" list="ticks_world"
                                    on:input=move |ev| {
                                        set_state.update(|state| {
                                            state.world_size = event_target_value(&ev).parse::<u16>().unwrap();
                                        });
                                    }
                                    prop:value=move || state.with(|state| state.world_size) />
                                <datalist id="ticks_world" class="ticks flex flex-col justify-between">
                                    <option value="32" label="32"></option>
                                    <option value="64" label="64"></option>
                                    <option value="96" label="96"></option>
                                    <option value="128" label="128"></option>
                                    <option value="160" label="160"></option>
                                    <option value="192" label="192"></option>
                                    <option value="224" label="224"></option>
                                    <option value="256" label="256"></option>
                                </datalist>
                            </div>
                        </div>

                        <div class="w-full flex items-center">
                            <label for="population" class="text-2xl whitespace-nowrap">"Population : "</label>

                            <div class="w-full ml-2 flex flex-col justify-between">
                                <input id="population" type="range" min="0.1" max="0.9" step="0.1" list="ticks_pop"
                                    on:input=move |ev| {
                                        set_state.update(|state| {
                                            state.pop_percent = event_target_value(&ev).parse::<f32>().unwrap();
                                        });
                                    }
                                    prop:value=move || state.with(|state| state.pop_percent) />
                                <datalist id="ticks_pop" class="ticks flex flex-col justify-between">
                                    <option value="0.1" label="10%"></option>
                                    <option value="0.2" label="20%"></option>
                                    <option value="0.3" label="30%"></option>
                                    <option value="0.4" label="40%"></option>
                                    <option value="0.5" label="50%"></option>
                                    <option value="0.6" label="60%"></option>
                                    <option value="0.7" label="70%"></option>
                                    <option value="0.8" label="80%"></option>
                                    <option value="0.9" label="90%"></option>
                                </datalist>
                            </div>
                        </div>

                        <div class="w-full flex items-center">
                            <label for="time" class="text-2xl whitespace-nowrap">"Time steps : "</label>

                            <div class="w-full ml-2 flex flex-col justify-between">
                                <input id="time" type="range" min="64" max="512" step="64" list="ticks_time"
                                    on:input=move |ev| {
                                        set_state.update(|state| {
                                            state.time_steps = event_target_value(&ev).parse::<u16>().unwrap();
                                        });
                                    }
                                    prop:value=move || state.with(|state| state.time_steps) />
                                <datalist id="ticks_time" class="ticks flex flex-col justify-between">
                                    <option value="64" label="64"></option>
                                    <option value="128" label="128"></option>
                                    <option value="192" label="192"></option>
                                    <option value="256" label="256"></option>
                                    <option value="320" label="320"></option>
                                    <option value="384" label="384"></option>
                                    <option value="448" label="448"></option>
                                    <option value="512" label="512"></option>
                                </datalist>
                            </div>
                        </div>

                        <button prop:disabled=compute_state
                                on:click=compute
                                class="bg-green-400 w-fit mt-2 py-2 px-4 text-lg rounded-full disabled:bg-gray-500 disabled:text-white disabled:opacity-80 hover:bg-green-300 transition-all">
                            "Compute"
                        </button>
                    </div>

                    <div class="relative mb-8 lg:m-0 aspect-square border shadow-2xl rounded-lg overflow-hidden">
                        <div id="console" class="absolute top-0 left-0 z-10 w-full h-full bg-neutral-900 text-neutral-100 text-md hidden">
                            <div id="log" class="m-16"></div>
                        </div>
                        <canvas ref=canvas_ref width=canvas_size height=canvas_size class="bg-white w-full h-full"/>
                    </div>
                </div>
            </div>
        }
    })
}
