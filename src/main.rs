use leptos::*;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use strum::{EnumCount, FromRepr};
use wasm_bindgen::JsCast;
use web_sys::CanvasRenderingContext2d;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Connection {
    from: u8,
    to: u8,
    weight: f32,
}

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
    MoveForward = 0,
    MoveRandom,
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

const INNER_STATES: u8 = 4;

#[derive(Debug, Clone)]
struct Cell {
    genome: Vec<Connection>,
    state: HashMap<Input, f32>,
    brain: HashMap<u8, f32>,
    output: HashMap<Output, f32>,
}

impl Cell {
    fn new(genome: Vec<Connection>) -> Self {
        let mut state = HashMap::new();
        let mut brain = HashMap::new();
        let mut output = HashMap::new();

        for gene in genome.iter() {
            let in_type = gene.from & 0b1000_0000;
            let from = gene.from & 0b0111_1111;
            let out_type = gene.to & 0b1000_0000;
            let to = gene.to & 0b0111_1111;

            if in_type == 0 {
                state.insert(Input::from_repr(from as usize).unwrap(), 0.0);
            } else {
                brain.insert(from, 0.0);
            }

            if out_type == 0 {
                output.insert(Output::from_repr(to as usize).unwrap(), 0.0);
            } else {
                brain.insert(to, 0.0);
            }
        }

        Self {
            genome,
            state,
            brain,
            output,
        }
    }

    fn update(&mut self, grid: &Vec<Vec<bool>>) {
        todo!()
    }

    fn calc(&mut self) {
        todo!()
    }

    fn intention(&self) -> u8 {
        todo!()
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
            let cell = self.population.get_mut(coord).unwrap();
            cell.update(&grid);
            cell.calc();

            let new_coord: (i16, i16) = match cell.intention() {
                0 => (coord.0 as i16, coord.1 as i16 - 1),
                1 => (coord.0 as i16 + 1, coord.1 as i16),
                2 => (coord.0 as i16, coord.1 as i16 + 1),
                3 => (coord.0 as i16 - 1, coord.1 as i16),
                _ => panic!("Invalid move: {}", cell.intention()),
            };

            if new_coord.0 < 0
                || new_coord.0 >= self.state.world_size as i16
                || new_coord.1 < 0
                || new_coord.1 >= self.state.world_size as i16
            {
                return;
            }

            let new_coord = (new_coord.0 as u16, new_coord.1 as u16);

            if self.population.contains_key(&new_coord) {
                return;
            }

            if let Some(cell) = self.population.remove(coord) {
                self.population.insert(new_coord, cell);
            }
        });
    }
}

fn valid_connection(state: &Render) -> Connection {
    let mut rng = rand::thread_rng();
    let in_type: bool = rng.gen();
    let mut from: u8;
    if in_type {
        from = rng.gen_range(0..Input::COUNT as u8);
    } else {
        from = rng.gen_range(0..INNER_STATES);
        from |= 0b1000_0000;
    }
    let out_type: bool = rng.gen();
    let mut to: u8;
    if out_type {
        to = rng.gen_range(0..Output::COUNT as u8);
    } else {
        to = rng.gen_range(0..INNER_STATES);
        to |= 0b1000_0000;
    }
    let weight: f32 =
        rng.gen_range((0.0 - state.weight_factor as f32)..(state.weight_factor as f32));

    Connection {
        from,
        to,
        weight: weight as f32 / state.weight_factor as f32,
    }
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
