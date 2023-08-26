use leptos::*;
use leptos_use::use_interval_fn_with_options;
use leptos_use::utils::Pausable;
use petgraph::algo::dijkstra;
use petgraph::prelude::*;
use petgraph::stable_graph::StableUnGraph;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use strum::{EnumCount, FromRepr};
use wasm_bindgen::JsCast;
use web_sys::CanvasRenderingContext2d;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumCount, FromRepr)]
enum InputType {
    Random = 0,
    Oscillator,
    Age,
    BlockLR,
    BlockForward,
    // BlockForwardLong,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Input(InputType, i16);

type InnerType = u8;
const INNER_STATES: u8 = 4;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Inner(InnerType, i16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumCount, FromRepr)]
enum OutputType {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Output(OutputType, i16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Node {
    Input(Input),
    Output(Output),
    Inner(Inner),
}

impl Node {
    fn get_value(&self) -> f32 {
        match self {
            Self::Input(i) => i.1 as f32 / i16::MIN as f32,
            Self::Output(o) => o.1 as f32 / i16::MIN as f32,
            Self::Inner(i) => i.1 as f32 / i16::MIN as f32,
        }
    }

    fn set_value(&mut self, value: f32) {
        match self {
            Self::Input(i) => i.1 = ((0.0 - value) * i16::MIN as f32) as i16,
            Self::Output(o) => o.1 = ((0.0 - value) * i16::MIN as f32) as i16,
            Self::Inner(i) => i.1 = ((0.0 - value) * i16::MIN as f32) as i16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Connection {
    from: Node,
    to: Node,
    weight: f32,
}

fn valid_connection(state: &Render) -> Connection {
    let mut rng = rand::thread_rng();
    let in_type: bool = rng.gen();
    let from = if in_type {
        Node::Input(Input(
            InputType::from_repr(rng.gen_range(0..InputType::COUNT)).unwrap(),
            0,
        ))
    } else {
        Node::Inner(Inner(rng.gen_range(0..INNER_STATES), 0))
    };
    let out_type: bool = rng.gen();
    let to = if out_type {
        Node::Output(Output(
            OutputType::from_repr(rng.gen_range(0..OutputType::COUNT)).unwrap(),
            0,
        ))
    } else {
        Node::Inner(Inner(rng.gen_range(0..INNER_STATES), 0))
    };
    let weight: f32 = rng.gen_range((0.0 - 1.0)..1.0) * (state.weight_factor as f32);

    Connection { from, to, weight }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Coord {
    x: u16,
    y: u16,
}

impl Coord {
    fn new(x: u16, y: u16) -> Self {
        Self { x, y }
    }

    fn neighbour(&self, bound: (u16, u16), direction: Direction) -> Option<Self> {
        match direction {
            Direction::North => {
                if self.y == bound.1 - 1 {
                    None
                } else {
                    Some(Self::new(self.x, self.y + 1))
                }
            }
            Direction::East => {
                if self.x == bound.0 - 1 {
                    None
                } else {
                    Some(Self::new(self.x + 1, self.y))
                }
            }
            Direction::South => {
                if self.y == 0 {
                    None
                } else {
                    Some(Self::new(self.x, self.y - 1))
                }
            }
            Direction::West => {
                if self.x == 0 {
                    None
                } else {
                    Some(Self::new(self.x - 1, self.y))
                }
            }
        }
    }

    fn neighbours(&self, bound: (u16, u16)) -> Vec<Self> {
        let mut neighbours = Vec::new();
        if let Some(neighbour) = self.neighbour(bound, Direction::North) {
            neighbours.push(neighbour);
            if let Some(neighbour) = neighbour.neighbour(bound, Direction::East) {
                neighbours.push(neighbour);
            }
            if let Some(neighbour) = neighbour.neighbour(bound, Direction::West) {
                neighbours.push(neighbour);
            }
        }
        if let Some(neighbour) = self.neighbour(bound, Direction::East) {
            neighbours.push(neighbour);
        }
        if let Some(neighbour) = self.neighbour(bound, Direction::South) {
            neighbours.push(neighbour);
            if let Some(neighbour) = neighbour.neighbour(bound, Direction::East) {
                neighbours.push(neighbour);
            }
            if let Some(neighbour) = neighbour.neighbour(bound, Direction::West) {
                neighbours.push(neighbour);
            }
        }
        if let Some(neighbour) = self.neighbour(bound, Direction::West) {
            neighbours.push(neighbour);
        }
        neighbours
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

fn new_brain(
    params: &Render,
) -> (
    StableUnGraph<Node, f32>,
    HashMap<InputType, NodeIndex>,
    HashMap<InnerType, NodeIndex>,
    HashMap<OutputType, NodeIndex>,
) {
    let mut graph = StableUnGraph::<Node, f32>::default();
    let mut inputs = HashMap::new();
    let mut internals = HashMap::new();
    let mut outputs = HashMap::new();

    let mut genome = Vec::with_capacity(params.genome_size as usize);
    for _ in 0..params.genome_size {
        genome.push(valid_connection(params));
    }

    for gene in genome.iter() {
        match gene.from {
            Node::Input(input) => {
                if inputs.get(&input.0).is_none() {
                    let node = graph.add_node(Node::Input(input));
                    inputs.insert(input.0, node);
                }
            }
            Node::Inner(inner) => {
                if internals.get(&inner.0).is_none() {
                    let node = graph.add_node(Node::Inner(inner));
                    internals.insert(inner.0, node);
                }
            }
            _ => unreachable!(),
        }

        match gene.to {
            Node::Inner(inner) => {
                if internals.get(&inner.0).is_none() {
                    let node = graph.add_node(Node::Inner(inner));
                    internals.insert(inner.0, node);
                }
            }
            Node::Output(output) => {
                if outputs.get(&output.0).is_none() {
                    let node = graph.add_node(Node::Output(output));
                    outputs.insert(output.0, node);
                }
            }
            _ => unreachable!(),
        }

        let from = match gene.from {
            Node::Input(input) => inputs.get(&input.0).copied().unwrap(),
            Node::Inner(inner) => internals.get(&inner.0).copied().unwrap(),
            _ => unreachable!(),
        };
        let to = match gene.to {
            Node::Inner(inner) => internals.get(&inner.0).copied().unwrap(),
            Node::Output(output) => outputs.get(&output.0).copied().unwrap(),
            _ => unreachable!(),
        };
        graph.add_edge(from, to, gene.weight);
    }

    inputs.retain(|_, i| {
        let paths = dijkstra(&graph, *i, None, |_| 1);
        let mut valid = paths.keys();
        let res = valid.any(|n| match graph.node_weight(*n).unwrap() {
            Node::Output(_) => true,
            _ => false,
        });
        if !res {
            graph.remove_node(*i);
        }
        res
    });
    internals.retain(|_, i| {
        let paths = dijkstra(&graph, *i, None, |_| 1);
        let mut valid = paths.keys();
        let ins = valid.any(|n| match graph.node_weight(*n).unwrap() {
            Node::Input(_) => true,
            _ => false,
        });
        let outs = valid.any(|n| match graph.node_weight(*n).unwrap() {
            Node::Output(_) => true,
            _ => false,
        });
        if !ins || !outs {
            graph.remove_node(*i);
        }
        ins && outs
    });
    outputs.retain(|_, i| {
        let paths = dijkstra(&graph, *i, None, |_| 1);
        let mut valid = paths.keys();
        let res = valid.any(|n| match graph.node_weight(*n).unwrap() {
            Node::Inner(_) => true,
            _ => false,
        });
        if !res {
            graph.remove_node(*i);
        }
        res
    });

    (graph, inputs, internals, outputs)
}

#[derive(Debug, Clone)]
struct Neuron {
    brain: StableUnGraph<Node, f32>,
    inputs: HashMap<InputType, NodeIndex>,
    internals: HashMap<InnerType, NodeIndex>,
    outputs: HashMap<OutputType, NodeIndex>,
    facing: Direction,
    last_move: (u16, u16),
}

impl Neuron {
    fn new(params: &Render) -> Self {
        let brain;
        let inputs;
        let internals;
        let outputs;
        loop {
            let res = new_brain(params);
            if res.0.edge_count() > 0 {
                brain = res.0;
                inputs = res.1;
                internals = res.2;
                outputs = res.3;
                break;
            }
        }

        Self {
            brain,
            inputs,
            internals,
            outputs,
            facing: Direction::from_repr(rand::thread_rng().gen_range(0..4)).unwrap(),
            last_move: (0, 0),
        }
    }

    fn try_move(&mut self, direction: Option<Direction>) {
        match direction {
            Some(dir) => {
                match dir {
                    Direction::North | Direction::South => {
                        self.last_move.0 += 1;
                        self.last_move.1 = 0;
                    }
                    Direction::East | Direction::West => {
                        self.last_move.0 = 0;
                        self.last_move.1 += 1;
                    }
                };
                self.facing = dir;
            }
            _ => {
                self.last_move.0 += 1;
                self.last_move.1 += 1;
            }
        }
    }

    fn update(&mut self, grid: &Vec<Vec<bool>>, time: f32, coord: Coord) {
        self.inputs.iter().for_each(|(input, index)| {
            let node = self.brain.node_weight_mut(*index).unwrap();
            node.set_value(match input {
                InputType::Random => rand::thread_rng().gen_range((0.0 - 1.0)..1.0),
                InputType::Oscillator => ((time * 256.0) / std::f32::consts::PI).sin(),
                InputType::Age => (time - 0.5) * 2.0,
                InputType::BlockLR => {
                    let left = self.facing.left();
                    let right = self.facing.right();
                    let left = coord.neighbour((grid.len() as u16, grid[0].len() as u16), left);
                    let right = coord.neighbour((grid.len() as u16, grid[0].len() as u16), right);
                    let left = left
                        .map(|coord| grid[coord.x as usize][coord.y as usize])
                        .unwrap_or(false);
                    let right = right
                        .map(|coord| grid[coord.x as usize][coord.y as usize])
                        .unwrap_or(false);
                    if left && right {
                        0.0
                    } else if left {
                        0.5
                    } else if right {
                        -0.5
                    } else {
                        1.0
                    }
                }
                InputType::BlockForward => {
                    let forward =
                        coord.neighbour((grid.len() as u16, grid[0].len() as u16), self.facing);
                    let forward = forward
                        .map(|coord| grid[coord.x as usize][coord.y as usize])
                        .unwrap_or(true);
                    if forward {
                        1.0
                    } else {
                        0.0 - 1.0
                    }
                }
                /* InputType::BlockForwardLong => {
                    let mut i = 0;
                    loop {
                        let forward =
                            coord.neighbour((grid.len() as u16, grid[0].len() as u16), self.facing);
                        let forward = forward
                            .map(|coord| grid[coord.x as usize][coord.y as usize])
                            .unwrap_or(true);
                        if forward {
                            break;
                        }
                        i += 1;
                    }
                    match self.facing {
                        Direction::North | Direction::South => {
                            (0.5 - (i as f32 / grid.len() as f32)) * 2.0
                        }
                        Direction::East | Direction::West => {
                            (0.5 - (i as f32 / grid[0].len() as f32)) * 2.0
                        }
                    }
                } */
                InputType::LocX => coord.x as f32 / grid.len() as f32,
                InputType::LocY => coord.y as f32 / grid[0].len() as f32,
                InputType::LastMoveX => (0.5 - (self.last_move.0 as f32 / time)) * 2.0,
                InputType::LastMoveY => (0.5 - (self.last_move.1 as f32 / time)) * 2.0,
                InputType::LocWallNS => (coord.y as f32 / grid[0].len() as f32) * 2.0 - 1.0,
                InputType::LocWallEW => (coord.x as f32 / grid.len() as f32) * 2.0 - 1.0,
                InputType::NearestWall => {
                    let ns = ((coord.y as f32 / grid[0].len() as f32) * 2.0 - 1.0).abs();
                    let ew = ((coord.x as f32 / grid.len() as f32) * 2.0 - 1.0).abs();
                    if ns < ew {
                        ns * 2.0 - 1.0
                    } else {
                        ew * 2.0 - 1.0
                    }
                }
                InputType::PopDensity => {
                    let neighbours = coord.neighbours((grid.len() as u16, grid[0].len() as u16));
                    neighbours.iter().fold(0.0, |acc, coord| {
                        if grid[coord.x as usize][coord.y as usize] {
                            acc + 1.0
                        } else {
                            acc
                        }
                    }) / 8.0
                }
            });
        });
    }

    fn calc(&mut self) {
        self.internals.iter().for_each(|(_, index)| {
            let mut value = 0.0;
            self.brain.neighbors(*index).for_each(|e| {
                let neighbour = self.brain.node_weight(e).unwrap();
                if let Node::Output(_) = neighbour {
                    return;
                }
                let weight = self
                    .brain
                    .edge_weight(self.brain.find_edge(*index, e).unwrap())
                    .unwrap();
                value += neighbour.get_value() * weight;
            });
            let node = self.brain.node_weight_mut(*index).unwrap();
            node.set_value(value.tanh());
        });

        self.outputs.iter().for_each(|(_, index)| {
            let mut value = 0.0;
            self.brain.neighbors(*index).for_each(|e| {
                let neighbour = self.brain.node_weight(e).unwrap();
                let weight = self
                    .brain
                    .edge_weight(self.brain.find_edge(*index, e).unwrap())
                    .unwrap();
                value += neighbour.get_value() * weight;
            });
            let node = self.brain.node_weight_mut(*index).unwrap();
            node.set_value(value.tanh());
        });
    }

    fn intention(&self) -> Intention {
        self.outputs
            .iter()
            .fold(Intention::new(self.facing), |acc, (output, index)| {
                let value = self.brain.node_weight(*index).unwrap().get_value();
                match output {
                    OutputType::MoveRandom => {
                        if value > 0.5 {
                            Intention::new(
                                Direction::from_repr(rand::thread_rng().gen_range(0..4)).unwrap(),
                            )
                        } else {
                            acc
                        }
                    }
                    OutputType::MoveForward => {
                        if value > 0.5 {
                            Intention::new(self.facing)
                        } else {
                            acc
                        }
                    }
                    OutputType::MoveReverse => {
                        if value > 0.5 {
                            Intention::new(self.facing.reverse())
                        } else {
                            acc
                        }
                    }
                    OutputType::MoveLR => {
                        if value > 0.5 {
                            Intention::new(self.facing.left())
                        } else if value < -0.5 {
                            Intention::new(self.facing.right())
                        } else {
                            acc
                        }
                    }
                    OutputType::MoveEW => {
                        if value > 0.5 {
                            Intention::new(Direction::East)
                        } else if value < -0.5 {
                            Intention::new(Direction::West)
                        } else {
                            acc
                        }
                    }
                    OutputType::MoveNS => {
                        if value > 0.5 {
                            Intention::new(Direction::North)
                        } else if value < -0.5 {
                            Intention::new(Direction::South)
                        } else {
                            acc
                        }
                    }
                }
            })
    }
}

#[derive(Debug, Clone)]
struct World {
    state: Render,
    time: usize,
    population: HashMap<Coord, Neuron>,
}

impl Default for World {
    fn default() -> Self {
        Self {
            state: Render::default(),
            time: 0,
            population: HashMap::new(),
        }
    }
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

        let mut population = HashMap::with_capacity(pop_size);
        for (x, y) in coords {
            population.insert(Coord::new(x, y), Neuron::new(state));
        }

        Self {
            state: state.clone(),
            time: 0,
            population,
        }
    }

    fn get_map(&self) -> Vec<Coord> {
        self.population.keys().copied().collect()
    }

    fn step(&mut self) {
        let mut grid = Vec::with_capacity(self.state.world_size as usize);
        for _ in 0..self.state.world_size {
            grid.push(vec![false; self.state.world_size as usize]);
        }

        for (coord, _) in &self.population {
            grid[coord.x as usize][coord.y as usize] = true;
        }
        /* debug_warn!(
            "grid: {:?}",
            grid.clone()
                .iter()
                .map(|x| x
                    .clone()
                    .iter()
                    .map(|y| if *y { 1 } else { 0 })
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        ); */

        let slice = self.population.keys().cloned().collect::<Vec<_>>();

        slice.iter().for_each(|coord| {
            debug_warn!("coord: {:?}", coord.clone());
            let mut neuron = self.population.remove(coord).unwrap();
            debug_warn!("neuron: {:?}", neuron.clone());
            neuron.update(
                &grid,
                self.time as f32 / self.state.time_steps as f32,
                coord.clone(),
            );
            neuron.calc();

            let res = neuron.intention();
            debug_warn!("intention: {:?}", res.clone());
            let new_coord = coord.neighbour(
                (self.state.world_size, self.state.world_size),
                res.direction,
            );

            match new_coord {
                Some(new_coord) => {
                    if self.population.contains_key(&new_coord) {
                        neuron.try_move(None);
                        self.population.insert(coord.clone(), neuron);
                    } else {
                        neuron.try_move(Some(res.direction));
                        self.population.insert(new_coord, neuron);
                    }
                }
                None => {
                    neuron.try_move(None);
                    self.population.insert(coord.clone(), neuron);
                }
            }
            debug_warn!("neuron moved");
        });
        self.time += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Render {
    world_size: u16,
    pop_percent: f32,
    time_steps: usize,
    genome_size: u8,
    mutation_rate: f32,
    weight_factor: u8,
}

impl Default for Render {
    fn default() -> Self {
        Self {
            world_size: 128,
            pop_percent: 0.2,
            time_steps: 192,
            genome_size: 4,
            mutation_rate: 0.01,
            weight_factor: 4,
        }
    }
}

fn display_world(state: &Render, map: Vec<Coord>, canvas: NodeRef<leptos::html::Canvas>) {
    debug_warn!("display_world");
    debug_warn!("map: {:?}", map.clone());
    let canvas = canvas.get().unwrap();
    let ctx = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap();
    ctx.clear_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);

    let pixel = move |x: u16, y: u16| {
        let y = state.world_size - y - 1;
        ctx.fill_rect(
            x as f64 * canvas.width() as f64 / state.world_size as f64,
            y as f64 * canvas.height() as f64 / state.world_size as f64,
            canvas.width() as f64 / state.world_size as f64,
            canvas.height() as f64 / state.world_size as f64,
        );
    };

    for coord in map {
        pixel(coord.x, coord.y);
    }
}

fn clear_canvas(canvas: NodeRef<leptos::html::Canvas>) {
    let canvas = canvas.get().unwrap();
    let ctx = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap();
    ctx.clear_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Clock {
    gen: usize,
    time: usize,
}

impl Default for Clock {
    fn default() -> Self {
        Self { gen: 0, time: 0 }
    }
}

impl Clock {
    fn set(&mut self, gen: usize, time: usize) {
        self.gen = gen;
        self.time = time;
    }
}

fn main() {
    mount_to_body(|cx| {
        const CANVAS_SIZE: usize = 720;
        let (clock, set_clock) = create_signal(cx, Clock::default());
        let (world, set_world) = create_signal(cx, World::default());
        let (history, set_history) = create_signal(cx, Vec::<Vec<Coord>>::new());
        let (state, set_state) = create_signal(cx, Render::default());

        let canvas_ref = create_node_ref::<leptos::html::Canvas>(cx);

        let Pausable {
            pause,
            resume,
            is_active,
        } = use_interval_fn_with_options(
            cx,
            move || {
                debug_warn!("interval");
                let state = state.get();
                let mut clock = clock.get();
                if clock.time < state.time_steps {
                    clock.time += 1;
                } else {
                    clock.time = 0;
                    clock.gen += 1;
                }
                set_clock(clock);
            },
            500,
            leptos_use::UseIntervalFnOptions {
                immediate: false,
                ..Default::default()
            },
        );

        let stop = watch(
            cx,
            move || clock.get(),
            move |clock, _, _| {
                debug_warn!("animation");
                debug_warn!("clock: {:?}", clock.clone());

                if history.get().len() <= 0 {
                    return;
                }

                display_world(
                    &state(),
                    history.with(|hist| hist[clock.time].clone()),
                    canvas_ref.clone(),
                );
            },
            false,
        );

        let pause_1 = pause.clone();
        let resume_1 = resume.clone();
        let compute = move |_| {
            let state = state.get();

            debug_warn!("state: {:?}", state);

            if is_active.get() {
                pause_1();
            }
            set_clock(Clock::default());
            set_history(vec![]);
            clear_canvas(canvas_ref.clone());

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

            set_world.set(World::new(&state));

            log("World loaded!");

            set_history.update(|hist| hist.push(world.with(|w| w.get_map())));

            console.class_list().add_1("hidden").unwrap();

            for i in 0..state.time_steps {
                debug_warn!("step {}", i);
                set_world.update(|w| w.step());
                set_history.update(|hist| hist.push(world.with(|w| w.get_map())));
            }

            if !is_active.get() {
                resume_1();
            }
            debug_warn!("done");
        };

        let pause_2 = pause.clone();
        let resume_2 = resume.clone();
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
                                            state.time_steps = event_target_value(&ev).parse().unwrap();
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

                        <div class="flex flex-row gap-2">
                            <button on:click=compute
                                class="bg-green-400 w-fit mt-2 py-2 px-4 text-lg rounded-full disabled:bg-gray-500 disabled:text-white disabled:opacity-80 hover:bg-green-300 transition-all">
                                "Compute"
                            </button>
                        </div>

                        <div class="flex flex-row gap-0 bg-neutral-50 rounded-lg border justify-center items-center">
                            <div class="flex flex-col gap-0 border-r">
                                <span class="pt-2 px-2 text-2xl">Generation: {move || clock.get().gen}</span>
                                <span class="py-2 px-2 text-2xl">Time: {move || clock.get().time}</span>
                            </div>
                            <div class="h-full grid grid-cols-3 gap-0">
                                    <button on:click=move |_| set_clock(Clock::default())
                                        class="h-full p-2 border-r border-b">
                                        <svg class="w-8 aspect-square" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                            <path d="M19 19L12.7071 12.7071C12.3166 12.3166 12.3166 11.6834 12.7071 11.2929L19 5" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                            <path d="M11 19L4.70711 12.7071C4.31658 12.3166 4.31658 11.6834 4.70711 11.2929L11 5" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        </svg>
                                    </button>
                                    <Show
                                        when=move || is_active()
                                        fallback=move |cx| {
                                            let resume = resume_2.clone();
                                            view! { cx,
                                                <button on:click=move |_| resume()
                                                    class="h-full p-2 border-b">
                                                    <svg class="h-full" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <path d="M8 17.1783V6.82167C8 6.03258 8.87115 5.55437 9.53688 5.97801L17.6742 11.1563C18.2917 11.5493 18.2917 12.4507 17.6742 12.8437L9.53688 18.022C8.87115 18.4456 8 17.9674 8 17.1783Z" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                                    </svg>
                                                </button>
                                            }
                                        } >
                                        {
                                            let pause = pause_2.clone();
                                            view! {cx,
                                                <button on:click=move |_| pause()
                                                    class="h-full p-2 border-b">
                                                    <svg class="h-full" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <rect x="6" y="6" width="4" height="12" rx="1" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                                        <rect x="14" y="6" width="4" height="12" rx="1" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                                    </svg>
                                                </button>
                                            }
                                        }
                                    </Show>
                                    <button on:click=move |_| {
                                            let mut clock = Clock::default();
                                            clock.set(0, state().time_steps);
                                            set_clock(clock);
                                        }
                                        class="h-full p-2 border-l border-b">
                                        <svg class="h-full" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                            <path d="M5 19L11.2929 12.7071C11.6834 12.3166 11.6834 11.6834 11.2929 11.2929L5 5" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                            <path d="M13 19L19.2929 12.7071C19.6834 12.3166 19.6834 11.6834 19.2929 11.2929L13 5" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        </svg>
                                    </button>
                                    <button on:click=move |_| {
                                                let state = state();
                                                let mut clock = clock.get();

                                                if clock.time > 0 {
                                                    clock.time -= 1;
                                                } else {
                                                    if clock.gen > 0 {
                                                        clock.gen -= 1;
                                                        clock.time = state.time_steps;
                                                    } else {
                                                        return;
                                                    }
                                                }
                                                set_clock(clock);
                                            }
                                        class="h-full p-2 border-r">
                                            <svg class="h-full" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                <path d="M15.5 19L9.20711 12.7071C8.81658 12.3166 8.81658 11.6834 9.20711 11.2929L15.5 5" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                            </svg>
                                    </button>
                                    <Show
                                        when=move || is_active()
                                        fallback=move |cx| {
                                            let resume = resume.clone();
                                            view! { cx,
                                                <button on:click=move |_| resume()
                                                    class="h-full p-2 border-b">
                                                    <svg class="h-full" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <path d="M8 17.1783V6.82167C8 6.03258 8.87115 5.55437 9.53688 5.97801L17.6742 11.1563C18.2917 11.5493 18.2917 12.4507 17.6742 12.8437L9.53688 18.022C8.87115 18.4456 8 17.9674 8 17.1783Z" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                                    </svg>
                                                </button>
                                            }
                                        } >
                                        {
                                            let pause = pause.clone();
                                            view! {cx,
                                                <button on:click=move |_| pause()
                                                    class="h-full p-2 border-b">
                                                    <svg class="h-full" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <rect x="6" y="6" width="4" height="12" rx="1" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                                        <rect x="14" y="6" width="4" height="12" rx="1" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                                    </svg>
                                                </button>
                                            }
                                        }
                                    </Show>
                                    <button on:click=move |_| {
                                            let state = state();
                                            let mut clock = clock.get();

                                            if clock.time < state.time_steps {
                                                clock.time += 1;
                                            } else {
                                                clock.time = 0;
                                                clock.gen += 1;
                                            }
                                            set_clock(clock);
                                        }
                                        class="h-full p-2 border-l">
                                        <svg class="h-full" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                            <path d="M9.5 5L15.7929 11.2929C16.1834 11.6834 16.1834 12.3166 15.7929 12.7071L9.5 19" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        </svg>
                                    </button>
                            </div>
                            <div></div>
                        </div>
                    </div>

                    <div class="relative mb-8 lg:m-0 aspect-square border shadow-2xl rounded-lg overflow-hidden">
                        <div id="console" class="absolute top-0 left-0 z-10 w-full h-full bg-neutral-900 text-neutral-100 text-md hidden">
                            <div id="log" class="m-16"></div>
                        </div>
                        <canvas ref=canvas_ref width=CANVAS_SIZE height=CANVAS_SIZE class="bg-white w-full h-full"/>
                    </div>
                </div>
            </div>
        }
    })
}
