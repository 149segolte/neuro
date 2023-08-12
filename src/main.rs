use leptos::*;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use wasm_bindgen::JsCast;
use web_sys::CanvasRenderingContext2d;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Connection {
    from: u8,
    to: u8,
    weight: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct Cell {
    genome: Vec<Connection>,
    state: u8,
}

#[derive(Debug, Clone)]
struct World {
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

            population.insert((x, y), Cell { genome, state: 0 });
        }

        Self { population }
    }
}

fn valid_connection(state: &Render) -> Connection {
    let mut rng = rand::thread_rng();
    let rand_num: u32 = rng.gen();
    let from = (rand_num >> 24) as u8;
    let to = ((rand_num >> 16) & 0xFF) as u8;
    let weight = (((rand_num << 16) & 0xFFFF0000) >> 16) as i16;

    Connection {
        from,
        to,
        weight: weight as f32 / state.scale_factor as f32,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Render {
    world_size: u16,
    pop_percent: f32,
    time_steps: u16,
    genome_size: u8,
    mutation_rate: f32,
    input_states: u8,
    inner_states: u8,
    output_states: u8,
    scale_factor: usize,
}

impl Default for Render {
    fn default() -> Self {
        Self {
            world_size: 32,
            pop_percent: 0.2,
            time_steps: 32,
            genome_size: 1,
            mutation_rate: 0.01,
            input_states: 2,
            inner_states: 2,
            output_states: 2,
            scale_factor: 16384,
        }
    }
}

fn display_world(state: &Render, world: World, canvas: NodeRef<leptos::html::Canvas>) {
    let canvas = canvas.get().unwrap();
    canvas.set_width(state.world_size as u32);
    canvas.set_height(state.world_size as u32);

    let ctx = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap();
    ctx.set_fill_style(&"#000000".into());
    ctx.fill_rect(0.0, 0.0, state.world_size as f64, state.world_size as f64);

    for ((x, y), cell) in &world.population {
        let color = match cell.state {
            0 => "#000000",
            1 => "#FF0000",
            2 => "#00FF00",
            3 => "#0000FF",
            _ => "#FFFFFF",
        };

        ctx.set_fill_style(&color.into());
        ctx.fill_rect(*x as f64, *y as f64, 1.0, 1.0);
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
            input_states: 11,
            inner_states: 4,
            output_states: 4,
            scale_factor: 8192,
        });

        let (render, set_render) = create_signal(cx, Render::default());
        let mut world: Option<World> = None;

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

            world = Some(World::new(&state));

            log("World created!");
            unimplemented!();

            display_world(&state, world.clone().unwrap(), canvas_ref);

            log("World loaded!");

            console.class_list().add_1("hidden").unwrap();
        };

        view! { cx,
            <div class="bg-green-50 w-screen h-screen font-sans flex flex-col items-center">
                <p class="pt-8 text-4xl font-sans font-bold">"Welcome to Neuro!"</p>
                <p class="text-2xl font-sans">"This is a generational, artificial neural network experiment."</p>
                <div class="container grid grid-cols-1 lg:grid-cols-2 gap-8 my-auto">
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

                    <div class="relative aspect-square border shadow-2xl rounded-lg overflow-hidden">
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
