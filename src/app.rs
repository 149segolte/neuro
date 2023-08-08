use rand::distributions::{Distribution, Uniform};
use std::collections::HashMap;

use crate::error_template::{AppError, ErrorTemplate};
use leptos::*;
use leptos_meta::*;
use leptos_router::*;

#[component]
pub fn App(cx: Scope) -> impl IntoView {
    // Provides context that manages stylesheets, titles, meta tags, etc.
    provide_meta_context(cx);

    view! {
        cx,

        // injects a stylesheet into the document <head>
        // id=leptos means cargo-leptos will hot-reload this stylesheet
        <Stylesheet id="leptos" href="/pkg/start-axum.css"/>

        // sets the document title
        <Title text="Welcome to Leptos"/>

        // content for this welcome page
        <Router fallback=|cx| {
            let mut outside_errors = Errors::default();
            outside_errors.insert_with_default_key(AppError::NotFound);
            view! { cx,
                <ErrorTemplate outside_errors/>
            }
            .into_view(cx)
        }>
            <main>
                <Routes>
                    <Route path="" view=|cx| view! { cx, <HomePage/> }/>
                </Routes>
            </main>
        </Router>
    }
}

struct Cell {
    genome: [u8; 8],
    state: u8,
}

struct World {
    size: u8,
    population: HashMap<(u8, u8), Cell>,
    rng: rand::rngs::ThreadRng,
}

impl World {
    fn new(size: u8, pop_percent: u16) -> Self {
        let mut rng = rand::thread_rng();
        let coord = Uniform::from(0..size);
        let mut population = HashMap::new();
        let pop_size = (size as u16 * size as u16) * pop_percent / 100;
        let pop_size = 2u16.pow((pop_size as f32).log2().ceil() as u32);

        for _ in 0..pop_size {
            loop {
                let x = coord.sample(&mut rng);
                let y = coord.sample(&mut rng);
                if !population.contains_key(&(x, y)) {
                    population.insert(
                        (x, y),
                        Cell {
                            genome: rand::random::<[u8; 8]>(),
                            state: 0,
                        },
                    );
                    break;
                }
            }
        }
        Self {
            size,
            population,
            rng,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RenderState {
    world_size: u8,
    pop_size: u16,
    time_step: u16,
    genome_size: u8,
    mutation_rate: f32,
    inner_states: u8,
    scale_factor: usize,
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            world_size: 32,
            pop_size: 128,
            time_step: 32,
            genome_size: 1,
            mutation_rate: 0.01,
            inner_states: 2,
            scale_factor: 16384,
        }
    }
}

/// Renders the home page of your application.
#[component]
fn HomePage(cx: Scope) -> impl IntoView {
    let (canvas_size, _set_canvas_size) = create_signal(cx, 720);
    let (world_size, _set_world_size) = create_signal(cx, 128);
    let (pop_size, _set_pop_size) = create_signal(cx, 20);
    let (time_step, _set_time_step) = create_signal(cx, 256);
    let (genome_size, _set_genome_size) = create_signal(cx, 4);
    let (mutation_rate, _set_mutation_rate) = create_signal(cx, 0.01);
    let (inner_states, _set_inner_states) = create_signal(cx, 4);
    let scale_factor = 8192;

    let (state, set_state) = create_signal(cx, RenderState::default());
    let world: Option<World> = None;

    let compute_state = create_memo(cx, move |_| {
        let curr = RenderState {
            world_size: world_size.get(),
            pop_size: pop_size.get(),
            time_step: time_step.get(),
            genome_size: genome_size.get(),
            mutation_rate: mutation_rate.get(),
            inner_states: inner_states.get(),
            scale_factor,
        };

        return curr == state();
    });

    let canvas_ref = create_node_ref::<leptos::html::Canvas>(cx);

    let compute = move |_| {
        set_state(RenderState {
            world_size: world_size.get(),
            pop_size: pop_size.get(),
            time_step: time_step.get(),
            genome_size: genome_size.get(),
            mutation_rate: mutation_rate.get(),
            inner_states: inner_states.get(),
            scale_factor,
        });

        let console = document().get_element_by_id("console").unwrap();
        console.class_list().remove_1("hidden").unwrap();
        let log = document().get_element_by_id("log").unwrap();

        log.append_child(
            &document().create_text_node(
                format!(
                    "Loading a world of size {size}x{size}...",
                    size = state.get().world_size
                )
                .as_str(),
            ),
        )
        .unwrap();

        let mut world = World::new(state.get().world_size, state.get().pop_size);
    };

    view! { cx,
        <div class="bg-green-50 w-screen h-screen font-sans flex flex-col items-center">
            <p class="pt-8 text-4xl font-sans font-bold">"Welcome to Neuro!"</p>
            <p class="text-2xl font-sans">"This is a generational, artificial neural network experiment."</p>
            <div class="container grid grid-cols-2 gap-8 my-auto">
                <div class="h-auto flex flex-col gap-4 justify-center items-center">
                    <div class="w-full flex items-center">
                        <label for="size" class="text-2xl whitespace-nowrap">"World Size : "</label>

                        <div class="w-full ml-2 flex flex-col justify-between">
                            <input id="size" type="range" min="32" max="256" step="32" list="ticks_world" value=world_size prop:value=world_size />
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
                            <input id="population" type="range" min="10" max="90" step="10" list="ticks_pop" value=pop_size prop:value=pop_size />
                            <datalist id="ticks_pop" class="ticks flex flex-col justify-between">
                                <option value="10" label="10%"></option>
                                <option value="20" label="20%"></option>
                                <option value="30" label="30%"></option>
                                <option value="40" label="40%"></option>
                                <option value="50" label="50%"></option>
                                <option value="60" label="60%"></option>
                                <option value="70" label="70%"></option>
                                <option value="80" label="80%"></option>
                                <option value="90" label="90%"></option>
                            </datalist>
                        </div>
                    </div>

                    <div class="w-full flex items-center">
                        <label for="time" class="text-2xl whitespace-nowrap">"Time steps : "</label>

                        <div class="w-full ml-2 flex flex-col justify-between">
                            <input id="time" type="range" min="64" max="512" step="64" list="ticks_time" value=time_step prop:value=time_step />
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
}
