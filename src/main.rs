use std::f32::EPSILON;
use std::io;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::symbols::{self, Marker};
use ratatui::widgets::canvas::{Canvas, Circle, Rectangle};
use ratatui::widgets::{Block, Borders, Paragraph, RenderDirection, Sparkline};
use ratatui::Terminal;

use crossterm::event::{self, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

use rusty_net::{HaltCondition, NN};

fn main() -> Result<(), io::Error> {
    // Training examples for the neural network
    let examples = [
        (vec![0f64, 0f64], vec![0f64]), // 0 AND 0 = 0
        (vec![0f64, 1f64], vec![0f64]), // 0 AND 1 = 0
        (vec![1f64, 0f64], vec![0f64]), // 1 AND 0 = 0
        (vec![1f64, 1f64], vec![1f64]), // 1 AND 1 = 1
    ];

    let network_layers = vec![2, 7,5, 1];

    let layers = network_layers.clone();
    // Shared vector for Ratatui to read data updated by the neural network
    let data = Arc::new(Mutex::new(vec![0u64]));
    let data_for_training = Arc::clone(&data);
    let count_epochs = Arc::new(Mutex::new(0));
    let current_error = Arc::new(Mutex::new(0.0));

    let count_epochs_for_ui = Arc::clone(&count_epochs);
    let current_error_for_ui = Arc::clone(&current_error);

    // Start training in a separate thread
    thread::spawn(move || {
        let l = network_layers.clone();
        let mut net = NN::new(&l);
        let epoch_count = Arc::new(Mutex::new(0));
        let epoch_count_for_training = Arc::clone(&epoch_count);

        net.train(&examples)
            .halt_condition(HaltCondition::MSE((EPSILON as f64)))
            .log_interval(Some(1))
            .momentum(0.1)
            .rate(0.3)
            .progress_callback(move |_progress, error| {
                let mut epoch_count = epoch_count_for_training.lock().unwrap();
                *current_error.lock().unwrap() = error;
                *count_epochs.lock().unwrap() = _progress;

                // *epoch_count += 1;
                let mut data = data_for_training.lock().unwrap();
                data.push((error / EPSILON as f64) as u64); // Add the current epoch count for display
                                                            // if data.len() > 10 { data.remove(0); } // Limit vector length
            })
            .go();
    });

    // Terminal setup for Ratatui
    enable_raw_mode()?;
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let refresh_rate = Duration::from_millis(100);
    let mut last_update = Instant::now();

    // clear screen
    terminal.clear()?;

    loop {
        // Quit on 'q' press
        if event::poll(Duration::from_millis(500))? {
            if let event::Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }

        // Update data for Ratatui at intervals
        if last_update.elapsed() >= refresh_rate {
            last_update = Instant::now();

            let data = data.lock().unwrap().clone();

            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .margin(0)
                    .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
                    .split(f.area());

                // Draw the topology on a canvas in the top part
                let canvas = Canvas::default()
                    .block(
                        Block::default()
                            .title("Neural Network Topology")
                            .borders(Borders::ALL),
                    )
                    .paint(|ctx| {
                        let width = chunks[0].width as f64;
                        let height = chunks[0].height as f64;

                        // Calculate layer positions
                        let layer_spacing = width / (layers.len() as f64 + 1.0);
                        for (i, &nodes) in layers.iter().enumerate() {
                            let x = (i as f64 + 1.0) * layer_spacing;

                            // Calculate node positions within each layer
                            let node_spacing = height / (nodes as f64 + 1.0);
                            for j in 0..nodes {
                                let y = (j as f64 + 1.0) * node_spacing;


                                ctx.draw(&Rectangle {
                                    x: x - 1.0,
                                    y: y - 1.0,
                                    width: 2.0,
                                    height: 2.0,
                                    color: Color::White,
                                });

                                // Draw multiple concentric circles to simulate a filled circle
                                // let radius = 1.0; // Adjust to set the circle's overall size
                                // let fill_steps = 5; // Number of circles to draw for filling effect

                                // Draw lines connecting nodes to the next layer
                                if i < layers.len() - 1 {
                                    let next_nodes = layers[i + 1];
                                    let next_node_spacing = height / (next_nodes as f64 + 1.0);
                                    for k in 0..next_nodes {
                                        let next_y = (k as f64 + 1.0) * next_node_spacing;
                                        let next_x = (i as f64 + 2.0) * layer_spacing;
                                        ctx.draw(&ratatui::widgets::canvas::Line {
                                            x1: x,
                                            y1: y,
                                            x2: next_x,
                                            y2: next_y,
                                            color: Color::Blue,
                                        });
                                    }
                                }

                                // for k in (0..=fill_steps).rev() {
                                //     ctx.draw(&Circle {
                                //         x,
                                //         y,
                                //         radius: radius * (k as f64 / fill_steps as f64),
                                //         color: Color::White,
                                //     });
                                // }

                            }
                        }
                    })
                    .x_bounds([0.0, chunks[0].width as f64])
                    .y_bounds([0.0, chunks[0].height as f64]);

                f.render_widget(canvas, chunks[0]);

                // Further split chunks[0] to add text in the top-right corner
                let sparkline_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(10), Constraint::Percentage(90)].as_ref())
                    .split(chunks[1]);

                let display_data = {
                    let width = sparkline_chunks[0].width as usize;

                    // If data length exceeds width, take only the last `width` elements
                    if data.len() > width {
                        data[data.len() - width..].to_vec()
                    } else {
                        data.clone() // If data is within width, clone the whole data
                    }
                };

                let spark = Sparkline::default()
                    .block(
                        Block::default()
                            .title("Training")
                            .borders(ratatui::widgets::Borders::ALL),
                    )
                    .data(&display_data)
                    .direction(RenderDirection::LeftToRight)
                    .style(Style::default().fg(Color::Red))
                    .absent_value_symbol(symbols::shade::FULL);

                f.render_widget(spark, sparkline_chunks[1]);

                // Display the top-right text
                let epoch_count = *count_epochs_for_ui.lock().unwrap();
                let error = *current_error_for_ui.lock().unwrap();

                let bottom_left_text = Paragraph::new(format!("Epoch: {}", epoch_count))
                    .style(Style::default().fg(Color::White))
                    .alignment(ratatui::layout::Alignment::Left);

                let top_right_text = Paragraph::new(format!("Error: {:.6}", error))
                    .style(Style::default().fg(Color::White))
                    .alignment(ratatui::layout::Alignment::Right);

                f.render_widget(bottom_left_text, sparkline_chunks[0]);

                f.render_widget(top_right_text, sparkline_chunks[0]);

                // let paragraph = Paragraph::new("Hello, world!").block(Block::bordered().title("Paragraph"));
                // f.render_widget(paragraph, chunks[1]);
            })?;
        }
    }

    // Clean up the terminal
    disable_raw_mode()?;
    terminal.show_cursor()?;

    Ok(())
}
