use std::time::{Duration, Instant};
use std::io;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::Color;
use ratatui::widgets::canvas::{Canvas, Line, Rectangle};
use ratatui::widgets::{Block, Borders};
use ratatui::Terminal;
use crossterm::event::{self, KeyCode};
use crossterm::terminal::{enable_raw_mode, disable_raw_mode};

fn main() -> Result<(), io::Error> {
    // Example layer structure for the neural network
    let layers = vec![2, 7, 5, 2, 4, 1];

    enable_raw_mode()?;
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let refresh_rate = Duration::from_millis(100);
    let mut last_update = Instant::now();
    terminal.clear()?;

    let mut zoom_level = 1.0;  // Initial zoom level
    let mut pan_x = 0.0;
    let mut pan_y = 0.0;
    let zoom_factor = 1.2;  // 

    loop {
        if event::poll(Duration::from_millis(100))? {
            if let event::Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Down => pan_y -= 10.0 / zoom_level, // Move up
                    KeyCode::Up => pan_y += 10.0 / zoom_level, // Move down
                    KeyCode::Left => pan_x -= 10.0 / zoom_level, // Move left
                    KeyCode::Right => pan_x += 10.0 / zoom_level, // Move right
                    KeyCode::Char('+') => zoom_level *= zoom_factor, // Zoom in
                    KeyCode::Char('-') => zoom_level /= zoom_factor, // Zoom out
                    _ => {}
                }
            }
        }

        if last_update.elapsed() >= refresh_rate {
            last_update = Instant::now();
            
            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .margin(0)
                    .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
                    .split(f.area());
                let x_min = pan_x;
                let x_max = pan_x + chunks[0].width as f64 / zoom_level;
                let y_min = pan_y;
                let y_max = pan_y + chunks[0].height as f64 / zoom_level;
                
                let canvas = Canvas::default()
                    .block(
                        Block::default()
                            .title("Neural Network Topology")
                            .borders(Borders::ALL),
                    )
                    .paint(|ctx| {
                        let width = chunks[0].width as f64;
                        let height = chunks[0].height as f64;
        
                        let layer_spacing = width / (layers.len() as f64 + 1.0);
                        for (i, &nodes) in layers.iter().enumerate() {
                            let x = (i as f64 + 1.0) * layer_spacing;
        
                            let node_spacing = height / (nodes as f64 + 1.0);
                            for j in 0..nodes {
                                let y = (j as f64 + 1.0) * node_spacing;
        
                                if i < layers.len() - 1 {
                                    let next_nodes = layers[i + 1];
                                    let next_node_spacing = height / (next_nodes as f64 + 1.0);
                                    for k in 0..next_nodes {
                                        let next_y = (k as f64 + 1.0) * next_node_spacing;
                                        let next_x = (i as f64 + 2.0) * layer_spacing;

                                        // Draw lines connecting nodes to the next layer regardless of bounds
                                        ctx.draw(&Line {
                                            x1: x,
                                            y1: y,
                                            x2: next_x,
                                            y2: next_y,
                                            color: Color::Blue,
                                        });
                                    }
                                }
        
                                // Draw the node as a rectangle
                                ctx.draw(&Rectangle {
                                    x: x - 1.0,
                                    y: y - 1.0,
                                    width: 2.0,
                                    height: 2.0,
                                    color: Color::White,
                                });
                            }
                        }
                    })
                    .x_bounds([x_min, x_max])
                    .y_bounds([y_min, y_max]);
        
                f.render_widget(canvas, chunks[0]);
            })?;
        }
    }

    disable_raw_mode()?;
    terminal.show_cursor()?;
    Ok(())
}
