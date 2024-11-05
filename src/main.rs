use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, MouseButton, MouseEventKind};
use ratatui::{
    backend::CrosstermBackend,
    layout::Rect,
    style::{Color, Style},
    text::Span,
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use std::{io::{self, stdout}, process::exit};

use rusty_net::NN;
use rusty_net::HaltCondition::Epochs;

use csv::ReaderBuilder;
use std::fs::File;


const CANVAS_SIZE: usize = 28;

fn getchar() {
    let mut input: String = String::new();
    let string = std::io::stdin().read_line(&mut input);
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize terminal and configure backend
    

    let mut nn  = NN::new(&vec![784, 512, 10]);

    let file = File::open("./samples/train2.csv")?;

    // Create a CSV reader with default options.
    let mut rdr = ReaderBuilder::new().from_reader(file);

    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut outputs: Vec<Vec<f64>> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let mut input = Vec::with_capacity(784);
        let mut output = vec![0.0; 10];
        
        for (i, value) in record.iter().enumerate() {
            if i == 0 {
                output[value.parse::<usize>()?] = 1.0;
            } else {
                input.push(value.parse::<f64>()?/255.0);
            }
        }

        println!("{:?}",output);
        inputs.push(input);
        outputs.push(output);
    }

    let examples: Vec<(Vec<f64>, Vec<f64>)> = inputs.into_iter().zip(outputs.into_iter()).collect();

    println!("Training...");

    nn.train(&examples).momentum(0.9).log_interval(Some(1)).halt_condition(Epochs(50)).rate(0.001).go();

    println!("Training done!");


    // Create a CSV reader with default options


    for example in examples {
        let (input, output) = example;
        let guess = nn.run(&input);
    
        // Calculate the softmax (if you still need it, but seems like you want to create labels instead)
        let exp_sum: f64 = guess.iter().map(|&x| x.exp()).sum();
        let softmax: Vec<f64> = guess.iter().map(|&x| x.exp() / exp_sum).collect();
    
        // Print softmax guess for debugging
        println!("Softmax Guess: {:?}", softmax);
    
        // Create a label vector where the largest value's index is set to 1.0, others to 0.0
        let mut label = vec![0.0; softmax.len()];
        if let Some((max_index, _)) = softmax
            .iter()
            .enumerate()
            .fold(None, |acc, (index, &value)| {
                match acc {
                    Some((max_idx, max_val)) if value > max_val => Some((index, value)),
                    None => Some((index, value)),
                    _ => acc,
                }
            }) 
        {
            label[max_index] = 1.0;
        }
    
        // Print the created label vector
        println!("Label: {:?}", label);
        println!("Output: {:?}", output);
        getchar();
    }




    exit(0);

    let mut stdout = stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Enable raw mode and mouse capture for terminal
    crossterm::terminal::enable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), EnableMouseCapture)?;

    // Initialize canvas and drawing flag
    let mut canvas = [[false; CANVAS_SIZE]; CANVAS_SIZE];
    let mut drawing = false;

    // Clear terminal on start
    terminal.clear()?;

    loop {
        // Calculate terminal size and widget dimensions
        let size = terminal.size()?;
        let cell_width = 2; // Each block occupies 2 columns width-wise
        let cell_height = 1;
        let widget_width = CANVAS_SIZE as u16 * cell_width;
        let widget_height = CANVAS_SIZE as u16 * cell_height;

        // Positioning offsets for the canvas and reset button
        let x_offset = 0;
        let y_offset = 0;

        // Define the area for the canvas block with a bottom-right reset button
        let block_area = Rect::new(x_offset, y_offset, widget_width + 2, widget_height + 3);
        let reset_button_area = Rect::new(
            block_area.x + block_area.width - 9,  // Right-aligned within the canvas widget
            block_area.y + block_area.height - 2, // Positioned at the bottom of the widget
            7,
            1,
        );

        // Draw canvas, reset button, and grid
        terminal.draw(|f| {
            // Create bordered canvas block
            let block = Block::default()
                .title(Span::styled("Draw a number!", Style::default().fg(Color::Cyan)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White));

            f.render_widget(block, block_area);

            // Render the reset button at the bottom-right of the canvas
            let reset_button = Paragraph::new(Span::styled(" Reset ", Style::default().bg(Color::Red).fg(Color::White)));
            f.render_widget(reset_button, reset_button_area);

            // Render each cell in the 28x28 grid, each occupying a 2x1 area on the screen
            for y in 0..CANVAS_SIZE {
                for x in 0..CANVAS_SIZE {
                    let color = if canvas[y][x] { Color::White } else { Color::Black };
                    let cell = Paragraph::new(Span::styled("  ", Style::default().bg(color))); // Two spaces to cover 2 columns
                    let area = Rect::new(
                        x_offset + 1 + x as u16 * cell_width, // Adjust for left border
                        y_offset + 1 + y as u16 * cell_height, // Adjust for top border
                        cell_width,
                        cell_height,
                    );
                    f.render_widget(cell, area);
                }
            }
        })?;

        // Handle events (mouse and keyboard)
        if event::poll(std::time::Duration::from_millis(10))? {
            match event::read()? {
                Event::Mouse(event) => {
                    let x = event.column as u16;
                    let y = event.row as u16;

                    // Check if the reset button was clicked
                    if x >= reset_button_area.x
                        && x < reset_button_area.x + reset_button_area.width
                        && y >= reset_button_area.y
                        && y < reset_button_area.y + reset_button_area.height
                        && event.kind == MouseEventKind::Down(MouseButton::Left)
                    {
                        canvas = [[false; CANVAS_SIZE]; CANVAS_SIZE]; // Clear canvas
                    } else {
                        // Calculate block coordinates in the canvas grid
                        let block_x = ((x as usize).saturating_sub(x_offset as usize + 1)) / cell_width as usize;
                        let block_y = ((y as usize).saturating_sub(y_offset as usize + 1)) / cell_height as usize;

                        // Draw on the canvas if within bounds
                        if block_x < CANVAS_SIZE && block_y < CANVAS_SIZE {
                            match event.kind {
                                MouseEventKind::Down(_) => {
                                    drawing = true;
                                    canvas[block_y][block_x] = true;
                                }
                                MouseEventKind::Drag(_) if drawing => {
                                    canvas[block_y][block_x] = true;
                                }
                                MouseEventKind::Up(_) => {
                                    drawing = false;
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Event::Key(key) if key.code == KeyCode::Char('q') => break, // Quit on 'q' key
                _ => {}
            }
        }
    }

    // Restore terminal settings
    crossterm::execute!(terminal.backend_mut(), DisableMouseCapture)?;
    crossterm::terminal::disable_raw_mode()?;
    terminal.show_cursor()?;

    Ok(())
}
