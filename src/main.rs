use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, MouseButton, MouseEventKind,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use rayon::vec;
use std::{
    io::{self, stdout},
    process::exit,
};

use rusty_net::NN;
use rusty_net::{
    HaltCondition::{self, Epochs},
    LossFunction,
};

use csv::ReaderBuilder;
use std::fs::File;

const CANVAS_SIZE: usize = 28;

fn render_guess_widget(f: &mut ratatui::Frame, area: Rect, predictions: &[f64]) {
    let block = Block::default()
        .title(Span::styled(
            " Predictions ",
            Style::default().fg(Color::Cyan),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::White));

    let inner_area = block.inner(area);
    f.render_widget(block, area);

    let exp_sum: f64 = predictions.iter().map(|&x| x.exp()).sum();
    let softmax: Vec<f64> = predictions.iter().map(|&x| x.exp() / exp_sum).collect();

    let mut spans = Vec::new();
    let mut max_value = 0.0;
    let mut max_index = 0;

    for (i, &pred) in softmax.iter().enumerate() {
        if pred > max_value {
            max_value = pred;
            max_index = i;
        }
    }

    spans.push(Line::from(Span::styled(
        "Most likely digit:",
        Style::default().fg(Color::White),
    )));
    spans.push(Line::from(""));

    // Display the maximum prediction with larger emphasis
    let max_text = format!("   {} ({:.1}%)", max_index, max_value * 100.0);
    spans.push(Line::from(vec![Span::styled(
        max_text,
        Style::default().fg(Color::Green),
    )]));
    spans.push(Line::from(""));
    spans.push(Line::from(Span::styled(
        "All predictions:",
        Style::default().fg(Color::White),
    )));
    spans.push(Line::from(""));

    for (i, &pred) in softmax.iter().enumerate() {
        let percentage = (pred * 100.0).round();
        let color = if i == max_index {
            Color::Green
        } else {
            Color::Gray
        };

        let text = format!("  {}: {: >5.1}%", i, percentage);
        spans.push(Line::from(vec![Span::styled(
            text,
            Style::default().fg(color),
        )]));
    }

    let pred_paragraph = Paragraph::new(spans).style(Style::default().fg(Color::White));

    f.render_widget(pred_paragraph, inner_area);
}

fn getchar() {
    let mut input: String = String::new();
    let string = std::io::stdin().read_line(&mut input);
}

fn read_file(path: &str) -> Vec<(Vec<f64>, Vec<f64>)> {
    let file = File::open(path).unwrap();

    let mut rdr = ReaderBuilder::new().from_reader(file);

    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut outputs: Vec<Vec<f64>> = Vec::new();

    for result in rdr.records() {
        let record = result.unwrap();
        let mut input = Vec::with_capacity(784);
        let mut output = vec![0.0; 10];

        for (i, value) in record.iter().enumerate() {
            if i == 0 {
                output[value.parse::<usize>().unwrap()] = 1.0;
            } else {
                input.push(value.parse::<f64>().unwrap() / 255.0);
            }
        }

        inputs.push(input);
        outputs.push(output);
    }

    let examples: Vec<(Vec<f64>, Vec<f64>)> = inputs.into_iter().zip(outputs.into_iter()).collect();

    examples
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let nn = NN::load_from_json("nn.json").unwrap();


    // exit(1);

    let mut nn = NN::new(&vec![784, 512, 10]);
    nn.activation(rusty_net::ActivationFunction::LeakyReLU);
    let examples = read_file("samples/train2.csv");
    nn.train(&examples)
        .halt_condition(HaltCondition::Epochs(50))
        .log_interval(Some(1))
        .rate(0.002)
        .momentum(0.9)
        .loss_function(LossFunction::CrossEntropy)
        .go();

    nn.save_as_json("nn.json");




    let examples = read_file("./samples/test.csv");

    let mut count_correct = 0;

    for example in &examples {
        let (input, output) = example;
        let guess = nn.run(&input);

        // Calculate the softmax (if you still need it, but seems like you want to create labels instead)
        let exp_sum: f64 = guess.iter().map(|&x| x.exp()).sum();
        let softmax: Vec<f64> = guess.iter().map(|&x| x.exp() / exp_sum).collect();

        // Print softmax guess for debugging
        println!("Softmax Guess: {:?}", softmax);

        // Create a label vector where the largest value's index is set to 1.0, others to 0.0
        let mut label = vec![0.0; softmax.len()];
        if let Some((max_index, _)) =
            softmax
                .iter()
                .enumerate()
                .fold(None, |acc, (index, &value)| match acc {
                    Some((max_idx, max_val)) if value > max_val => Some((index, value)),
                    None => Some((index, value)),
                    _ => acc,
                })
        {
            label[max_index] = 1.0;
        }

        // Print the created label vector
        println!("Label: {:?}", label);

        println!("Output: {:?}", output);

        if label == *output {
            count_correct += 1;
        }
    }

    println!("Correct: {}", count_correct);
    println!("Total: {}", examples.len());
    println!("Accuracy: {}", count_correct as f64 / examples.len() as f64);





    exit(1);

    let mut stdout = stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    crossterm::terminal::enable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), EnableMouseCapture)?;
    let mut canvas = [[false; CANVAS_SIZE]; CANVAS_SIZE];
    let mut drawing = false;

    terminal.clear()?;

    loop {
        let canvas_float = canvas
            .iter()
            .flatten()
            .map(|&x| if x { 1.0 } else { 0.0 })
            .collect::<Vec<f64>>();
        let predictions = nn.run(&canvas_float);

        let size = terminal.size()?;
        let cell_width = 2;
        let cell_height = 1;
        let canvas_widget_width = CANVAS_SIZE as u16 * cell_width + 2; // +2 for borders
        let canvas_widget_height = CANVAS_SIZE as u16 * cell_height + 2;

        // Center the canvas vertically
        let y_offset = (size.height - canvas_widget_height) / 2;
        let canvas_area = Rect::new(0, y_offset, canvas_widget_width, canvas_widget_height);


        if size.width < canvas_widget_width + 10 || size.height < canvas_widget_height + 10 {
            terminal.draw(|f| {
                let warning = Paragraph::new(Span::styled(
                    "Terminal too small. Please resize.",
                    Style::default().fg(Color::Red),
                ));
                f.render_widget(warning, Rect::new(0, 0, size.width, size.height));
            })?;
            continue;
        }

                // Handle mouse events
                if event::poll(std::time::Duration::from_millis(10))? {
                    match event::read()? {
                        Event::Mouse(event) => {
                            let x = event.column as u16;
                            let y = event.row as u16;
        
                            // Check for reset button click
                            if x >= canvas_area.x + canvas_area.width - 9
                                && x < canvas_area.x + canvas_area.width - 2
                                && y >= canvas_area.y + canvas_area.height + 1
                                && y < canvas_area.y + canvas_area.height + 2
                                && event.kind == MouseEventKind::Down(MouseButton::Left)
                            {
                                canvas = [[false; CANVAS_SIZE]; CANVAS_SIZE];
                            } else {
                                // Handle drawing on canvas
                                let block_x = ((x as usize).saturating_sub(canvas_area.x as usize + 1))
                                    / cell_width as usize;
                                let block_y = ((y as usize).saturating_sub(canvas_area.y as usize + 1))
                                    / cell_height as usize;
        
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
                        Event::Key(key) if key.code == KeyCode::Char('q') => {
                            crossterm::execute!(terminal.backend_mut(), DisableMouseCapture)?;
                            crossterm::terminal::disable_raw_mode()?;
                            terminal.show_cursor()?;
                            break;
                        }
  
                        _ => {}
                    }
                }


        terminal.draw(|f| {
            // Create a horizontal layout
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(canvas_widget_width + 4), // Canvas width + padding
                    Constraint::Min(30),                         // Predictions width
                ])
                .margin(1)
                .split(f.area());

            // Canvas area with vertical centering
            let canvas_area = Rect::new(
                chunks[0].x,
                y_offset,
                canvas_widget_width,
                canvas_widget_height,
            );

            // Reset button below the canvas
            let reset_button_area = Rect::new(
                canvas_area.x + canvas_area.width - 9,
                canvas_area.y + canvas_area.height + 1,
                7,
                1,
            );

            // Create bordered canvas block
            let block = Block::default()
                .title(Span::styled(
                    " Draw a number! ",
                    Style::default().fg(Color::Cyan),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White));

            f.render_widget(block, canvas_area);

            // Render the reset button
            let reset_button = Paragraph::new(Span::styled(
                " Reset ",
                Style::default().bg(Color::Red).fg(Color::White),
            ));
            f.render_widget(reset_button, reset_button_area);

            // Render canvas cells
            for y in 0..CANVAS_SIZE {
                for x in 0..CANVAS_SIZE {
                    let color = if canvas[y][x] {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let cell = Paragraph::new(Span::styled("  ", Style::default().bg(color)));
                    let area = Rect::new(
                        canvas_area.x + 1 + x as u16 * cell_width,
                        canvas_area.y + 1 + y as u16 * cell_height,
                        cell_width,
                        cell_height,
                    );
                    f.render_widget(cell, area);
                }
            }

            // Render the predictions widget
            let predictions_area =
                Rect::new(chunks[1].x, y_offset, chunks[1].width, canvas_widget_height);
            render_guess_widget(f, predictions_area, &predictions);
        })?;
    }

    Ok(())
}
