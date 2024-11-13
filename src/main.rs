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
use std::io::stdout;
use rusty_net::NN;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nn = NN::load_from_json("nn.json").unwrap();

    let stdout = stdout();
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

                if event::poll(std::time::Duration::from_millis(10))? {
                    match event::read()? {
                        Event::Mouse(event) => {
                            let x = event.column as u16;
                            let y = event.row as u16;
        
                            if x >= canvas_area.x + canvas_area.width - 9
                                && x < canvas_area.x + canvas_area.width - 2
                                && y >= canvas_area.y + canvas_area.height + 1
                                && y < canvas_area.y + canvas_area.height + 2
                                && event.kind == MouseEventKind::Down(MouseButton::Left)
                            {
                                canvas = [[false; CANVAS_SIZE]; CANVAS_SIZE];
                            } else {
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
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(canvas_widget_width + 4),
                    Constraint::Min(30),                         
                ])
                .margin(1)
                .split(f.area());


            let canvas_area = Rect::new(
                chunks[0].x,
                y_offset,
                canvas_widget_width,
                canvas_widget_height,
            );

            let reset_button_area = Rect::new(
                canvas_area.x + canvas_area.width - 9,
                canvas_area.y + canvas_area.height + 1,
                7,
                1,
            );

            let block = Block::default()
                .title(Span::styled(
                    " Draw a number! ",
                    Style::default().fg(Color::Cyan),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White));

            f.render_widget(block, canvas_area);

            let reset_button = Paragraph::new(Span::styled(
                " Reset ",
                Style::default().bg(Color::Red).fg(Color::White),
            ));
            f.render_widget(reset_button, reset_button_area);

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

            let predictions_area =
                Rect::new(chunks[1].x, y_offset, chunks[1].width, canvas_widget_height);
            render_guess_widget(f, predictions_area, &predictions);
        })?;
    }

    Ok(())
}
