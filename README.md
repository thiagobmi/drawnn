![octologo_drawnn_1731471595](https://github.com/user-attachments/assets/4350a3ac-cea5-46e3-812f-d5eefa85c193)

# Overview
Drawnn is a simple terminal-based application that allows you to draw digits using your mouse and get real-time predictions from a neural network classifier.

The neural network is a multi-layer perceptron trained on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) using [rusty_net](https://github.com/thiagobmi/rusty_net).

<p align="center">
  <img src="https://github.com/user-attachments/assets/33aaa9fd-2fb5-4ae8-b79f-5343dcd2c66a" alt="description"/>
</p>

# Dependencies
- [rusty_net](https://github.com/thiagobmi/rusty_net) - My neural network lib.
- ratatui - Terminal user interface library.
- crossterm - Enables mouse in terminal.

# How it works
- The application loads pre-trained neural network weights from `nn.json`
- The drawing interface captures your input in a 28x28 grid (MNIST format)
- The neural network processes the drawing in real-time
- Predictions and confidence scores are updated continuously
