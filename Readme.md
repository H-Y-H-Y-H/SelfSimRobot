# Fully_body_VSM: Teaching Robots to Build Simulations of Themselves

Fully_body_VSM is a project aimed at enabling robots to understand and predict their own physical presence in the world through a single camera. Our system teaches robots to build accurate simulations of themselves, enhancing their ability to interact with the real-world environment. This repository contains all the code and models to set up and reproduce our results.

## Environment Setup

To run the Fully_body_VSM project, you need to have Python 3.9 installed on your system. If you do not have Python installed, we recommend downloading it from the official Python website or using a package manager like Homebrew for macOS, apt for Ubuntu, or Chocolatey for Windows.

### Dependencies

Once Python is installed, you will need to install the project dependencies. Clone this repository to your local machine, navigate to the cloned directory, and run the following command:

```bash
pip install -r requirements.txt
```

This command installs all the necessary Python packages listed in the requirements.txt file.

Running the Simulation  
To run the robot body prediction simulation in real-time with a graphical user interface (GUI), use the visualize_bullet.py script included in this repository. The script can be executed with the following command:

```bash
python visualize_bullet.py
```

We have provided pre-trained models to help you get started without the need to train the models from scratch. These models are located in the train_log folder. The visualize_bullet.py script automatically uses these models to run the simulations.



