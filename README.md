## Installation

1. **Install the Python development environment** 

	Check if your Python environment is already configured on your system. 
	We recommend using Python 3.
	```bash
	python --version
	pip --version
	virtualenv --version
	```

	If these packages are already installed, skip to the next step.
	Otherwise, install on Ubuntu by running:
	```bash
	sudo apt update
	sudo apt install python-dev python-pip
	sudo pip install -U virtualenv  # system-wide install
	```

2. **Create a virtual environment (recommended)** 

	Create a new virtual environment in the root directory or anywhere else:
	```bash
	virtualenv --system-site-packages -p python3 .venv
	```

	Activate the virtual environment every time before you use the package:
	```bash
	source .venv/bin/activate
	```

	And exit the virtual environment when you are done:
	```bash
	deactivate
	```

3. **Install dependencies** 

	The package can be installed by running:
	  ```bash
	  pip install -r requirements.txt
	  ```

## Examples

Run training and evaluation:
```bash
python train_eval_ppo.py --root_dir outputs
```
