# Trying out SMARTS: Multi-agent RL for Autonomous Driving
- Author: Ismail Geles
- Date: 19.06.2021

Homework project for the subject of *Selected Topics of Control & Dynamic Systems* at TU Graz.  
Some slight modifications have been added compared to the base SMARTS repo, check latest commit of forked repo or comments below.  
Original paper: https://arxiv.org/abs/2010.09776  
Original repository: https://github.com/huawei-noah/SMARTS  

## Setting up SMARTS
- using normal package installation, I encountered some errors on MacOS (all packages installed successfully, but tests are failing)
- Docker works but inside the docker container, you might need to install ijson and cached_property using pip  
	```bash
	pip install ijson
	pip install cached_property
	```
- or just install the missing packages again with the requirements file:
   ``` bash
   pip install -r requirements.txt
   ```
 
### Using Docker 

steps (mostly from the SMARTS repo) to start the environment with Docker  

```bash
cd /path/to/SMARTS
docker run --rm -it -v $PWD:/src -p 8081:8081 huaweinoah/smarts:<version>
# E.g. docker run --rm -it -v $PWD:/src -p 8081:8081 huaweinoah/smarts:v0.4.12
# <press enter>

# Run Envision server in the background
# This will only need to be run if you want visualisation
scl envision start -s ./scenarios -p 8081 &
```
	
## Example with Visualisation
following steps to be done if visualization is wanted and how to execute an example

```bash
# Build an example
# This needs to be done the first time and after changes to the example
scl scenario build scenarios/loop --clean

# Run an example
# add --headless if you do not need visualisation
python examples/single_agent.py scenarios/loop

# On your host machine visit http://localhost:8081  
# to see the running simulation in Envision.
```
 

 
## Another Example with more Agents
this might currently result in some errors (SMARTS release version 0.4), but it is good for visualisation of potential use cases of SMARTS 

```bash
scl scenario build scenarios/figure_eight --clean
python examples/multi_agent.py scenarios/figure_eight
```
  
## RL Benchmarks
- test some environments with different agents in benchmark folder (which is placed in SMARTS folder)
- reduced the total time steps to 5000 (from 14400) and increased the default number of workers to 3 (from 1 worker).
- some necessary modifications are desribed in each '# note' below

  ```bash
  cd benchmark
  # note: there was a typo in benchmark/scenarios/double_merge/cross/scenario.py, changed "overrite" to "overwrite"
  
  # only if visualisation needed, can be removed using --headless argument
  scl envision start -s ./scenarios -p 8081 &
  scl scenario build ./scenarios/double_merge/cross --clean
  
  # note: removed a ".parent" in the file: /benchmark/agents/__init__.py (line 149)
  python run.py ./scenarios/double_merge/cross -f ./agents/ppo/baseline-lane-control.yaml --num_workers 3 --headless
  
  # ... wait until training finished (~1.5h for 3 workers and 5000 steps)
  # meanwhile one can check tensorboard in benchmark/log folder execute: tensorboard --logdir ./results
  
  # train again with lr increased to 1e-3 from 1e-4 and SGD-batch_size increased to 64 from 32
  python run.py ./scenarios/double_merge/cross -f ./agents/ppo/baseline-lane-control.yaml --num_workers 3 --headless
  # train again with lr increased to 3e-4 from 1e-4 and SGD-batch_size increased to 64 from 32
  python run.py ./scenarios/double_merge/cross -f ./agents/ppo/baseline-lane-control.yaml --num_workers 3 --headless
  
  # other agents
  # note: ray.rllib.utils.types is now ray.rllib.utils.typing (changed in benchmark/networks)
  python run.py ./scenarios/double_merge/cross -f ./agents/maac/baseline-lane-control.yaml --num_workers 3 --headless
  python run.py ./scenarios/double_merge/cross -f ./agents/mfac/baseline-lane-control.yaml --num_workers 3 --headless
  python run.py ./scenarios/double_merge/cross -f ./agents/dqn/baseline-lane-control.yaml --num_workers 3 --headless
  python run.py ./scenarios/double_merge/cross -f ./agents/a2c/baseline-lane-control.yaml --num_workers 3 --headless
  
  # maddpg and networked_pg had some errors...
  
  # compare mean reward in tensorboard for improvement check...
  ```

## Overall Impression

- a huge project with lots of flexibility and options to make your own environments, agents, or use a few common MARL libraries
- many bugs and still at early stage, many errors occur during runtime...
- **conclusion:** future stable versions might be very useful for autonomous driving and RL research but currenlty (release version 0.4) there is still a lot to be fixed 




## Other notes...

if the python/ray processes do not quit by themselves (check with ```top``` or ```htop```),  
force kill python/ray: 

```bash
pkill python
pkill ray
```

other problems that I noticed:

```bash
### TODO: evaluate has some problems that could not be resolved by myself...
apt-get update
apt-get install rsync grsync
python evaluate.py ./scenarios/double_merge/cross -f ./agents/ppo/baseline-lane-control.yaml \\  
--log_dir ./log/results/run/cross-4/PPO_FrameStack_77d6d_00000_0_2021-06-20_14-19-04/  --plot  
# also the last model checkout fails because of some errors...
###
```
