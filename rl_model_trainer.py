import os
import json
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import Dense, Flatten, InputLayer  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from collections import deque
import random
import datetime
import glob
import time
from simulation.environment import DeadlockEnvironment


# Create model directory
os.makedirs("model", exist_ok=True)

class DeadlockPreventionAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural network for deep Q learning
        model = Sequential()
        model.add(InputLayer(input_shape=(self.state_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        # Epsilon-greedy action selection
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # Training from experience replay
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def create_state_vector(state_data, num_processes, num_resources):
    """Convert a simulation state to a flat vector for the RL model"""
    allocation = np.array(state_data['allocation'])
    request = np.array(state_data['request'])
    available = np.array(state_data['available'])
    
    # Flatten and concatenate
    state_vector = np.concatenate([
        allocation.flatten(),
        request.flatten(),
        available.flatten(),
        np.array(state_data['completed'])
    ])
    
    return state_vector


def get_action_space_size(num_processes, num_resources):
    """
    Calculate action space size:
    For each process-resource pair, we can:
    1. Allocate requested resources
    2. Preempt resources from a process
    """
    return num_processes * num_resources * 2


def action_to_allocation_decision(action, num_processes, num_resources):
    """Convert action index to process ID, resource ID and action type"""
    total_pairs = num_processes * num_resources
    
    if action < total_pairs:
        # Allocate resources
        process_id = action // num_resources
        resource_id = action % num_resources
        action_type = "allocate"
    else:
        # Preempt resources
        adjusted_action = action - total_pairs
        process_id = adjusted_action // num_resources
        resource_id = adjusted_action % num_resources
        action_type = "preempt"
    
    return process_id, resource_id, action_type


def load_simulation_data(base_dir="simulation_results", max_simulations=None):
    """Load simulation data from JSON files for training"""
    simulation_dirs = sorted(glob.glob(f"{base_dir}/simulation_*"))
    if not simulation_dirs:
        print(f"No simulation directories found in {base_dir}")
        return None, None, None
    
    # List all simulation directories with timestamps
    print("\nAvailable simulation datasets:")
    for i, sim_dir in enumerate(simulation_dirs):
        try:
            with open(f"{sim_dir}/summary.json", "r") as f:
                summary = json.load(f)
                timestamp = summary.get('timestamp', 'Unknown date')
                num_sims = summary.get('total_simulations', 'Unknown')
                processes = summary.get('parameters', {}).get('num_processes', 'Unknown')
                resources = summary.get('parameters', {}).get('num_resources', 'Unknown')
                print(f"{i+1}. {os.path.basename(sim_dir)} - {timestamp} - {num_sims} simulations - {processes} processes, {resources} resources")
        except:
            print(f"{i+1}. {os.path.basename(sim_dir)} - No summary available")
    
    # Let user select a simulation directory
    while True:
        try:
            choice = int(input("\nSelect a simulation dataset (number) or 0 to cancel: "))
            if choice == 0:
                return None, None, None
            if 1 <= choice <= len(simulation_dirs):
                selected_dir = simulation_dirs[choice-1]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Load summary to get parameters
    try:
        with open(f"{selected_dir}/summary.json", "r") as f:
            summary = json.load(f)
    except FileNotFoundError:
        print(f"Summary file not found in {selected_dir}")
        return None, None, None
    
    # Extract parameters
    num_processes = summary['parameters']['num_processes']
    num_resources = summary['parameters']['num_resources']
    
    # Load all simulation files
    all_data = []
    simulation_files = sorted(glob.glob(f"{selected_dir}/[0-9]*.json"))
    
    if max_simulations:
        simulation_files = simulation_files[:max_simulations]
    
    for sim_file in simulation_files:
        try:
            with open(sim_file, "r") as f:
                sim_data = json.load(f)
                all_data.append(sim_data)
        except Exception as e:
            print(f"Error loading {sim_file}: {e}")
    
    if not all_data:
        print("No simulation data loaded")
        return None, None, None
    
    print(f"Loaded {len(all_data)} simulations from {selected_dir}")
    return all_data, num_processes, num_resources

def load_existing_model():
    """List and load existing models"""
    model_dirs = sorted(glob.glob("model/deadlock_agent_*"))
    if not model_dirs:
        print("No existing models found.")
        return None
    
    print("\nAvailable models:")
    for i, model_dir in enumerate(model_dirs):
        try:
            with open(f"{model_dir}/model_info.json", "r") as f:
                info = json.load(f)
                timestamp = info.get('timestamp', 'Unknown date')
                processes = info.get('num_processes', 'Unknown')
                resources = info.get('num_resources', 'Unknown')
                print(f"{i+1}. {os.path.basename(model_dir)} - {timestamp} - {processes} processes, {resources} resources")
        except:
            print(f"{i+1}. {os.path.basename(model_dir)} - No model info available")
    
    while True:
        try:
            choice = int(input("\nSelect a model (number) or 0 to cancel: "))
            if choice == 0:
                return None
            if 1 <= choice <= len(model_dirs):
                selected_dir = model_dirs[choice-1]
                
                # Load model information
                with open(f"{selected_dir}/model_info.json", "r") as f:
                    model_info = json.load(f)
                
                # Create agent with same parameters
                agent = DeadlockPreventionAgent(
                    model_info['state_size'], 
                    model_info['action_size']
                )
                
                # Load weights
                agent.load(f"{selected_dir}/weights.weights.h5")
                
                print(f"Loaded model from {selected_dir}")
                return agent, model_info
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    return None

def extract_training_sequences(simulation_data, num_processes, num_resources):
    """Extract state-action-reward-next_state sequences from simulation data"""
    training_data = []
    
    for sim in simulation_data:
        history = sim['history']
        deadlocked = sim['deadlocked']
        
        for i in range(len(history) - 1):
            current_state = create_state_vector(history[i], num_processes, num_resources)
            next_state = create_state_vector(history[i+1], num_processes, num_resources)
            
            # Determine allocation changes
            curr_allocation = np.array(history[i]['allocation'])
            next_allocation = np.array(history[i+1]['allocation'])
            diff = next_allocation - curr_allocation
            
            # Find what changed (simplified action extraction)
            actions = []
            for p in range(num_processes):
                for r in range(num_resources):
                    if diff[p][r] > 0:
                        # Resource was allocated
                        action_idx = p * num_resources + r
                        actions.append(action_idx)
                    elif diff[p][r] < 0:
                        # Resource was released/preempted
                        action_idx = (num_processes * num_resources) + (p * num_resources + r)
                        actions.append(action_idx)
            
            # Use first action if multiple changes (simplification)
            if actions:
                action = actions[0]
            else:
                # No change, pick random action
                action = random.randrange(get_action_space_size(num_processes, num_resources))
            
            # Reward calculation
            reward = 0
            
            # Positive reward for completing processes
            completed_diff = sum(history[i+1]['completed']) - sum(history[i]['completed'])
            if completed_diff > 0:
                reward += 10 * completed_diff
                
            # Small negative reward for each step (to encourage efficiency)
            reward -= 0.1
            
            # Large negative reward if we end in deadlock
            if i == len(history) - 2 and deadlocked:
                reward -= 50
                
            # Large positive reward if all processes complete
            if i == len(history) - 2 and all(history[i+1]['completed']):
                reward += 100
            
            # Package as training example
            is_done = (i == len(history) - 2)
            training_data.append((current_state, action, reward, next_state, is_done))
    
    return training_data


def run_simulation(num_processes, num_resources, max_resource_instances, steps=50, render=True):
    """Run a single simulation with given parameters"""
    # Create environment with increased deadlock probability
    env = DeadlockEnvironment(
        num_processes=num_processes,
        num_resources=num_resources,
        max_resource_instances=max_resource_instances,
        deadlock_prone=True  # Enable deadlock-prone mode
    )
    
    # Rest of the function remains the same
    state = env.reset()
    history = [state]
    
    for i in range(steps):
        state, done = env.step()
        history.append(state)
        
        if render:
            env.render()
            time.sleep(0.5)
        
        if done:
            break
    
    return {
        'deadlocked': state['deadlocked'],
        'steps': state['steps'],
        'completed_processes': sum(state['completed']),
        'final_state': state,
        'history': history,
        'config': {
            'num_processes': num_processes,
            'num_resources': num_resources,
            'max_resource_instances': max_resource_instances,
            'max_steps': steps
        }
    }

def get_user_input_for_simulations():
    """Get user input for multiple simulation runs"""
    print("\n=== DEADLOCK SIMULATION CONFIGURATION ===")
    
    # Get number of processes
    while True:
        try:
            num_processes = int(input("Enter number of processes (2-10): ").strip())
            if 2 <= num_processes <= 10:
                break
            print("Please enter a value between 2 and 10.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get number of resources
    while True:
        try:
            num_resources = int(input("Enter number of resource types (2-10): ").strip())
            if 2 <= num_resources <= 10:
                break
            print("Please enter a value between 2 and 10.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get max resource instances
    while True:
        try:
            max_resource_instances = int(input("Enter maximum instances per resource (1-5): ").strip())
            if 1 <= max_resource_instances <= 5:
                break
            print("Please enter a value between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get number of simulations
    while True:
        try:
            num_simulations = int(input("Enter number of simulations to run (1-100): ").strip())
            if 1 <= num_simulations <= 100:
                break
            print("Please enter a value between 1 and 100.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get maximum steps per simulation
    while True:
        try:
            max_steps = int(input("Enter maximum steps per simulation (10-100): ").strip())
            if 10 <= max_steps <= 100:
                break
            print("Please enter a value between 10 and 100.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get render option
    render = input("Render simulation steps? (y/n, default: n): ").strip().lower() == 'y'
    
    return {
        'num_processes': num_processes,
        'num_resources': num_resources,
        'max_resource_instances': max_resource_instances,
        'num_simulations': num_simulations,
        'max_steps': max_steps,
        'render': render
    }


def generate_multiple_simulations():
    """Generate multiple simulations for training"""
    # Get user input for simulation parameters
    params = get_user_input_for_simulations()
    
    num_simulations = params['num_simulations']
    print(f"\n=== RUNNING {num_simulations} SIMULATIONS ===")
    
    # Run simulations and collect results
    simulations = []
    deadlock_count = 0
    total_steps = 0
    completed_processes = 0
    
    for i in range(num_simulations):
        print(f"\nSimulation {i+1}/{num_simulations}")
        
        # Run simulation with user parameters
        results = run_simulation(
            params['num_processes'],
            params['num_resources'],
            params['max_resource_instances'],
            steps=params['max_steps'],
            render=params['render']
        )
        
        # Add to collection
        simulations.append(results)
        
        # Update statistics
        if results['deadlocked']:
            deadlock_count += 1
        total_steps += results['steps']
        completed_processes += results['completed_processes']
        
        # Print results for this simulation
        print(f"  Results: {'Deadlocked' if results['deadlocked'] else 'Completed'} after {results['steps']} steps")
        print(f"  Processes completed: {results['completed_processes']}/{params['num_processes']}")
    
    # Print summary statistics
    print("\n=== SIMULATION SUMMARY ===")
    print(f"Total simulations: {num_simulations}")
    print(f"Deadlocks: {deadlock_count}/{num_simulations} ({deadlock_count/num_simulations*100:.1f}%)")
    print(f"Average steps: {total_steps/num_simulations:.1f}")
    print(f"Average processes completed: {completed_processes/num_simulations:.1f}/{params['num_processes']}")
    
    return simulations, params['num_processes'], params['num_resources']


def train_rl_model():
    print("\n=== TRAINING RL MODEL FOR DEADLOCK PREVENTION ===")
    
    # Ask whether to continue training existing model or create new one
    print("\nOptions:")
    print("1. Train new model")
    print("2. Continue training existing model")
    print("3. Use existing simulation data")
    print("4. Generate new simulations")
    
    choice = 0
    while choice not in [1, 2, 3, 4]:
        try:
            choice = int(input("\nEnter your choice (1-4): "))
            if choice not in [1, 2, 3, 4]:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Load existing model or create new agent
    existing_agent = None
    model_info = None
    
    if choice == 2:
        # Load existing model
        result = load_existing_model()
        if not result:
            print("No model selected. Exiting.")
            return
        existing_agent, model_info = result
    
    # Load simulation data or generate new
    if choice in [1, 2]:
        # Ask for simulation data source
        data_choice = 0
        while data_choice not in [1, 2]:
            try:
                data_choice = int(input("\nUse existing simulation data (1) or generate new (2)? "))
                if data_choice not in [1, 2]:
                    print("Please enter 1 or 2.")
            except ValueError:
                print("Please enter a valid number.")
        
        if data_choice == 1:
            # Use existing simulation data
            sim_data, num_processes, num_resources = load_simulation_data()
        else:
            # Generate new simulations
            sim_data, num_processes, num_resources = generate_multiple_simulations()
    elif choice == 3:
        # Use existing simulation data
        sim_data, num_processes, num_resources = load_simulation_data()
    else:  # choice == 4
        # Generate new simulations
        sim_data, num_processes, num_resources = generate_multiple_simulations()
    
    if not sim_data:
        print("No simulation data available. Cannot train model.")
        return
    
    # Define state and action space
    state_size = num_processes * num_resources * 2 + num_resources + num_processes
    action_size = get_action_space_size(num_processes, num_resources)
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create agent or use existing
    if existing_agent:
        agent = existing_agent
        print("Continuing training with existing model...")
    else:
        # Create new agent
        agent = DeadlockPreventionAgent(state_size, action_size)
        print("Created new model for training...")
    
    # Extract training sequences
    print("Extracting training sequences...")
    training_data = extract_training_sequences(sim_data, num_processes, num_resources)
    
    if not training_data:
        print("No training sequences extracted.")
        return
    
    print(f"Extracted {len(training_data)} training sequences")
    
    # Get training parameters
    print("\n=== TRAINING CONFIGURATION ===")
    
    # Ask for batch size
    while True:
        try:
            batch_size = int(input("Enter batch size (8-128, default: 32): ").strip() or "32")
            if 8 <= batch_size <= 128:
                break
            print("Please enter a value between 8 and 128.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Ask for number of epochs
    while True:
        try:
            num_epochs = int(input("Enter number of training epochs (10-200, default: 50): ").strip() or "50")
            if 10 <= num_epochs <= 200:
                break
            print("Please enter a value between 10 and 200.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nTraining for {num_epochs} epochs with batch size {batch_size}...")
    
    # Training loop
    for epoch in range(num_epochs):
        # Add all experiences to memory
        for state, action, reward, next_state, done in training_data:
            # Reshape state for model input
            state_reshaped = np.reshape(state, [1, state_size])
            next_state_reshaped = np.reshape(next_state, [1, state_size])
            agent.remember(state_reshaped, action, reward, next_state_reshaped, done)
        
        # Train from memory
        agent.replay(min(batch_size, len(agent.memory)))
        
        # Update target network periodically
        if epoch % 5 == 0:
            agent.update_target_model()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} complete - epsilon: {agent.epsilon:.4f}")
    
    # Save the trained model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"model/deadlock_agent_{timestamp}"
    os.makedirs(model_path, exist_ok=True)
     
    # Save weights
    agent.save(f"{model_path}/weights.weights.h5")
    
    # Save full model
    agent.model.save(f"{model_path}/full_model.keras")
    
    # Save model parameters
    model_info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "state_size": state_size,
        "action_size": action_size,
        "num_processes": num_processes,
        "num_resources": num_resources,
        "training_sequences": len(training_data),
        "epochs": num_epochs,
        "batch_size": batch_size
    }
    
    with open(f"{model_path}/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {model_path}")

if __name__ == "__main__":
    train_rl_model()