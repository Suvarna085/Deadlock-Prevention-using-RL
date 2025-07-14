import os
import json
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import glob
import time
import datetime
import argparse
from simulation.environment import DeadlockEnvironment

def get_latest_model():
    """Get a list of all available models"""
    model_dirs = sorted(glob.glob("model/deadlock_agent_*"))
    if not model_dirs:
        print("No trained models found")
        return None
        
    return model_dirs


def select_model():
    """Let the user select which model to use for simulation"""
    model_dirs = get_latest_model()
    
    if not model_dirs:
        print("No models available to run simulations")
        return None, None
    
    print("\n=== AVAILABLE MODELS ===")
    for i, model_dir in enumerate(model_dirs):
        # Extract timestamp from directory name
        model_name = model_dir.split("/")[-1]
        
        # Load model info
        try:
            with open(f"{model_dir}/model_info.json", "r") as f:
                model_info = json.load(f)
                model_description = f"P:{model_info['num_processes']}, R:{model_info['num_resources']}"
        except FileNotFoundError:
            model_description = "Unknown configuration"
        
        print(f"  {i+1}. {model_name} - {model_description}")
    
    # Get user selection
    while True:
        try:
            selection = int(input("\nSelect a model to use (number): ").strip())
            if 1 <= selection <= len(model_dirs):
                selected_dir = model_dirs[selection - 1]
                break
            print(f"Please enter a number between 1 and {len(model_dirs)}.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Load selected model
    try:
        with open(f"{selected_dir}/model_info.json", "r") as f:
            model_info = json.load(f)
            
        model = load_model(f"{selected_dir}/full_model.keras")
        print(f"Loaded model from {selected_dir}")
        return model, model_info
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


class ModelDimensionAdapter:
    """Adapter to test a model on environments with different dimensions"""
    
    def __init__(self, model, model_info, target_processes, target_resources):
        """
        Initialize adapter with model and target dimensions
        
        Args:
            model: The loaded TensorFlow model
            model_info: Dictionary with model information
            target_processes: Number of processes in the target environment
            target_resources: Number of resources in the target environment
        """
        self.model = model
        self.model_info = model_info
        
        # Original dimensions
        self.orig_processes = model_info['num_processes']
        self.orig_resources = model_info['num_resources']
        self.orig_state_size = model_info['state_size']
        self.orig_action_size = model_info['action_size']
        
        # Target dimensions
        self.target_processes = target_processes
        self.target_resources = target_resources
        self.target_state_size = (target_processes * target_resources * 2) + target_resources + target_processes
        self.target_action_size = target_processes * target_resources * 2  # allocate + preempt actions
        
        print(f"\n=== MODEL DIMENSION ADAPTER ===")
        print(f"Original dimensions: {self.orig_processes} processes, {self.orig_resources} resources")
        print(f"Target dimensions: {self.target_processes} processes, {self.target_resources} resources")
        print(f"Original state size: {self.orig_state_size}, action size: {self.orig_action_size}")
        print(f"Target state size: {self.target_state_size}, action size: {self.target_action_size}")
    
    def adapt_state(self, target_state_data):
        """
        Adapt a state from the target environment to match the input dimensions 
        expected by the original model
        
        Args:
            target_state_data: State data from the target environment
            
        Returns:
            A numpy array with the state adapted to the original model dimensions
        """
        # Extract components from target state
        target_allocation = np.array(target_state_data['allocation'])
        target_request = np.array(target_state_data['request'])
        target_available = np.array(target_state_data['available'])
        target_completed = np.array(target_state_data['completed'])
        
        # Create adapted versions with original dimensions
        adapted_allocation = np.zeros((self.orig_processes, self.orig_resources))
        adapted_request = np.zeros((self.orig_processes, self.orig_resources))
        adapted_available = np.zeros(self.orig_resources)
        adapted_completed = np.zeros(self.orig_processes)
        
        # Copy values up to the smaller of the dimensions
        p_limit = min(self.orig_processes, self.target_processes)
        r_limit = min(self.orig_resources, self.target_resources)
        
        # Copy allocation matrix (subset)
        adapted_allocation[:p_limit, :r_limit] = target_allocation[:p_limit, :r_limit]
        
        # Copy request matrix (subset)
        adapted_request[:p_limit, :r_limit] = target_request[:p_limit, :r_limit]
        
        # Copy available resources (subset)
        adapted_available[:r_limit] = target_available[:r_limit]
        
        # Copy completed processes (subset)
        adapted_completed[:p_limit] = target_completed[:p_limit]
        
        # Flatten and concatenate into state vector for model input
        adapted_state_vector = np.concatenate([
            adapted_allocation.flatten(),
            adapted_request.flatten(),
            adapted_available.flatten(),
            adapted_completed
        ])
        
        # Reshape for model input (batch size of 1)
        return np.reshape(adapted_state_vector, [1, self.orig_state_size])
    
    def predict_action(self, target_state_data):
        """
        Predict an action for the target environment by adapting the state,
        getting a prediction from the model, and then adapting the action
        
        Args:
            target_state_data: State data from the target environment
            
        Returns:
            process_id, resource_id, action_type tuple for the target environment
        """
        # Adapt state for model input
        adapted_state = self.adapt_state(target_state_data)
        
        # Get action prediction from the model
        action_probs = self.model.predict(adapted_state, verbose=0)[0]
        orig_action = np.argmax(action_probs)
        
        # Convert original action to process_id, resource_id, action_type
        orig_process_id, orig_resource_id, action_type = self.action_to_allocation_decision(
            orig_action, self.orig_processes, self.orig_resources
        )
        
        # Map to target dimensions (bounds check)
        target_process_id = min(orig_process_id, self.target_processes - 1)
        target_resource_id = min(orig_resource_id, self.target_resources - 1)
        
        # Convert to target action index
        if action_type == "allocate":
            target_action = target_process_id * self.target_resources + target_resource_id
        else:  # action_type == "preempt"
            target_action = (self.target_processes * self.target_resources) + (target_process_id * self.target_resources + target_resource_id)
        
        return target_process_id, target_resource_id, action_type, action_probs[orig_action]
    
    def action_to_allocation_decision(self, action, num_processes, num_resources):
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


def generate_deadlock_simulations(num_simulations, num_processes, num_resources, max_resource_instances, max_steps=50, render=True):
    """Generate simulations that result in deadlock"""
    deadlock_simulations = []
    attempts = 0
    max_attempts = num_simulations * 5  # Limit the number of attempts to find deadlocks
    
    print(f"\nGenerating {num_simulations} deadlock scenarios...")
    
    while len(deadlock_simulations) < num_simulations and attempts < max_attempts:
        attempts += 1
        
        # Create environment
        env = DeadlockEnvironment(
            num_processes=num_processes,
            num_resources=num_resources,
            max_resource_instances=max_resource_instances
        )
        
        # Save initial state for reproducibility
        initial_state = env.reset()
        seed_state = env.get_seed_state()
        
        # Run simulation
        states_history = [initial_state.copy()]
        
        # Show which deadlock scenario we're trying to generate
        current_scenario = len(deadlock_simulations) + 1
        if render:
            print(f"\nAttempting to generate deadlock scenario {current_scenario}/{num_simulations} (attempt {attempts}):")
        
        for step in range(max_steps):
            state, done = env.step()
            states_history.append(state.copy())
            
            # Always render if render is True, not just for the first attempt
            if render:
                print(f"Step {step}:")
                env.render()
                print("Deadlocked:", state['deadlocked'])
                print("Completed:", state['completed'])
                time.sleep(0.1)
            
            if done:
                # Verify this is actually a deadlock and not just completion
                if state['deadlocked'] and not all(state['completed']):
                    break
                elif all(state['completed']):
                    # This is completion, not a deadlock
                    if render:
                        print("All processes completed successfully - not a deadlock")
                    break
                else:
                    break
        
        # If we found a deadlock, save it
        if state['deadlocked'] and not all(state['completed']):
            print(f"Found deadlock scenario {len(deadlock_simulations) + 1}/{num_simulations} (attempt {attempts})")
            deadlock_sim = {
                'seed_state': seed_state,
                'history': states_history,
                'steps': state['steps'],
                'deadlocked': True
            }
            deadlock_simulations.append(deadlock_sim)
        
        # Check if we've reached the exact number requested
        if len(deadlock_simulations) >= num_simulations:
            break
    
    if len(deadlock_simulations) < num_simulations:
        print(f"Warning: Only found {len(deadlock_simulations)} deadlock scenarios after {attempts} attempts.")
    
    return deadlock_simulations[:num_simulations]  # Ensure we return exactly the number requested

def run_simulation_with_adapted_model(adapter, scenario, render=True, delay=0.5):
    """Run a simulation using the adapted RL model to make decisions"""
    # Extract parameters
    num_processes = adapter.target_processes
    num_resources = adapter.target_resources
    
    # Create environment with the same seed state
    # Use max_resource_instances from seed_state or default to 3
    max_resource_value = 3  # Default value
    
    # If resource_instances exists in seed_state, use max value from there
    if 'resource_instances' in scenario['seed_state']:
        max_resource_value = max(max(scenario['seed_state']['resource_instances']), 1)
    
    env = DeadlockEnvironment(
        num_processes=num_processes,
        num_resources=num_resources,
        max_resource_instances=max_resource_value
    )
    
    # Initialize with the same seed state
    state = env.reset_with_seed_state(scenario['seed_state'])
    
    # Store history for comparison
    rl_history = [state.copy()]
    
    # Run until done or max steps
    max_steps = 50
    original_was_deadlocked = scenario['deadlocked']
    deadlock_resolved = False
    for step in range(max_steps):
        # Get action from the adapted model
        process_id, resource_id, action_type, confidence = adapter.predict_action(state)
        
        if render:
            print(f"\nStep {step}: {action_type} resource {resource_id} "
                  f"{'to' if action_type == 'allocate' else 'from'} process {process_id}")
            print(f"Action confidence: {confidence:.4f}")
        
        # Apply the RL model's decision to the environment
        if action_type == "allocate":
            env.allocate_resource(process_id, resource_id)
        elif action_type == "preempt":
            env.preempt_resource(process_id, resource_id)
        
        # Update state after taking action
        state, done = env.step()
        rl_history.append(state.copy())
        
        if render:
            env.render()
            # Show completion and deadlock status
            print("Deadlocked:", state['deadlocked'])
            print("Completed:", state['completed'])

            if original_was_deadlocked and not state['deadlocked'] and step > 0:
                deadlock_resolved = True
                print("🔓 DEADLOCK RESOLVED!")
                
            time.sleep(delay)
        
        # Check if all processes completed (successful termination)
        all_completed = all(state['completed'])
        
        if done:
            if render:
                if all_completed:
                    print("\n✅ ALL PROCESSES COMPLETED SUCCESSFULLY!")
                elif state['deadlocked']:
                    print("\n❌ DEADLOCK DETECTED!")
                else:
                    print("\n⚠️ SIMULATION ENDED (MAX STEPS REACHED)")
            break
    
    # Return the results of this run
    all_completed = all(state['completed'])
    return {
        'original_deadlocked': scenario['deadlocked'],
        'rl_deadlocked': state['deadlocked'],
        'rl_all_completed': all_completed,
        'original_steps': scenario['steps'],
        'rl_steps': state['steps'],
        'original_history': scenario['history'],
        'rl_history': rl_history,
        'completed_processes': sum(state['completed']),
        'total_processes': len(state['completed']),
        'prevented_deadlock': scenario['deadlocked'] and not state['deadlocked'],
        'deadlock_resolved': deadlock_resolved
    }


def compare_state_differences(original_state, rl_state):
    """Compare key differences between original and RL-influenced states"""
    differences = {
        'allocation': [],
        'request': [],
        'available': [],
        'completed': []
    }
    
    # Compare allocation matrices
    for p in range(len(original_state['allocation'])):
        for r in range(len(original_state['allocation'][p])):
            orig_val = original_state['allocation'][p][r]
            rl_val = rl_state['allocation'][p][r]
            if orig_val != rl_val:
                differences['allocation'].append(f"P{p}-R{r}: {orig_val} -> {rl_val}")
    
    # Compare request matrices
    for p in range(len(original_state['request'])):
        for r in range(len(original_state['request'][p])):
            orig_val = original_state['request'][p][r]
            rl_val = rl_state['request'][p][r]
            if orig_val != rl_val:
                differences['request'].append(f"P{p}-R{r}: {orig_val} -> {rl_val}")
    
    # Compare available resources
    for r in range(len(original_state['available'])):
        orig_val = original_state['available'][r]
        rl_val = rl_state['available'][r]
        if orig_val != rl_val:
            differences['available'].append(f"R{r}: {orig_val} -> {rl_val}")
    
    # Compare completed processes
    for p in range(len(original_state['completed'])):
        orig_val = original_state['completed'][p]
        rl_val = rl_state['completed'][p]
        if orig_val != rl_val:
            differences['completed'].append(f"P{p}: {orig_val} -> {rl_val}")
    
    return differences


def print_comparison_results(results):
    """Print detailed comparison between original and RL-influenced simulations"""
    print("\n=== COMPARISON RESULTS ===")
    print(f"Original simulation ended with deadlock: {results['original_deadlocked']}")
    
    # Better description of RL simulation outcome
    if results['rl_all_completed']:
        print(f"RL-influenced simulation result: ALL PROCESSES COMPLETED SUCCESSFULLY")
    elif results['rl_deadlocked']:
        print(f"RL-influenced simulation result: DEADLOCKED")
    else:
        print(f"RL-influenced simulation result: TERMINATED (NOT DEADLOCKED)")
    
    if results['prevented_deadlock']:
        print("\n✅ RL MODEL SUCCESSFULLY PREVENTED DEADLOCK!")
    else:
        if results['original_deadlocked'] and results['rl_deadlocked']:
            print("\n❌ RL MODEL FAILED TO PREVENT DEADLOCK.")
        elif not results['original_deadlocked']:
            print("\n⚠️ ORIGINAL SCENARIO DID NOT HAVE DEADLOCK.")
    
    print(f"\nOriginal simulation steps: {results['original_steps']}")
    print(f"RL simulation steps: {results['rl_steps']}")
    print(f"Processes completed with RL: {results['completed_processes']}/{results['total_processes']}")
    
    # Compare the final states
    original_final = results['original_history'][-1]
    rl_final = results['rl_history'][-1]
    
    print("\nKey differences in final state:")
    differences = compare_state_differences(original_final, rl_final)
    
    for category, diffs in differences.items():
        if diffs:
            print(f"  {category.capitalize()}:")
            for diff in diffs:
                print(f"    {diff}")
        else:
            print(f"  {category.capitalize()}: No differences")


def display_deadlock_scenario(scenario, index, total):
    """Display a deadlock scenario in detail without skipping steps"""
    final_state = scenario['history'][-1]
    if not final_state['deadlocked'] or all(final_state['completed']):
        print(f"\n=== SCENARIO {index}/{total} IS NOT A TRUE DEADLOCK ===")
        print("Skipping to next scenario...")
        return
    
    print(f"\n=== DEADLOCK SCENARIO {index}/{total} ===")
    print(f"Total steps: {scenario['steps']}")
    print(f"Deadlocked: {scenario['deadlocked']}")
    
    # Get environment dimensions from the first state
    first_state = scenario['history'][0]
    num_processes = len(first_state['allocation'])
    num_resources = len(first_state['available'])
    
    # Create environment for visualization
    env = DeadlockEnvironment(
        num_processes=num_processes,
        num_resources=num_resources,
        max_resource_instances=max(np.max(first_state['available']), 1)
    )
    
    # Show all states in the history
    for step, state in enumerate(scenario['history']):
        print(f"\nState at step {step}:")
        env.set_state(state)
        env.render()
        
        # Show completion and deadlock status
        print("Deadlocked:", state['deadlocked'])
        print("Completed:", state['completed']) 
        
        # Add small delay for better readability
        time.sleep(0.5)
        
        # Check for deadlock or completion
        if state['deadlocked']:
            print("❌ DEADLOCK DETECTED!")
        elif all(state['completed']):
            print("✅ ALL PROCESSES COMPLETED SUCCESSFULLY!")
    
    # Don't wait for user input, just add a small pause between scenarios
    time.sleep(1.5)
    print("\nMoving to next scenario...")

def review_deadlock_scenarios(scenarios, num_to_review=None):
    """Let the user review the generated deadlock scenarios"""
    if not scenarios:
        print("No scenarios to review.")
        return
    
    # Filter out non-deadlock scenarios
    true_deadlock_scenarios = [s for s in scenarios if s['deadlocked'] and not all(s['history'][-1]['completed'])]
    
    if not true_deadlock_scenarios:
        print("No true deadlock scenarios found to review.")
        return
        
    total = len(true_deadlock_scenarios)
    if num_to_review is None or num_to_review > total:
        num_to_review = total
        
    print(f"\n=== REVIEWING {num_to_review} DEADLOCK SCENARIOS ===")
    print("These are the deadlock scenarios that will be used for testing.")
    
    for i in range(num_to_review):
        display_deadlock_scenario(true_deadlock_scenarios[i], i+1, num_to_review)

def test_cross_dimensional_model():
    """Test a model on scenarios with different process and resource counts"""
    print("\n=== CROSS-DIMENSIONAL MODEL TESTING ===")
    print("This allows testing a model on environments with different dimensions than it was trained on.")
    
    # Select and load model
    model, model_info = select_model()
    if not model or not model_info:
        print("No model selected. Exiting.")
        return
    
    # Get target dimensions from user
    print("\nTarget Environment Configuration:")
    
    # Get number of processes
    while True:
        try:
            target_processes = int(input(f"Enter number of processes (trained on {model_info['num_processes']}): ").strip())
            if target_processes > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get number of resources
    while True:
        try:
            target_resources = int(input(f"Enter number of resources (trained on {model_info['num_resources']}): ").strip())
            if target_resources > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Create adapter
    adapter = ModelDimensionAdapter(model, model_info, target_processes, target_resources)
    
    # Get number of scenarios to test
    while True:
        try:
            num_scenarios = int(input("Enter number of deadlock scenarios to generate (1-5): ").strip())
            if 1 <= num_scenarios <= 5:
                break
            print("Please enter a value between 1 and 5.")
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
    
    # Get render option
    render = input("Render simulation steps? (y/n, default: y): ").strip().lower() != 'n'
    
    # Generate deadlock scenarios with target dimensions
    scenarios = generate_deadlock_simulations(
        num_scenarios,
        target_processes,
        target_resources,
        max_resource_instances,
        render=render
    )
    
    if not scenarios:
        print("Failed to generate deadlock scenarios. Exiting.")
        return
    
    # Allow user to review scenarios
    review_option = input("\nWould you like to review the deadlock scenarios before testing? (y/n, default: y): ").strip().lower() != 'n'
    if review_option:
        review_deadlock_scenarios(scenarios, num_scenarios)
    
    # Run each scenario with the adapted model
    prevented_count = 0
    success_count = 0
    total_scenarios = len(scenarios)

    print("---------------------------------------------------------------------------------------------------------------------------")
    
    for i in range(total_scenarios):
        print(f"\n\n=== TESTING SCENARIO {i+1}/{total_scenarios} ===")
        
        # Validate that the original scenario actually has a deadlock
        original_final_state = scenarios[i]['history'][-1]
        if not original_final_state['deadlocked']:
            print("\n⚠️ WARNING: This scenario does not end in deadlock! Skipping...")
            continue
            
        # Display original scenario ending
        env = DeadlockEnvironment(
            num_processes=target_processes,
            num_resources=target_resources,
            max_resource_instances=max_resource_instances
        )
        
        # Show original deadlocked state
        print("\nOriginal scenario final state (deadlocked):")
        env.set_state(original_final_state)
        env.render()
        print("Deadlocked:", original_final_state['deadlocked'])
        print("Completed:", original_final_state['completed'])
        time.sleep(1)
        
        # Run with adapted model
        print("\nRunning scenario with adapted RL model:")
        results = run_simulation_with_adapted_model(
            adapter, 
            scenarios[i], 
            render=render,
            delay=0.3
        )
        
        # Print comparison
        print_comparison_results(results)
        
        if results['prevented_deadlock']:
            prevented_count += 1
            if results['rl_all_completed']:
                success_count += 1
    
    # Print summary
    print("\n\n=== TEST SUMMARY ===")
    print(f"Original model dimensions: {model_info['num_processes']} processes, {model_info['num_resources']} resources")
    print(f"Target environment dimensions: {target_processes} processes, {target_resources} resources")
    print(f"Total deadlock scenarios tested: {total_scenarios}")
    print(f"Deadlocks prevented by adapted model: {prevented_count}/{total_scenarios} "
          f"({prevented_count/total_scenarios*100:.1f}%)")
    print(f"Scenarios where all processes completed successfully: {success_count}/{total_scenarios} "
          f"({success_count/total_scenarios*100:.1f}%)")
    
    # Compare with original model's expected performance
    print("\nThis result should be compared with the model's performance on its original dimensions.")
    print("Differences in performance may indicate how well the model generalizes to new environments.")


if __name__ == "__main__":
    test_cross_dimensional_model()