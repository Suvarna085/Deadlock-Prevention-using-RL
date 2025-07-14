from simulation.environment import DeadlockEnvironment
import time
import os
import json
import datetime
import sys

def run_simulation(num_processes, num_resources, max_resource_instances, steps=50, render=True):
    # Create environment with user-specified parameters
    env = DeadlockEnvironment(
        num_processes=num_processes,
        num_resources=num_resources,
        max_resource_instances=max_resource_instances
    )
    
    # Reset and get initial state
    state = env.reset()
    
    # Store history of states for analysis
    history = [state]
    
    # Run simulation for specified steps or until done
    for i in range(steps):
        state, done = env.step()
        history.append(state)
        
        if render:
            env.render()
            time.sleep(0.5)  # Slow down for readability
        
        if done:
            break
    
    # Return detailed results
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

def save_simulation_results(results, sim_folder, sim_id):
    # Create simulation directory structure if it doesn't exist
    os.makedirs(sim_folder, exist_ok=True)
    
    # Create a filename for this simulation
    filename = f"{sim_folder}/{sim_id}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    for state in results['history']:
        state['allocation'] = state['allocation'].tolist()
        state['request'] = state['request'].tolist()
        state['available'] = state['available'].tolist()
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filename

def get_user_input():
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
    render = input("Render simulation steps? (y/n, default: y): ").strip().lower() != 'n'
    
    return {
        'num_processes': num_processes,
        'num_resources': num_resources,
        'max_resource_instances': max_resource_instances,
        'num_simulations': num_simulations,
        'max_steps': max_steps,
        'render': render
    }

if __name__ == "__main__":
    # Get user input for simulation parameters
    params = get_user_input()
    
    # Create base directory for all simulation results
    base_dir = "simulation_results"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a timestamped directory for this set of simulations
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_dir = f"{base_dir}/simulation_{timestamp}"
    os.makedirs(sim_dir, exist_ok=True)
    
    # Run simulations and count deadlocks
    num_simulations = params['num_simulations']
    deadlock_count = 0
    total_steps = 0
    completed_processes = 0
    
    # Save summary data
    summary = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': {
            'num_processes': params['num_processes'],
            'num_resources': params['num_resources'],
            'max_resource_instances': params['max_resource_instances'],
            'max_steps': params['max_steps']
        },
        'total_simulations': num_simulations,
        'simulations': []
    }
    
    try:
        for i in range(num_simulations):
            print(f"\n\n=== SIMULATION {i+1}/{num_simulations} ===")
            
            # Run simulation with user parameters
            results = run_simulation(
                params['num_processes'],
                params['num_resources'],
                params['max_resource_instances'],
                steps=params['max_steps'],
                render=params['render']
            )
            
            # Save detailed results for this simulation
            filename = save_simulation_results(results, sim_dir, i+1)
            print(f"Results saved to {filename}")
            
            # Update counters
            deadlocked = results['deadlocked']
            steps = results['steps']
            completed = results['completed_processes']
            
            deadlock_count += int(deadlocked)
            total_steps += steps
            completed_processes += completed
            
            # Add to summary
            summary['simulations'].append({
                'id': i+1,
                'deadlocked': deadlocked,
                'steps': steps,
                'completed_processes': completed,
                'file': f"{i+1}.json"
            })
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    
    # Calculate final stats
    num_completed = len(summary['simulations'])
    if num_completed > 0:
        summary['stats'] = {
            'deadlock_rate': deadlock_count / num_completed * 100,
            'avg_steps': total_steps / num_completed,
            'avg_completed_processes': completed_processes / num_completed
        }
    
    # Save summary
    with open(f"{sim_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n\n=== SUMMARY ===")
    print(f"Simulations run: {num_completed}")
    if num_completed > 0:
        print(f"Deadlocks occurred: {deadlock_count} ({deadlock_count/num_completed*100:.1f}%)")
        print(f"Average steps per simulation: {total_steps/num_completed:.1f}")
        print(f"Average completed processes: {completed_processes/num_completed:.1f}")
    print(f"Results saved to {sim_dir}/summary.json")