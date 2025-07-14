# simulation/environment.py
import random
import numpy as np
import os
import json
from datetime import datetime
from .process import Process
from .resource import Resource

class DeadlockEnvironment:
    def __init__(self, num_processes, num_resources, max_resource_instances, deadlock_prone=False):
        self.num_processes = num_processes
        self.num_resources = num_resources
        self.max_resource_instances = max_resource_instances
        self.deadlock_prone = deadlock_prone
        
        # Create processes and resources
        self.processes = [Process(i) for i in range(num_processes)]
        
        if deadlock_prone:
            # Fewer resource instances for deadlock-prone environments
            self.resources = [Resource(i, random.randint(1, max(1, max_resource_instances // 2))) 
                         for i in range(num_resources)]
        else:
            self.resources = [Resource(i, random.randint(1, max_resource_instances)) 
                         for i in range(num_resources)]
    
        
        # Allocation matrix: processes × resources
        self.allocation_matrix = np.zeros((num_processes, num_resources), dtype=int)
        
        # Request matrix: processes × resources
        self.request_matrix = np.zeros((num_processes, num_resources), dtype=int)
        
        # Available resources vector
        self.available = np.array([r.instances for r in self.resources])
        
        # Keep track of completed processes
        self.completed = [False] * num_processes
        
        # Steps since last allocation (for deadlock detection)
        self.steps_since_allocation = 0
        self.deadlocked = False
        self.steps = 0
    
    def reset(self):
        """Reset the environment to initial state with new random resource requirements"""
        # Reset resources
        if self.deadlock_prone:
            for r in self.resources:
                r.instances = random.randint(1, max(1, self.max_resource_instances // 2))
        else:
            for r in self.resources:
                r.instances = random.randint(1, self.max_resource_instances)
        
        # Reset matrices
        self.allocation_matrix = np.zeros((self.num_processes, self.num_resources), dtype=int)
        self.request_matrix = np.zeros((self.num_processes, self.num_resources), dtype=int)
        
        # Reset available resources
        self.available = np.array([r.instances for r in self.resources])
        
        # Generate random resource needs for processes
        for p in self.processes:
            p.reset()
            
            # Assign random resource needs
            if self.deadlock_prone:
                # More aggressive resource needs - processes demand more resources
                # Make processes require most or all available resources of each type
                p.total_needs = [random.randint(1, self.resources[r].instances) 
                            for r in range(self.num_resources)]
            else:
                p.total_needs = [random.randint(0, min(2, self.resources[r].instances)) 
                            for r in range(self.num_resources)]
        
        self.completed = [False] * self.num_processes
        self.steps_since_allocation = 0
        self.deadlocked = False
        self.steps = 0
        
        return self._get_state()
    
    def step(self):
        """Simulate one step in the environment"""
        self.steps += 1
        allocation_happened = False
        
        # First, check if any process can complete and release resources
        for p_id, process in enumerate(self.processes):
            if self.completed[p_id]:
                continue
                
            # Check if process has all needed resources
            has_all_needs = True
            for r_id in range(self.num_resources):
                if self.allocation_matrix[p_id][r_id] < process.total_needs[r_id]:
                    has_all_needs = False
                    break
            
            # If process has all resources, complete it and release resources
            if has_all_needs:
                self.completed[p_id] = True
                for r_id in range(self.num_resources):
                    self.available[r_id] += self.allocation_matrix[p_id][r_id]
                    self.allocation_matrix[p_id][r_id] = 0
                allocation_happened = True

        # Set request probability higher for deadlock-prone mode
        request_probability = 0.8 if self.deadlock_prone else 0.3
        
        # Generate new resource requests for active processes
        for p_id, process in enumerate(self.processes):
            if self.completed[p_id]:
                continue
                
            # Generate requests for resources still needed
            for r_id in range(self.num_resources):
                needed = process.total_needs[r_id] - self.allocation_matrix[p_id][r_id]
                if needed > 0 and self.request_matrix[p_id][r_id] == 0:
                    # Request with some probability
                    if random.random() < request_probability:
                        if self.deadlock_prone:
                            # Request all needed resources at once - more likely to cause deadlock
                            request_amount = needed
                        else:
                            # More conservative requests
                            request_amount = min(needed, random.randint(1, needed))
                        self.request_matrix[p_id][r_id] = request_amount
        
        # Try to allocate requested resources
        # In deadlock_prone mode, we use an allocation strategy more likely to cause deadlocks
        process_order = list(range(self.num_processes))
        if self.deadlock_prone:
            # Randomize the order to increase chances of partial allocations
            random.shuffle(process_order)
        
        for p_id in process_order:
            if self.completed[p_id]:
                continue
                
            all_requests_granted = True
            
            for r_id in range(self.num_resources):
                requested = self.request_matrix[p_id][r_id]
                if requested > 0:
                    # Check if enough resources available
                    if self.available[r_id] >= requested:
                        # Allocate resources
                        self.allocation_matrix[p_id][r_id] += requested
                        self.available[r_id] -= requested
                        self.request_matrix[p_id][r_id] = 0
                        allocation_happened = True
                    else:
                        all_requests_granted = False
        
        # Check for potential deadlock (no allocations for several steps)
        if allocation_happened:
            self.steps_since_allocation = 0
        else:
            self.steps_since_allocation += 1
            
        # If no allocations for some time and processes still active, likely deadlocked
        if self.steps_since_allocation > 3 and not all(self.completed) and not allocation_happened:
            if self._check_circular_wait():
                self.deadlocked = True
        
        done = self.deadlocked or all(self.completed)
        
        return self._get_state(), done
    
    def _check_circular_wait(self):
        """Improved circular wait detection for deadlock"""
        # Create wait-for graph adjacency matrix
        n = self.num_processes
        wait_for = np.zeros((n, n), dtype=int)
        
        # For each active process, check which other processes hold resources it's waiting for
        for p1 in range(n):
            if self.completed[p1]:
                continue
                
            for r_id in range(self.num_resources):
                # If p1 is requesting this resource
                if self.request_matrix[p1][r_id] > 0:
                    # Check which processes hold this resource
                    for p2 in range(n):
                        if p1 != p2 and not self.completed[p2] and self.allocation_matrix[p2][r_id] > 0:
                            # p1 is waiting for p2
                            wait_for[p1][p2] = 1
        
        # Check for cycles in wait-for graph using DFS
        def has_cycle(node, visited, rec_stack):
            visited[node] = True
            rec_stack[node] = True
            
            for neighbor in range(n):
                if wait_for[node][neighbor] == 1:
                    if not visited[neighbor]:
                        if has_cycle(neighbor, visited, rec_stack):
                            return True
                    elif rec_stack[neighbor]:
                        return True
            
            rec_stack[node] = False
            return False
        
        # Start DFS from each unvisited node
        visited = [False] * n
        rec_stack = [False] * n
        
        for node in range(n):
            if not self.completed[node] and not visited[node]:
                if has_cycle(node, visited, rec_stack):
                    return True
        
        return False
        
    def _get_state(self):
        """Return current state of the environment"""
        return {
            'allocation': self.allocation_matrix.copy(),
            'request': self.request_matrix.copy(),
            'available': self.available.copy(),
            'completed': self.completed.copy(),
            'deadlocked': self.deadlocked,
            'steps': self.steps
        }
    
    def render(self):
        """Print current state of the environment"""
        print(f"\n=== Step {self.steps} ===")
        print("Available resources:", self.available)
        print("\nAllocation Matrix:")
        print(self.allocation_matrix)
        print("\nRequest Matrix:")
        print(self.request_matrix)
        print("\nCompleted processes:", [i for i, c in enumerate(self.completed) if c])
        if self.deadlocked:
            print("\n⚠️ DEADLOCK DETECTED!")
        elif all(self.completed):
            print("\n✅ All processes completed successfully!")
    
    def save_state(self, folder_path, simulation_id, step):
        """Save current state to a JSON file"""
        # Create a serializable state dictionary
        state = self._get_state()
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_state = {
            'allocation': state['allocation'].tolist(),
            'request': state['request'].tolist(),
            'available': state['available'].tolist(),
            'completed': state['completed'],
            'deadlocked': state['deadlocked'],
            'steps': state['steps'],
            'total_needs': [p.total_needs for p in self.processes]
        }
        
        # Create the file path
        file_path = os.path.join(folder_path, f"sim_{simulation_id}_step_{step}.json")
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(serializable_state, f, indent=2)
        
        return file_path
    
    def get_seed_state(self):
        """Return the seed state needed to recreate this simulation"""
        seed_state = {
            'num_processes': self.num_processes,
            'num_resources': self.num_resources,
            'max_resource_instances': self.max_resource_instances,
            'resource_instances': [r.instances for r in self.resources],
            'process_needs': [p.total_needs.copy() if hasattr(p, 'total_needs') else [] for p in self.processes]
        }
        return seed_state

    def reset_with_seed_state(self, seed_state):
        """Reset the environment using a specific seed state"""
        # Reset basic parameters if they match
        if (seed_state['num_processes'] == self.num_processes and
                seed_state['num_resources'] == self.num_resources):
            
            # Set resource instances
            for r_id, instances in enumerate(seed_state['resource_instances']):
                if r_id < len(self.resources):
                    self.resources[r_id].instances = instances
            
            # Reset matrices
            self.allocation_matrix = np.zeros((self.num_processes, self.num_resources), dtype=int)
            self.request_matrix = np.zeros((self.num_processes, self.num_resources), dtype=int)
            
            # Reset available resources
            self.available = np.array([r.instances for r in self.resources])
            
            # Set process needs
            for p_id, needs in enumerate(seed_state['process_needs']):
                if p_id < len(self.processes):
                    self.processes[p_id].reset()
                    self.processes[p_id].total_needs = needs.copy() if isinstance(needs, list) else needs
            
            self.completed = [False] * self.num_processes
            self.steps_since_allocation = 0
            self.deadlocked = False
            self.steps = 0
            
            return self._get_state()
        else:
            print("Warning: Seed state parameters don't match environment parameters")
            return self.reset()

    def set_state(self, state):
        """Set the environment to a specific state"""
        # Ensure state has all required components
        required_keys = ['allocation', 'request', 'available', 'completed', 'deadlocked', 'steps']
        if not all(key in state for key in required_keys):
            print("Error: Invalid state format")
            return False
        
        # Convert lists to numpy arrays if needed
        if not isinstance(state['allocation'], np.ndarray):
            state['allocation'] = np.array(state['allocation'])
        
        if not isinstance(state['request'], np.ndarray):
            state['request'] = np.array(state['request'])
        
        if not isinstance(state['available'], np.ndarray):
            state['available'] = np.array(state['available'])
        
        # Set the state components
        self.allocation_matrix = state['allocation'].copy()
        self.request_matrix = state['request'].copy()
        self.available = state['available'].copy()
        self.completed = state['completed'].copy()
        self.deadlocked = state['deadlocked']
        self.steps = state['steps']
        
        # Calculate steps since allocation (for deadlock detection)
        self.steps_since_allocation = 5 if state['deadlocked'] else 0
        
        return True

    def allocate_resource(self, process_id, resource_id):
        """Explicitly allocate a resource to a process"""
        # Check if the process and resource IDs are valid
        if (0 <= process_id < self.num_processes and 
                0 <= resource_id < self.num_resources):
            
            # Check if the process isn't completed
            if not self.completed[process_id]:
                # Check if there's a pending request
                request_amount = self.request_matrix[process_id][resource_id]
                
                # If no explicit request, create an implicit one based on process needs
                if request_amount == 0:
                    current = self.allocation_matrix[process_id][resource_id]
                    needed = self.processes[process_id].total_needs[resource_id] - current
                    if needed > 0:
                        request_amount = 1  # Default to requesting 1 resource
                
                # Check if there are available resources and the process needs them
                if (request_amount > 0 and 
                        self.available[resource_id] >= 1):
                    
                    # Allocate one resource unit
                    self.allocation_matrix[process_id][resource_id] += 1
                    self.available[resource_id] -= 1
                    
                    # Update request matrix if it had a pending request
                    if self.request_matrix[process_id][resource_id] > 0:
                        self.request_matrix[process_id][resource_id] -= 1
                    
                    return True
        
        return False

    def preempt_resource(self, process_id, resource_id):
        """Explicitly preempt a resource from a process"""
        # Check if the process and resource IDs are valid
        if (0 <= process_id < self.num_processes and 
                0 <= resource_id < self.num_resources):
            
            # Check if the process has this resource allocated
            if self.allocation_matrix[process_id][resource_id] > 0:
                # Remove one resource unit
                self.allocation_matrix[process_id][resource_id] -= 1
                self.available[resource_id] += 1
                return True
        
        return False

    def force_deadlock_scenario(self):
        """Create a scenario that's almost guaranteed to deadlock"""
        # Reset the environment first
        self.reset()
        
        # Create a situation with limited resources
        for r in self.resources:
            r.instances = max(1, min(2, self.num_processes // 2))
        
        # Reset available resources
        self.available = np.array([r.instances for r in self.resources])
        
        # Make all processes need ALL types of resources
        for p in self.processes:
            p.total_needs = [max(1, r.instances) for r in self.resources]
        
        # Pre-allocate some resources to create a circular wait condition
        for p_id in range(self.num_processes):
            r_id = p_id % self.num_resources
            if self.available[r_id] > 0:
                self.allocation_matrix[p_id][r_id] = 1
                self.available[r_id] -= 1
            
            # Have each process request the next resource in the cycle
            next_r_id = (r_id + 1) % self.num_resources
            self.request_matrix[p_id][next_r_id] = 1
        
        return self._get_state()