import json
import glob
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from pathlib import Path

class MetricsAnalyzer:
    def __init__(self):
        self.sim_data = []
        self.model_data = []
    
    def load_simulation_data(self, sim_dir=None):
        """Load all simulation data"""
        if sim_dir:
            dirs = [sim_dir]
        else:
            dirs = sorted(glob.glob("simulation_results/simulation_*"))
        
        for dir_path in dirs:
            try:
                with open(f"{dir_path}/summary.json") as f:
                    summary = json.load(f)
                    summary['path'] = dir_path
                    self.sim_data.append(summary)
            except Exception as e:
                print(f"Error loading {dir_path}: {e}")
    
    def load_model_data(self, model_dir=None):
        """Load all model data"""
        if model_dir:
            dirs = [model_dir]
        else:
            dirs = sorted(glob.glob("model/deadlock_agent_*"))
        
        for dir_path in dirs:
            try:
                with open(f"{dir_path}/model_info.json") as f:
                    info = json.load(f)
                    info['path'] = dir_path
                    self.model_data.append(info)
            except Exception as e:
                print(f"Error loading {dir_path}: {e}")
    
    def analyze_simulations(self):
        """Analyze simulation performance"""
        if not self.sim_data:
            print("No simulation data loaded")
            return
        
        print("=== SIMULATION ANALYSIS ===")
        total_sims = sum(len(s['simulations']) for s in self.sim_data)
        total_deadlocks = sum(sum(1 for sim in s['simulations'] if sim['deadlocked']) for s in self.sim_data)
        
        print(f"Total simulations: {total_sims}")
        print(f"Total deadlocks: {total_deadlocks} ({total_deadlocks/total_sims*100:.1f}%)")
        
        # Per-dataset analysis
        for data in self.sim_data:
            params = data['parameters']
            stats = data.get('stats', {})
            print(f"\nDataset: {Path(data['path']).name}")
            print(f"  Config: {params['num_processes']}P/{params['num_resources']}R")
            print(f"  Deadlock rate: {stats.get('deadlock_rate', 0):.1f}%")
            print(f"  Avg steps: {stats.get('avg_steps', 0):.1f}")
            print(f"  Avg completed: {stats.get('avg_completed_processes', 0):.1f}")
    
    def analyze_models(self):
        """Analyze model training metrics"""
        if not self.model_data:
            print("No model data loaded")
            return
        
        print("\n=== MODEL ANALYSIS ===")
        for model in self.model_data:
            print(f"\nModel: {Path(model['path']).name}")
            print(f"  Dimensions: {model['num_processes']}P/{model['num_resources']}R")
            print(f"  Training sequences: {model.get('training_sequences', 'N/A')}")
            print(f"  Epochs: {model.get('epochs', 'N/A')}")
            print(f"  State/Action size: {model['state_size']}/{model['action_size']}")
    
    def compare_configurations(self):
        """Compare different process/resource configurations"""
        if not self.sim_data:
            return
        
        print("\n=== CONFIGURATION COMPARISON ===")
        configs = {}
        
        for data in self.sim_data:
            params = data['parameters']
            key = f"{params['num_processes']}P/{params['num_resources']}R"
            stats = data.get('stats', {})
            
            if key not in configs:
                configs[key] = []
            configs[key].append(stats)
        
        for config, stats_list in configs.items():
            if stats_list:
                avg_deadlock = np.mean([s.get('deadlock_rate', 0) for s in stats_list])
                avg_steps = np.mean([s.get('avg_steps', 0) for s in stats_list])
                print(f"{config}: {avg_deadlock:.1f}% deadlock, {avg_steps:.1f} avg steps")
    
    def evaluate_model_performance(self, model_path, test_scenarios=5):
        """Quick model performance evaluation"""
        try:
            from rl_model_tester import ModelDimensionAdapter
            from tensorflow.keras.models import load_model # type: ignore
            
            # Load model
            with open(f"{model_path}/model_info.json") as f:
                model_info = json.load(f)
            
            model = load_model(f"{model_path}/full_model.keras")
            
            # Test on different dimensions
            results = []
            for processes in [3, 4, 5]:
                for resources in [2, 3, 4]:
                    adapter = ModelDimensionAdapter(model, model_info, processes, resources)
                    
                    # Generate test scenarios
                    from rl_model_tester import generate_deadlock_simulations
                    scenarios = generate_deadlock_simulations(
                        test_scenarios, processes, resources, 3, render=False
                    )
                    
                    if scenarios:
                        prevented = 0
                        for scenario in scenarios:
                            from rl_model_tester import run_simulation_with_adapted_model
                            result = run_simulation_with_adapted_model(
                                adapter, scenario, render=False
                            )
                            if result['prevented_deadlock']:
                                prevented += 1
                        
                        success_rate = prevented / len(scenarios) * 100
                        results.append((f"{processes}P/{resources}R", success_rate))
            
            print(f"\n=== MODEL PERFORMANCE: {Path(model_path).name} ===")
            for config, rate in results:
                print(f"{config}: {rate:.1f}% deadlock prevention")
                
        except Exception as e:
            print(f"Error evaluating model: {e}")
    
    def generate_report(self, output_file="metrics_report.txt"):
        """Generate comprehensive report"""
        with open(output_file, 'w') as f:
            f.write("DEADLOCK PREVENTION METRICS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Simulation metrics
            if self.sim_data:
                f.write("SIMULATION METRICS\n")
                f.write("-" * 20 + "\n")
                for data in self.sim_data:
                    params = data['parameters']
                    stats = data.get('stats', {})
                    f.write(f"Dataset: {Path(data['path']).name}\n")
                    f.write(f"  Config: {params['num_processes']}P/{params['num_resources']}R\n")
                    f.write(f"  Deadlock rate: {stats.get('deadlock_rate', 0):.1f}%\n")
                    f.write(f"  Avg steps: {stats.get('avg_steps', 0):.1f}\n\n")
            
            # Model metrics
            if self.model_data:
                f.write("MODEL METRICS\n")
                f.write("-" * 15 + "\n")
                for model in self.model_data:
                    f.write(f"Model: {Path(model['path']).name}\n")
                    f.write(f"  Dimensions: {model['num_processes']}P/{model['num_resources']}R\n")
                    f.write(f"  Training sequences: {model.get('training_sequences', 'N/A')}\n\n")
        
        print(f"Report saved to {output_file}")

def main():
    analyzer = MetricsAnalyzer()
    
    print("=== DEADLOCK PREVENTION METRICS ANALYZER ===\n")
    print("1. Analyze simulations")
    print("2. Analyze models")
    print("3. Compare configurations")
    print("4. Evaluate model performance")
    print("5. Generate report")
    print("6. Full analysis")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice in ['1', '3', '5', '6']:
        analyzer.load_simulation_data()
    
    if choice in ['2', '4', '5', '6']:
        analyzer.load_model_data()
    
    if choice == '1':
        analyzer.analyze_simulations()
    elif choice == '2':
        analyzer.analyze_models()
    elif choice == '3':
        analyzer.compare_configurations()
    elif choice == '4':
        if analyzer.model_data:
            model_path = analyzer.model_data[-1]['path']  # Use latest model
            analyzer.evaluate_model_performance(model_path)
        else:
            print("No models found")
    elif choice == '5':
        analyzer.generate_report()
    elif choice == '6':
        analyzer.analyze_simulations()
        analyzer.analyze_models()
        analyzer.compare_configurations()
        analyzer.generate_report()

if __name__ == "__main__":
    main()