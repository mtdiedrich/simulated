#!/usr/bin/env python3
"""
Demo script for the ecosystem simulation.
"""

from ecosystem_simulation import EcosystemSimulation

def main():
    print("=== Ecosystem Simulation Demo ===")
    
    # Create smaller ecosystem for faster demo
    simulation = EcosystemSimulation(150, 150)
    
    print("\nInitial ecosystem state:")
    simulation.visualize_ecosystem()
    
    # Run simulation for shorter duration
    print("\nRunning simulation for 100 steps...")
    result = simulation.run_simulation(100)
    
    print("\n=== Final Results ===")
    print(f"Simulation completed after {result['final_step']} steps")
    print(f"Ecosystem stable: {result['ecosystem_stable']}")
    print("\nFinal populations:")
    for species, count in result['final_populations'].items():
        print(f"  {species}: {count}")
    
    print("\nFinal ecosystem state:")
    simulation.visualize_ecosystem()
    
    print("\n=== Ecosystem Statistics ===")
    stats = simulation.get_statistics()
    print(f"Total organisms: {stats['ecosystem']['total_organisms']}")
    print(f"Species diversity: {stats['ecosystem']['species_diversity']}")
    print(f"Total energy: {stats['ecosystem']['total_energy']:.1f}")

if __name__ == "__main__":
    main()