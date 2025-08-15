#!/usr/bin/env python3
"""
Ecosystem Simulation Training and Analysis

Train and analyze the ecosystem simulation with different parameters
to find stable, diverse ecosystems.
"""

import json
from ecosystem_simulation import EcosystemSimulation


def train_ecosystem(num_runs: int = 5, steps_per_run: int = 200):
    """Train multiple ecosystem instances and analyze results."""
    print(f"Training {num_runs} ecosystem instances for {steps_per_run} steps each...")
    print("=" * 60)
    
    results = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        print("-" * 30)
        
        sim = EcosystemSimulation()
        result = sim.run_simulation(steps_per_run)
        
        # Analyze the result
        analysis = analyze_ecosystem_run(result, sim)
        results.append({
            'run': run + 1,
            'result': result,
            'analysis': analysis
        })
        
        print(f"Final populations: {result['final_populations']}")
        print(f"Stability score: {analysis['stability_score']:.2f}")
        print(f"Diversity score: {analysis['diversity_score']:.2f}")
    
    # Aggregate analysis
    print("\n" + "=" * 60)
    print("AGGREGATE ANALYSIS")
    print("=" * 60)
    
    avg_stability = sum(r['analysis']['stability_score'] for r in results) / len(results)
    avg_diversity = sum(r['analysis']['diversity_score'] for r in results) / len(results)
    
    print(f"Average stability score: {avg_stability:.2f}")
    print(f"Average diversity score: {avg_diversity:.2f}")
    
    # Find best run
    best_run = max(results, key=lambda r: r['analysis']['overall_score'])
    print(f"\nBest run: #{best_run['run']}")
    print(f"  Overall score: {best_run['analysis']['overall_score']:.2f}")
    print(f"  Final populations: {best_run['result']['final_populations']}")
    
    # Save results
    with open('ecosystem_training_results.json', 'w') as f:
        json.dump({
            'summary': {
                'num_runs': num_runs,
                'steps_per_run': steps_per_run,
                'avg_stability': avg_stability,
                'avg_diversity': avg_diversity,
                'best_run': best_run['run'],
                'best_score': best_run['analysis']['overall_score']
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to ecosystem_training_results.json")
    
    return results


def analyze_ecosystem_run(result, simulation):
    """Analyze a single ecosystem run and calculate scores."""
    final_pops = result['final_populations']
    
    # Stability score - how many organisms survived
    total_final = sum(final_pops.values())
    stability_score = min(total_final / 20.0, 1.0)  # Normalize to 0-1
    
    # Diversity score - how many different species survived
    surviving_species = len([count for count in final_pops.values() if count > 0])
    max_species = len(final_pops)
    diversity_score = surviving_species / max_species
    
    # Balance score - check if predator-prey relationships are maintained
    has_plants = final_pops.get('grass', 0) + final_pops.get('trees', 0) > 0
    has_herbivores = final_pops.get('rabbits', 0) + final_pops.get('deer', 0) > 0  
    has_carnivores = final_pops.get('wolves', 0) + final_pops.get('hawks', 0) > 0
    has_omnivores = final_pops.get('bears', 0) > 0
    
    trophic_levels = sum([has_plants, has_herbivores, has_carnivores, has_omnivores])
    balance_score = trophic_levels / 4.0
    
    # Overall score
    overall_score = (stability_score * 0.4 + diversity_score * 0.3 + balance_score * 0.3)
    
    return {
        'stability_score': stability_score,
        'diversity_score': diversity_score,
        'balance_score': balance_score,
        'overall_score': overall_score,
        'surviving_species': surviving_species,
        'total_organisms': total_final,
        'trophic_levels': trophic_levels
    }


def demonstrate_stable_ecosystem():
    """Demonstrate a well-balanced ecosystem run."""
    print("=" * 60)
    print("STABLE ECOSYSTEM DEMONSTRATION")
    print("=" * 60)
    
    sim = EcosystemSimulation()
    
    print("\nInitial state:")
    for name, pop in sim.populations.items():
        print(f"  {name}: {pop.get_count()}")
    
    print(f"\nRunning ecosystem simulation...")
    
    # Run and show periodic updates
    for step in range(300):
        result = sim.step()
        
        if step % 50 == 0:
            print(f"\nStep {step}:")
            for species, count in result['populations'].items():
                if count > 0:
                    print(f"  {species}: {count}")
            
            # Show brief visualization every 100 steps
            if step % 100 == 0:
                print("Ecosystem state:")
                sim.visualize_ecosystem()
        
        # Check for early termination
        if result['total_organisms'] == 0:
            print(f"\nEcosystem collapsed at step {step}")
            break
    
    print(f"\n" + "=" * 40)
    print("FINAL ECOSYSTEM STATE")
    print("=" * 40)
    sim.visualize_ecosystem()
    
    final_stats = sim.get_statistics()
    print(f"Total organisms: {final_stats['ecosystem']['total_organisms']}")
    print(f"Species diversity: {final_stats['ecosystem']['species_diversity']}")
    
    return sim


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Training mode
        train_ecosystem()
    else:
        # Demonstration mode
        demonstrate_stable_ecosystem()
        
        print("\n" + "=" * 60)
        print("ECOSYSTEM SIMULATION FEATURES DEMONSTRATED:")
        print("- Multiple species with different roles (plants, herbivores, carnivores, omnivores)")
        print("- Predator-prey relationships and food chains")
        print("- Environmental niches (grassland, forest, water, rocky)")
        print("- Population dynamics (birth, death, reproduction)")
        print("- Energy-based survival system")
        print("- Spatial movement and interactions")
        print("- Resource competition and carrying capacity")
        print("=" * 60)
        
        print("\nTo run training analysis: python train_ecosystem.py train")