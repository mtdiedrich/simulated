"""
Tests for the Ecosystem Simulation

Comprehensive tests to verify the ecosystem components work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ecosystem_simulation import (
    Vector2D, SpeciesType, Niche, EcosystemEnvironment, 
    Organism, Population, EcosystemSimulation
)


def test_ecosystem_environment():
    """Test EcosystemEnvironment functionality."""
    print("Testing EcosystemEnvironment...")
    
    env = EcosystemEnvironment(100, 100)
    
    # Test niche detection
    grassland_pos = Vector2D(25, 25)
    forest_pos = Vector2D(75, 25)
    water_pos = Vector2D(25, 75)
    rocky_pos = Vector2D(75, 75)
    
    assert env.get_niche_at_position(grassland_pos) == Niche.GRASSLAND
    assert env.get_niche_at_position(forest_pos) == Niche.FOREST
    assert env.get_niche_at_position(water_pos) == Niche.WATER
    assert env.get_niche_at_position(rocky_pos) == Niche.ROCKY
    
    # Test boundary keeping
    out_of_bounds = Vector2D(-10, 150)
    bounded = env.keep_in_bounds(out_of_bounds)
    assert 0 <= bounded.x < env.width
    assert 0 <= bounded.y < env.height
    
    # Test resource quality
    quality = env.get_resource_quality(grassland_pos, 'plant_growth_rate')
    assert 0 <= quality <= 1.0
    
    print("✓ EcosystemEnvironment tests passed")


def test_organism():
    """Test Organism behavior."""
    print("Testing Organism...")
    
    env = EcosystemEnvironment()
    
    # Test different species types
    plant = Organism('grass', SpeciesType.PLANT, Vector2D(50, 50))
    herbivore = Organism('rabbit', SpeciesType.HERBIVORE, Vector2D(60, 60))
    carnivore = Organism('wolf', SpeciesType.CARNIVORE, Vector2D(70, 70))
    
    assert plant.species_type == SpeciesType.PLANT
    assert herbivore.species_type == SpeciesType.HERBIVORE
    assert carnivore.species_type == SpeciesType.CARNIVORE
    
    # Test diet relationships
    assert herbivore.can_eat(plant)
    assert carnivore.can_eat(herbivore)
    assert not plant.can_eat(herbivore)
    assert not herbivore.can_eat(carnivore)
    
    # Test initial state
    assert plant.alive
    assert plant.energy > 0
    assert plant.age == 0
    
    print("✓ Organism tests passed")


def test_population():
    """Test Population management."""
    print("Testing Population...")
    
    env = EcosystemEnvironment()
    pop = Population('test_herbivores', SpeciesType.HERBIVORE, 10, env)
    
    # Test initial population
    assert pop.get_count() == 10
    assert pop.species_type == SpeciesType.HERBIVORE
    
    # Test statistics
    stats = pop.get_statistics()
    assert stats['count'] == 10
    assert stats['avg_age'] == 0  # All start at age 0
    assert stats['avg_energy'] > 0
    
    # Kill some organisms to test cleanup
    for i in range(5):
        pop.organisms[i].alive = False
    
    # Update should remove dead organisms
    pop.update(pop.organisms, env)
    assert pop.get_count() == 5
    
    print("✓ Population tests passed")


def test_ecosystem_simulation():
    """Test full ecosystem simulation."""
    print("Testing EcosystemSimulation...")
    
    sim = EcosystemSimulation(100, 100)
    
    # Test initial state
    assert len(sim.populations) > 0
    initial_total = sum(pop.get_count() for pop in sim.populations.values())
    assert initial_total > 0
    
    # Test single step
    result = sim.step()
    assert 'step' in result
    assert 'total_organisms' in result
    assert 'populations' in result
    assert result['step'] == 1
    
    # Test statistics
    stats = sim.get_statistics()
    assert 'ecosystem' in stats
    assert stats['ecosystem']['total_organisms'] > 0
    
    print("✓ EcosystemSimulation tests passed")


def test_predator_prey_relationships():
    """Test that predator-prey relationships work correctly."""
    print("Testing predator-prey relationships...")
    
    env = EcosystemEnvironment()
    
    # Create a simple food chain
    plant = Organism('grass', SpeciesType.PLANT, Vector2D(50, 50))
    rabbit = Organism('rabbit', SpeciesType.HERBIVORE, Vector2D(52, 52))
    wolf = Organism('wolf', SpeciesType.CARNIVORE, Vector2D(54, 54))
    
    organisms = [plant, rabbit, wolf]
    
    # Test food finding
    rabbit_food = rabbit.find_food(organisms, env)
    wolf_food = wolf.find_food(organisms, env)
    
    assert rabbit_food == plant  # Rabbit should find plant
    assert wolf_food == rabbit   # Wolf should find rabbit
    
    # Test that plants don't find food
    plant_food = plant.find_food(organisms, env)
    assert plant_food is None
    
    print("✓ Predator-prey relationship tests passed")


def test_ecosystem_balance():
    """Test ecosystem balance over multiple steps."""
    print("Testing ecosystem balance...")
    
    sim = EcosystemSimulation(100, 100)
    
    # Run for a few steps and check stability
    initial_counts = {name: pop.get_count() for name, pop in sim.populations.items()}
    
    for _ in range(10):
        sim.step()
    
    # Check that ecosystem hasn't completely collapsed or exploded
    final_counts = {name: pop.get_count() for name, pop in sim.populations.items()}
    total_initial = sum(initial_counts.values())
    total_final = sum(final_counts.values())
    
    # Should not have zero organisms or excessive growth
    assert total_final > 0, "Ecosystem collapsed too quickly"
    assert total_final < total_initial * 10, "Population explosion detected"
    
    # Should have some species diversity
    living_species = len([count for count in final_counts.values() if count > 0])
    assert living_species >= 2, "Too much species extinction"
    
    print("✓ Ecosystem balance tests passed")


def run_all_tests():
    """Run all ecosystem tests."""
    print("Running Ecosystem Simulation Tests")
    print("=" * 50)
    
    try:
        test_ecosystem_environment()
        test_organism()
        test_population()
        test_ecosystem_simulation()
        test_predator_prey_relationships()
        test_ecosystem_balance()
        
        print("=" * 50)
        print("✓ All ecosystem tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)