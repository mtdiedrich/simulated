"""
Ecosystem Simulation Environment

A comprehensive ecosystem simulation with multiple species, predator-prey relationships,
niches, and population dynamics. Built upon the Vector2D foundation from the soccer simulation.
"""

import math
import random
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum

# Import Vector2D from the existing soccer simulation
try:
    from soccer_simulation import Vector2D
except ImportError:
    # Fallback implementation if soccer_simulation is not available
    class Vector2D:
        def __init__(self, x: float = 0.0, y: float = 0.0):
            self.x = x
            self.y = y
        
        def __add__(self, other: 'Vector2D') -> 'Vector2D':
            return Vector2D(self.x + other.x, self.y + other.y)
        
        def __sub__(self, other: 'Vector2D') -> 'Vector2D':
            return Vector2D(self.x - other.x, self.y - other.y)
        
        def __mul__(self, scalar: float) -> 'Vector2D':
            return Vector2D(self.x * scalar, self.y * scalar)
        
        def magnitude(self) -> float:
            return math.sqrt(self.x ** 2 + self.y ** 2)
        
        def normalize(self) -> 'Vector2D':
            mag = self.magnitude()
            if mag == 0:
                return Vector2D(0, 0)
            return Vector2D(self.x / mag, self.y / mag)
        
        def distance_to(self, other: 'Vector2D') -> float:
            return (self - other).magnitude()
        
        def copy(self) -> 'Vector2D':
            return Vector2D(self.x, self.y)


class SpeciesType(Enum):
    """Types of species in the ecosystem."""
    PLANT = "plant"
    HERBIVORE = "herbivore"  
    CARNIVORE = "carnivore"
    OMNIVORE = "omnivore"


class Niche(Enum):
    """Environmental niches/habitats."""
    GRASSLAND = "grassland"
    FOREST = "forest" 
    WATER = "water"
    ROCKY = "rocky"


class EcosystemEnvironment:
    """Represents the ecosystem environment with multiple niches."""
    
    def __init__(self, width: float = 200.0, height: float = 200.0):
        self.width = width
        self.height = height
        
        # Define niche regions (simplified as rectangular areas)
        self.niches = {
            Niche.GRASSLAND: {'bounds': (0, 0, width//2, height//2), 'carrying_capacity': 50},
            Niche.FOREST: {'bounds': (width//2, 0, width, height//2), 'carrying_capacity': 30},
            Niche.WATER: {'bounds': (0, height//2, width//2, height), 'carrying_capacity': 20},
            Niche.ROCKY: {'bounds': (width//2, height//2, width, height), 'carrying_capacity': 10}
        }
        
        # Resource distribution per niche
        self.resources = {
            Niche.GRASSLAND: {'plant_growth_rate': 0.8, 'water_availability': 0.6},
            Niche.FOREST: {'plant_growth_rate': 0.9, 'water_availability': 0.7},
            Niche.WATER: {'plant_growth_rate': 0.3, 'water_availability': 1.0},
            Niche.ROCKY: {'plant_growth_rate': 0.1, 'water_availability': 0.2}
        }
    
    def get_niche_at_position(self, position: Vector2D) -> Niche:
        """Determine which niche a position belongs to."""
        for niche, data in self.niches.items():
            x1, y1, x2, y2 = data['bounds']
            if x1 <= position.x < x2 and y1 <= position.y < y2:
                return niche
        return Niche.GRASSLAND  # Default fallback
    
    def keep_in_bounds(self, position: Vector2D) -> Vector2D:
        """Keep position within environment boundaries."""
        x = max(0, min(self.width - 1, position.x))
        y = max(0, min(self.height - 1, position.y))
        return Vector2D(x, y)
    
    def get_resource_quality(self, position: Vector2D, resource_type: str) -> float:
        """Get resource quality at a specific position."""
        niche = self.get_niche_at_position(position)
        return self.resources[niche].get(resource_type, 0.0)


class Organism:
    """Base class for all organisms in the ecosystem."""
    
    def __init__(self, species_id: str, species_type: SpeciesType, position: Vector2D, 
                 age: int = 0, energy: float = 100.0):
        self.species_id = species_id
        self.species_type = species_type
        self.position = position.copy()
        self.age = age
        self.energy = energy
        self.max_energy = 100.0
        self.reproduction_threshold = 80.0
        self.death_threshold = 0.0
        self.max_age = 100
        self.speed = 2.0
        self.vision_range = 10.0
        self.alive = True
        
        # Species-specific attributes
        self.preferred_niches = self._get_preferred_niches()
        self.diet = self._get_diet()
        self.reproduction_rate = self._get_reproduction_rate()
        self.energy_consumption_rate = self._get_energy_consumption_rate()
    
    def _get_preferred_niches(self) -> List[Niche]:
        """Get preferred niches for this species type."""
        preferences = {
            SpeciesType.PLANT: [Niche.GRASSLAND, Niche.FOREST],
            SpeciesType.HERBIVORE: [Niche.GRASSLAND, Niche.FOREST],
            SpeciesType.CARNIVORE: [Niche.FOREST, Niche.ROCKY],
            SpeciesType.OMNIVORE: [Niche.GRASSLAND, Niche.FOREST, Niche.WATER]
        }
        return preferences.get(self.species_type, [Niche.GRASSLAND])
    
    def _get_diet(self) -> List[SpeciesType]:
        """Get what this species can eat."""
        diets = {
            SpeciesType.PLANT: [],  # Plants don't eat other organisms
            SpeciesType.HERBIVORE: [SpeciesType.PLANT],
            SpeciesType.CARNIVORE: [SpeciesType.HERBIVORE],
            SpeciesType.OMNIVORE: [SpeciesType.PLANT, SpeciesType.HERBIVORE]
        }
        return diets.get(self.species_type, [])
    
    def _get_reproduction_rate(self) -> float:
        """Get reproduction probability per step when conditions are met."""
        rates = {
            SpeciesType.PLANT: 0.005,  # Much slower plant reproduction
            SpeciesType.HERBIVORE: 0.008,
            SpeciesType.CARNIVORE: 0.003,
            SpeciesType.OMNIVORE: 0.005
        }
        return rates.get(self.species_type, 0.001)
    
    def _get_energy_consumption_rate(self) -> float:
        """Get energy consumption per step."""
        rates = {
            SpeciesType.PLANT: 0.1,  # Plants lose less energy
            SpeciesType.HERBIVORE: 0.5,
            SpeciesType.CARNIVORE: 0.8,
            SpeciesType.OMNIVORE: 0.6
        }
        return rates.get(self.species_type, 0.5)
    
    def can_eat(self, other: 'Organism') -> bool:
        """Check if this organism can eat another organism."""
        return other.species_type in self.diet and other.alive
    
    def find_food(self, organisms: List['Organism'], environment: EcosystemEnvironment) -> Optional['Organism']:
        """Find nearest edible organism within vision range."""
        edible_organisms = [org for org in organisms 
                          if self.can_eat(org) and 
                          self.position.distance_to(org.position) <= self.vision_range]
        
        if not edible_organisms:
            return None
        
        # Return the nearest food source
        return min(edible_organisms, key=lambda org: self.position.distance_to(org.position))
    
    def move_towards(self, target_position: Vector2D, environment: EcosystemEnvironment):
        """Move towards a target position."""
        if self.species_type == SpeciesType.PLANT:
            return  # Plants don't move
        
        direction = (target_position - self.position).normalize()
        new_position = self.position + direction * self.speed
        self.position = environment.keep_in_bounds(new_position)
    
    def reproduce(self, environment: EcosystemEnvironment) -> Optional['Organism']:
        """Attempt to reproduce if conditions are met."""
        if (self.energy >= self.reproduction_threshold and 
            random.random() < self.reproduction_rate):
            
            # Check local population density (carrying capacity)
            niche = environment.get_niche_at_position(self.position)
            carrying_capacity = environment.niches[niche]['carrying_capacity']
            
            # Count nearby organisms of same type within a reasonable distance
            nearby_distance = 15.0
            nearby_same_type = 0
            # This would need to be calculated from outside, so we'll simplify
            # and just reduce reproduction chance based on energy
            
            # Only reproduce if not at maximum energy (suggests abundance)
            if self.energy >= self.max_energy * 0.95:
                return None  # Too much competition for resources
            
            # Create offspring near parent
            offset = Vector2D(random.uniform(-5, 5), random.uniform(-5, 5))
            offspring_position = environment.keep_in_bounds(self.position + offset)
            
            offspring = Organism(
                species_id=self.species_id,
                species_type=self.species_type,
                position=offspring_position,
                age=0,
                energy=self.max_energy * 0.5  # Start with less energy
            )
            
            # Parent loses more energy from reproduction
            self.energy -= 40.0
            
            return offspring
        
        return None
    
    def update(self, organisms: List['Organism'], environment: EcosystemEnvironment) -> Optional['Organism']:
        """Update organism state for one simulation step."""
        if not self.alive:
            return None
        
        # Age and consume energy
        self.age += 1
        self.energy -= self.energy_consumption_rate
        
        # Check for death conditions
        if self.energy <= self.death_threshold or self.age >= self.max_age:
            self.alive = False
            return None
        
        # Plants gain energy from photosynthesis based on niche
        if self.species_type == SpeciesType.PLANT:
            plant_growth = environment.get_resource_quality(self.position, 'plant_growth_rate')
            energy_gain = plant_growth * 2.0  # Reduced energy gain
            self.energy += energy_gain
            self.energy = min(self.energy, self.max_energy)
        else:
            # Animals try to find and eat food
            food = self.find_food(organisms, environment)
            if food:
                self.move_towards(food.position, environment)
                
                # If close enough, eat the food
                if self.position.distance_to(food.position) <= 2.0:
                    energy_gained = min(food.energy, 50.0)
                    self.energy += energy_gained
                    self.energy = min(self.energy, self.max_energy)
                    food.alive = False  # Food is consumed
            else:
                # Random movement if no food found
                if self.species_type != SpeciesType.PLANT:
                    random_direction = Vector2D(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
                    new_position = self.position + random_direction * (self.speed * 0.5)
                    self.position = environment.keep_in_bounds(new_position)
        
        # Try to reproduce
        return self.reproduce(environment)


class Population:
    """Manages a population of organisms of the same species."""
    
    def __init__(self, species_id: str, species_type: SpeciesType, initial_count: int, 
                 environment: EcosystemEnvironment):
        self.species_id = species_id
        self.species_type = species_type
        self.organisms = []
        
        # Create initial population
        for _ in range(initial_count):
            position = Vector2D(
                random.uniform(0, environment.width),
                random.uniform(0, environment.height)
            )
            organism = Organism(species_id, species_type, position)
            self.organisms.append(organism)
    
    def update(self, all_organisms: List[Organism], environment: EcosystemEnvironment):
        """Update all organisms in this population."""
        new_organisms = []
        
        for organism in self.organisms:
            offspring = organism.update(all_organisms, environment)
            if offspring:
                new_organisms.append(offspring)
        
        # Add new organisms to population
        self.organisms.extend(new_organisms)
        
        # Remove dead organisms
        self.organisms = [org for org in self.organisms if org.alive]
    
    def get_count(self) -> int:
        """Get current population count."""
        return len(self.organisms)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get population statistics."""
        if not self.organisms:
            return {
                'count': 0,
                'avg_age': 0,
                'avg_energy': 0,
                'total_energy': 0
            }
        
        ages = [org.age for org in self.organisms]
        energies = [org.energy for org in self.organisms]
        
        return {
            'count': len(self.organisms),
            'avg_age': sum(ages) / len(ages),
            'avg_energy': sum(energies) / len(energies),
            'total_energy': sum(energies)
        }


class EcosystemSimulation:
    """Main ecosystem simulation managing all species and interactions."""
    
    def __init__(self, width: float = 200.0, height: float = 200.0):
        self.environment = EcosystemEnvironment(width, height)
        self.populations = {}
        self.step_count = 0
        self.max_steps = 10000
        
        # Initialize diverse ecosystem
        self._initialize_ecosystem()
        
        # Tracking
        self.statistics_history = []
    
    def _initialize_ecosystem(self):
        """Initialize the ecosystem with various species."""
        # Plants - primary producers
        self.populations['grass'] = Population('grass', SpeciesType.PLANT, 40, self.environment)
        self.populations['trees'] = Population('trees', SpeciesType.PLANT, 20, self.environment)
        
        # Herbivores - primary consumers  
        self.populations['rabbits'] = Population('rabbits', SpeciesType.HERBIVORE, 15, self.environment)
        self.populations['deer'] = Population('deer', SpeciesType.HERBIVORE, 8, self.environment)
        
        # Carnivores - secondary consumers
        self.populations['wolves'] = Population('wolves', SpeciesType.CARNIVORE, 3, self.environment)
        self.populations['hawks'] = Population('hawks', SpeciesType.CARNIVORE, 2, self.environment)
        
        # Omnivores
        self.populations['bears'] = Population('bears', SpeciesType.OMNIVORE, 2, self.environment)
    
    def get_all_organisms(self) -> List[Organism]:
        """Get all organisms from all populations."""
        all_organisms = []
        for population in self.populations.values():
            all_organisms.extend(population.organisms)
        return all_organisms
    
    def step(self) -> Dict[str, Any]:
        """Execute one simulation step."""
        all_organisms = self.get_all_organisms()
        
        # Update all populations
        for population in self.populations.values():
            population.update(all_organisms, self.environment)
        
        self.step_count += 1
        
        # Collect statistics
        stats = self.get_statistics()
        self.statistics_history.append(stats)
        
        return {
            'step': self.step_count,
            'total_organisms': sum(pop.get_count() for pop in self.populations.values()),
            'populations': {name: pop.get_count() for name, pop in self.populations.items()},
            'statistics': stats
        }
    
    def run_simulation(self, steps: int = 1000) -> Dict[str, Any]:
        """Run the simulation for a specified number of steps."""
        print(f"Starting ecosystem simulation for {steps} steps...")
        
        for step in range(steps):
            result = self.step()
            
            if step % 100 == 0 or step == steps - 1:
                print(f"Step {step + 1}/{steps}")
                print(f"  Total organisms: {result['total_organisms']}")
                for species, count in result['populations'].items():
                    print(f"  {species}: {count}")
                print("-" * 40)
            
            # Check for ecosystem collapse
            total_organisms = sum(result['populations'].values())
            if total_organisms == 0:
                print("Ecosystem collapse detected - all organisms died")
                break
        
        return {
            'final_step': self.step_count,
            'final_populations': {name: pop.get_count() for name, pop in self.populations.items()},
            'statistics_history': self.statistics_history[-10:],  # Last 10 steps
            'ecosystem_stable': sum(pop.get_count() for pop in self.populations.values()) > 0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ecosystem statistics."""
        stats = {}
        
        for name, population in self.populations.items():
            stats[name] = population.get_statistics()
        
        # Calculate ecosystem-wide metrics
        total_organisms = sum(pop.get_count() for pop in self.populations.values())
        total_energy = sum(pop.get_statistics()['total_energy'] for pop in self.populations.values())
        
        stats['ecosystem'] = {
            'total_organisms': total_organisms,
            'total_energy': total_energy,
            'species_diversity': len([pop for pop in self.populations.values() if pop.get_count() > 0]),
            'step': self.step_count
        }
        
        return stats
    
    def visualize_ecosystem(self):
        """Simple ASCII visualization of the ecosystem."""
        print("\nEcosystem Visualization:")
        print("P=Plant, H=Herbivore, C=Carnivore, O=Omnivore")
        
        # Create grid representation
        grid_size = 20
        grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
        
        all_organisms = self.get_all_organisms()
        for organism in all_organisms:
            if not organism.alive:
                continue
                
            # Scale position to grid
            x = int(organism.position.x * grid_size / self.environment.width)
            y = int(organism.position.y * grid_size / self.environment.height)
            x = max(0, min(grid_size - 1, x))
            y = max(0, min(grid_size - 1, y))
            
            # Use symbols based on species type
            if organism.species_type == SpeciesType.PLANT:
                grid[y][x] = 'P'
            elif organism.species_type == SpeciesType.HERBIVORE:
                grid[y][x] = 'H'
            elif organism.species_type == SpeciesType.CARNIVORE:
                grid[y][x] = 'C'
            elif organism.species_type == SpeciesType.OMNIVORE:
                grid[y][x] = 'O'
        
        # Print grid
        for row in grid:
            print(''.join(row))
        
        print(f"\nStep: {self.step_count}")
        for name, pop in self.populations.items():
            count = pop.get_count()
            if count > 0:
                print(f"{name}: {count}")


if __name__ == "__main__":
    # Demo the ecosystem simulation
    simulation = EcosystemSimulation()
    
    print("=== Ecosystem Simulation Demo ===")
    simulation.visualize_ecosystem()
    
    result = simulation.run_simulation(500)
    
    print("\n=== Final Results ===")
    print(f"Simulation completed after {result['final_step']} steps")
    print(f"Ecosystem stable: {result['ecosystem_stable']}")
    print("\nFinal populations:")
    for species, count in result['final_populations'].items():
        print(f"  {species}: {count}")
    
    simulation.visualize_ecosystem()