"""
Advanced Simulated Annealing Engine for XIBRA Network
Supports multi-objective optimization, parallel state exploration,
and adaptive cooling strategies with Boltzmann acceptance criteria
"""

import math
import logging
import time
import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import concurrent.futures

logger = logging.getLogger("xibra.optimize.annealing")

class AnnealingError(Exception):
    """Base exception for annealing process failures"""
    pass

class EnergyFunctionError(AnnealingError):
    """Raised when energy calculation fails"""
    pass

@dataclass(frozen=True)
class AnnealingState:
    configuration: Any
    energy: float
    temperature: float
    iteration: int

class CoolingSchedule(ABC):
    """Abstract base class for cooling strategies"""
    
    @abstractmethod
    def next_temperature(self, current_temp: float, iteration: int) -> float:
        pass

class ExponentialCooling(CoolingSchedule):
    """T(n) = T0 * alpha^n"""
    
    def __init__(self, alpha: float = 0.95):
        self.alpha = alpha
    
    def next_temperature(self, current_temp: float, iteration: int) -> float:
        return current_temp * self.alpha

class LogarithmicCooling(CoolingSchedule):
    """T(n) = T0 / log(1 + n)"""
    
    def next_temperature(self, current_temp: float, iteration: int) -> float:
        return current_temp / math.log(1 + iteration + 1)

class AdaptiveCooling(CoolingSchedule):
    """Auto-adjust alpha based on acceptance rate"""
    
    def __init__(self, initial_alpha: float = 0.95, target_accept: float = 0.4):
        self.alpha = initial_alpha
        self.target_accept = target_accept
        self._accept_history = deque(maxlen=100)
    
    def next_temperature(self, current_temp: float, iteration: int) -> float:
        accept_rate = np.mean(self._accept_history) if self._accept_history else 0.0
        if accept_rate < self.target_accept:
            self.alpha *= 0.99
        else:
            self.alpha = min(self.alpha * 1.01, 0.999)
        return current_temp * self.alpha
    
    def update_acceptance(self, accepted: bool):
        self._accept_history.append(accepted)

class SimulatedAnnealing:
    def __init__(
        self,
        initial_state: Any,
        energy_fn: Callable[[Any], float],
        neighbor_fn: Callable[[Any, float], Any],
        schedule: CoolingSchedule = ExponentialCooling(),
        t0: float = 1000.0,
        t_min: float = 1e-8,
        max_iter: int = 10000,
        parallel_workers: int = 4
    ):
        self.current_state = AnnealingState(
            configuration=initial_state,
            energy=energy_fn(initial_state),
            temperature=t0,
            iteration=0
        )
        self.energy_fn = energy_fn
        self.neighbor_fn = neighbor_fn
        self.schedule = schedule
        self.t0 = t0
        self.t_min = t_min
        self.max_iter = max_iter
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers)
        self._best_state = self.current_state
        self._history = []
        self._lock = asyncio.Lock()

    async def optimize(self) -> AnnealingState:
        """Asynchronous optimization loop"""
        try:
            async with self._lock:
                while not self._stopping_condition():
                    await self._annealing_step()
                return self._best_state
        except Exception as e:
            logger.error(f"Annealing failed: {str(e)}")
            raise AnnealingError(f"Optimization interrupted: {str(e)}") from e

    async def _annealing_step(self):
        """Parallel state exploration and decision"""
        # Generate candidate states in parallel
        loop = asyncio.get_event_loop()
        candidates = await loop.run_in_executor(
            self.executor,
            self._generate_candidates,
            5  # Number of parallel candidates
        )

        # Evaluate candidates
        tasks = [self._evaluate_candidate(c) for c in candidates]
        evaluated = await asyncio.gather(*tasks)

        # Select best candidate
        best_candidate = min(evaluated, key=lambda x: x.energy)
        self._update_state(best_candidate)

    def _generate_candidates(self, num: int) -> List[Any]:
        """Generate multiple neighbor states"""
        return [self.neighbor_fn(self.current_state.configuration, self.current_state.temperature) 
                for _ in range(num)]

    async def _evaluate_candidate(self, candidate: Any) -> AnnealingState:
        """Calculate energy for a candidate state"""
        try:
            energy = await asyncio.get_event_loop().run_in_executor(
                None,
                self.energy_fn,
                candidate
            )
            return AnnealingState(
                configuration=candidate,
                energy=energy,
                temperature=self.current_state.temperature,
                iteration=self.current_state.iteration
            )
        except Exception as e:
            logger.error(f"Energy calculation failed: {str(e)}")
            raise EnergyFunctionError(f"Invalid state {candidate}") from e

    def _update_state(self, candidate: AnnealingState):
        """Update current state based on Metropolis criterion"""
        delta_energy = candidate.energy - self.current_state.energy
        accept_prob = math.exp(-delta_energy / self.current_state.temperature)

        if delta_energy < 0 or np.random.random() < accept_prob:
            self.current_state = AnnealingState(
                configuration=candidate.configuration,
                energy=candidate.energy,
                temperature=self.schedule.next_temperature(
                    self.current_state.temperature,
                    self.current_state.iteration
                ),
                iteration=self.current_state.iteration + 1
            )
            if self.current_state.energy < self._best_state.energy:
                self._best_state = self.current_state
            self._history.append(self.current_state)
        else:
            self.current_state = AnnealingState(
                configuration=self.current_state.configuration,
                energy=self.current_state.energy,
                temperature=self.schedule.next_temperature(
                    self.current_state.temperature,
                    self.current_state.iteration
                ),
                iteration=self.current_state.iteration + 1
            )

    def _stopping_condition(self) -> bool:
        """Check termination criteria"""
        return (self.current_state.temperature < self.t_min 
                or self.current_state.iteration >= self.max_iter 
                or self._convergence_check())

    def _convergence_check(self) -> bool:
        """Check energy stabilization"""
        if len(self._history) < 100:
            return False
        last_energies = [s.energy for s in self._history[-100:]]
        return np.std(last_energies) < 1e-6

class MultiObjectiveSA(SimulatedAnnealing):
    """Pareto-optimal solution search using weighted sum method"""
    
    def __init__(
        self,
        initial_state: Any,
        energy_fns: List[Callable[[Any], float]],
        weights: List[float],
        **kwargs
    ):
        super().__init__(
            initial_state,
            self._combined_energy(energy_fns, weights),
            **kwargs
        )
        self.objective_fns = energy_fns
        self.weights = weights

    def _combined_energy(self, fns: List[Callable], weights: List[float]) -> Callable:
        """Create scalarized multi-objective function"""
        def wrapper(state: Any) -> float:
            values = [fn(state) for fn in fns]
            return sum(w * v for w, v in zip(weights, values))
        return wrapper

    def get_pareto_front(self, num_samples: int = 100) -> List[AnnealingState]:
        """Extract non-dominated solutions from history"""
        front = []
        for state in self._history:
            if not any(self._dominates(other, state) for other in self._history):
                front.append(state)
        return sorted(front, key=lambda x: x.energy)[:num_samples]

    def _dominates(self, a: AnnealingState, b: AnnealingState) -> bool:
        """Pareto dominance check"""
        a_vals = [fn(a.configuration) for fn in self.objective_fns]
        b_vals = [fn(b.configuration) for fn in self.objective_fns]
        return all(a <= b for a, b in zip(a_vals, b_vals)) and any(a < b for a, b in zip(a_vals, b_vals))

# Unit Tests
import unittest
from unittest import IsolatedAsyncioTestCase

class TestSimulatedAnnealing(IsolatedAsyncioTestCase):
    async def test_basic_optimization(self):
        # Minimize f(x) = x^2
        def energy(x):
            return x**2
        
        def neighbor(x, temp):
            return x + np.random.normal(0, temp**0.5)
        
        sa = SimulatedAnnealing(
            initial_state=10.0,
            energy_fn=energy,
            neighbor_fn=neighbor,
            t0=100.0,
            max_iter=1000
        )
        result = await sa.optimize()
        self.assertLess(abs(result.configuration), 1.0)

    async def test_multi_objective(self):
        # Minimize f1(x)=x, f2(x)=(x-2)^2
        def f1(x):
            return x
        
        def f2(x):
            return (x - 2)**2
        
        sa = MultiObjectiveSA(
            initial_state=0.0,
            energy_fns=[f1, f2],
            weights=[0.5, 0.5],
            neighbor_fn=lambda x, t: x + np.random.uniform(-1, 1),
            t0=10.0,
            max_iter=500
        )
        await sa.optimize()
        front = sa.get_pareto_front()
        self.assertGreater(len(front), 10)

if __name__ == "__main__":
    unittest.main()
