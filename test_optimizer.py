import unittest

import numpy as np

from main import ACOConfig, AntColonyOptimizer, DeliveryRoutingProblem


class TestACOEngine(unittest.TestCase):
    def make_optimizer(self, count=20):
        problem = DeliveryRoutingProblem(count, seed=42)
        return problem, AntColonyOptimizer(problem, ACOConfig(checkpoints=count, ants=12, iterations=12, seed=42))

    def test_route_is_closed_and_visits_each_checkpoint_once(self):
        problem, optimizer = self.make_optimizer(20)
        optimizer.step()
        self.assertEqual(len(optimizer.best_tour), 20)
        self.assertEqual(set(optimizer.best_tour), set(range(20)))
        self.assertGreater(problem.tour_distance(optimizer.best_tour), 0)

    def test_best_history_never_increases_and_average_is_recorded(self):
        _, optimizer = self.make_optimizer()
        for _ in range(12): optimizer.step()
        self.assertEqual(len(optimizer.best_history), 13)
        self.assertEqual(len(optimizer.average_history), 13)
        self.assertTrue(np.all(np.diff(optimizer.best_history) <= 1e-10))

    def test_pheromone_is_symmetric(self):
        _, optimizer = self.make_optimizer()
        for _ in range(4): optimizer.step()
        np.testing.assert_allclose(optimizer.pheromone, optimizer.pheromone.T)

    def test_same_seed_is_reproducible_for_supported_sizes(self):
        for size in (5, 20, 60):
            p1, o1 = self.make_optimizer(size); p2, o2 = self.make_optimizer(size)
            for _ in range(3): o1.step(); o2.step()
            np.testing.assert_allclose(p1.coordinates, p2.coordinates)
            self.assertEqual(o1.best_tour, o2.best_tour)
            self.assertAlmostEqual(o1.best_distance, o2.best_distance)


if __name__ == "__main__":
    unittest.main()

