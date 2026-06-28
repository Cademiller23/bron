"""§8.7 Integration tests — the full load → score / generate → score pipelines."""

import math
import os

import darwin.problem as problem_pkg
from darwin.problem import loader, oracle
from darwin.problem.scorer import score
from darwin.problem.schemas import ObjectiveWeights, ProblemClass, Solution
from darwin.problem.generator import generate_instance
from darwin.problem.tests._helpers import make_solution

DATA = os.path.join(os.path.dirname(problem_pkg.__file__), "data")
COST = ObjectiveWeights.cost_only()


def test_happy_path_load_solve_score():
    loader.clear_cache()
    inst = loader.load_instance("industryor", os.path.join(DATA, "industryor_sample.json"))
    # verify the (labelled) optimum before trusting it as a denominator
    agrees, labeled, solver_value, _ = oracle.verify_label(inst)
    assert agrees and math.isclose(labeled, solver_value)

    result = oracle.solve_optimum(inst)
    sb = score(inst, result.solution, COST)
    assert sb.feasible
    assert math.isclose(sb.normalized_score, 1.0)


def test_bad_answer_ranks_below_optimal():
    loader.clear_cache()
    inst = loader.load_instance("industryor", os.path.join(DATA, "industryor_sample.json"))
    optimal = score(inst, oracle.solve_optimum(inst).solution, COST)
    poor = score(inst, make_solution(inst, {"S1-D1": 1.0}), COST)  # leaves most demand unmet
    assert not poor.feasible
    assert poor.final_fitness < optimal.final_fitness


def test_generate_and_score_is_sensible():
    inst = generate_instance(seed=99, problem_class=ProblemClass.TRANSPORTATION)
    # score the oracle's (valid) flow -> feasible, fitness in [0, 1]
    valid = score(inst, oracle.solve_optimum(inst).solution, COST)
    assert valid.feasible
    assert 0.0 <= valid.final_fitness <= 1.0

    # an empty solution is infeasible and ranks strictly lower
    empty = score(inst, Solution(solution_id="empty", instance_id=inst.instance_id, flows=[]), COST)
    assert empty.final_fitness < valid.final_fitness


def test_end_to_end_determinism():
    def run():
        loader.clear_cache()
        inst = loader.load_instance("industryor", os.path.join(DATA, "industryor_sample.json"))
        sol = oracle.solve_optimum(inst).solution
        return score(inst, sol, COST).final_fitness

    assert run() == run()


def test_generated_vrp_end_to_end():
    inst = generate_instance(seed=21, problem_class=ProblemClass.VEHICLE_ROUTING)
    result = oracle.solve_optimum(inst)
    sb = score(inst, result.solution, COST)
    assert sb.feasible and math.isclose(sb.normalized_score, 1.0)
