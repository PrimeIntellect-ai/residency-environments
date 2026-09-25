"""Small executable recipe: simulation, Python metrics, evolution, and harvesting.

Run with: blobkit generate generators/physim/recipes/small_search.py --registry outputs/small-search
This short-run mass metric is a workflow demonstration, not a quality benchmark.
"""

from blobkit import worlds
from blobkit.generation import HarvestAbove, MetricScore, SearchConfig, SearchRecipe, SimulationEvaluator, Variation


def measure(record):
    return {
        "mean_mass": sum(values[-1] for values in record["mass"].values()) / len(record["mass"]),
        "horizon": record["T"],
    }


def build_recipe():
    return SearchRecipe(
        name="small-mass-search",
        description="A short, local example of the complete generation and provenance workflow.",
        seeds=[worlds.load("m0"), worlds.load("m4")],
        config=SearchConfig(generations=2, offspring=2, population=2, seed=19),
        evaluate=SimulationEvaluator(measure, horizon=25, options={"L": 32, "workers": 1, "n_soup": 1, "noise": 0}),
        score=MetricScore("mean_mass"),
        propose=Variation(recombination_probability=0.25),
        harvest=HarvestAbove(0),
        parameters={"purpose": "workflow-example", "quality_claim": False},
    )
