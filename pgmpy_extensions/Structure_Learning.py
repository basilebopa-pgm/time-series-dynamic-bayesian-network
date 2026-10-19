import pandas as pd
import inspect
from pgmpy.causal_discovery import HillClimbSearch, ExpertKnowledge


from models.DLGBN import DynamicBayesianNetwork


class DBNStructureLearner:

    def __init__(
        self,
        variables,
        markov_order=1,
        scoring_method="bic-g",
        max_indegree=None,
        max_iter=10000,
        tabu_length=100,
        epsilon=1e-4,
        show_progress=True,
    ):

        if markov_order < 1:
            raise ValueError("markov_order must be >= 1.")

        self.variables = variables
        self.markov_order = markov_order
        self.scoring_method = scoring_method
        self.max_indegree = max_indegree
        self.max_iter = max_iter
        self.tabu_length = tabu_length
        self.epsilon = epsilon
        self.show_progress = show_progress

    def create_learning_data(self, data):

        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame.")

        if not all(
            isinstance(column, tuple) and len(column) == 2
            for column in data.columns
        ):
            raise ValueError(
                "Data columns must have the form (variable, time_slice)."
            )

        time_slices = sorted(
            set(column[1] for column in data.columns)
        )

        if len(time_slices) <= self.markov_order:
            raise ValueError(
                f"At least {self.markov_order + 1} time slices "
                "are required."
            )

        rows = []

        for t in range(
            self.markov_order,
            len(time_slices)
        ):

            current_time = time_slices[t]

            row = {}

            for variable in self.variables:

                column = (variable, current_time)

                row[
                    f"{variable}_0"
                ] = data[column].values

            for lag in range(1, self.markov_order + 1):

                previous_time = time_slices[t - lag]

                for variable in self.variables:

                    column = (
                        variable,
                        previous_time
                    )

                    row[
                        f"{variable}_{lag}"
                    ] = data[column].values

            rows.append(
                pd.DataFrame(row)
            )

        return pd.concat(
            rows,
            ignore_index=True
        )

    def create_expert_knowledge(self):

        nodes = []

        for lag in range(
            0,
            self.markov_order + 1
        ):
            for variable in self.variables:

                nodes.append(
                    f"{variable}_{lag}"
                )

        forbidden_edges = []

        for source in nodes:

            source_variable, source_lag = (
                source.rsplit("_", 1)
            )

            source_lag = int(source_lag)

            for target in nodes:

                if source == target:
                    continue

                target_variable, target_lag = (
                    target.rsplit("_", 1)
                )

                target_lag = int(target_lag)

                if (
                    source_lag > 0
                    and target_lag > 0
                ):
                    forbidden_edges.append(
                        (source, target)
                    )

                elif (
                    source_lag == 0
                    and target_lag > 0
                ):
                    forbidden_edges.append(
                        (source, target)
                    )

        return ExpertKnowledge(
            forbidden_edges=forbidden_edges
        )

    def DynamicHillClimbing(self, data):

        learning_data = self.create_learning_data(
            data
        )

        expert_knowledge = (
            self.create_expert_knowledge()
        )

        hc = HillClimbSearch(
            scoring_method=self.scoring_method,
            tabu_length=self.tabu_length,
            max_indegree=self.max_indegree,
            expert_knowledge=expert_knowledge,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            show_progress=self.show_progress,
        )

        learned_graph = hc.fit(
            learning_data
        )

        edges = list(
            learned_graph.causal_graph_.edges()
        )

        intra_edges = []
        inter_edges = []

        for parent, child in edges:

            parent_variable, parent_lag = (
                parent.rsplit("_", 1)
            )

            child_variable, child_lag = (
                child.rsplit("_", 1)
            )

            parent_lag = int(parent_lag)
            child_lag = int(child_lag)

            if (
                parent_lag == 0
                and child_lag == 0
            ):

                intra_edges.append(
                    (
                        parent_variable,
                        child_variable
                    )
                )


            elif (
                parent_lag > 0
                and child_lag == 0
            ):

                inter_edges.append(
                    (
                        parent_variable,
                        child_variable,
                        parent_lag
                    )
                )

        return {
            "graph": learned_graph.causal_graph_,
            "intra_edges": intra_edges,
            "inter_edges": inter_edges,
        }

    def build_dbn(self, result):

        if self.markov_order != 1:

            raise ValueError(
                "The current DynamicBayesianNetwork class "
                "only supports Markov order 1. "
                "Structure learning for higher orders is "
                "supported, but DLGBN.py must first be "
                "extended to represent higher-order edges."
            )

        dbn = DynamicBayesianNetwork()

        for variable in self.variables:
            dbn.add_node(variable)

        for parent, child in result["intra_edges"]:

            dbn.add_edge(
                (parent, 0),
                (child, 0)
            )

        for parent, child, lag in result["inter_edges"]:

            if lag != 1:
                continue

            dbn.add_edge(
                (parent, 0),
                (child, 1)
            )

        return dbn