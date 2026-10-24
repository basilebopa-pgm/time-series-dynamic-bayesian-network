import pandas as pd
import inspect
import networkx as nx
from pgmpy.causal_discovery import HillClimbSearch, ExpertKnowledge


from pgmpy_extensions.DLGBN import DynamicBayesianNetwork


class DBNStructureLearner:

    def __init__(
        self,
        variables,
        order=1,
        scoring_method="bic-g",
        max_indegree=None,
        max_iter=10000,
        tabu_length=100,
        epsilon=1e-4,
        show_progress=True,
    ):

        if order < 1:
            raise ValueError("order must be >= 1.")

        self.variables = variables
        self.order = order
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

        if len(time_slices) <= self.order:
            raise ValueError(
                f"At least {self.order + 1} time slices "
                "are required."
            )

        rows = []

        for t in range(
            self.order,
            len(time_slices)
        ):

            current_time = time_slices[t]

            row = {}

            for variable in self.variables:

                column = (variable, current_time)

                row[
                    f"{variable}_0"
                ] = data[column].values

            for lag in range(1, self.order + 1):

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
            self.order + 1
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

    @staticmethod
    def _parse(node):
        v, l = node.rsplit("_", 1)
        return v, int(l)

    def _split_edges(self, G):
        all_edges = set(G.edges())
        mutual = {frozenset(e) for e in all_edges if (e[1], e[0]) in all_edges}

        intra, inter = [], []

        # 1) directed edges
        for p, c in all_edges:
            if frozenset((p, c)) in mutual:
                continue
            (pv, pl), (cv, cl) = self._parse(p), self._parse(c)
            if pl == 0 and cl == 0:
                intra.append((pv, cv))
            elif pl > 0 and cl == 0:
                inter.append((pv, cv, pl))

        # 2) undirected edges: orient lagged ones by time, intra ones acyclically
        g = nx.DiGraph()
        g.add_nodes_from(self.variables)
        g.add_edges_from(intra)

        for pair in sorted(mutual, key=lambda s: sorted(s)):
            a, b = sorted(pair)
            (av, al), (bv, bl) = self._parse(a), self._parse(b)

            if al == 0 and bl == 0:
                if nx.has_path(g, bv, av):
                    av, bv = bv, av
                g.add_edge(av, bv)
                intra.append((av, bv))
            elif al > 0 and bl == 0:
                inter.append((av, bv, al))      # past -> present
            elif bl > 0 and al == 0:
                inter.append((bv, av, bl))

        return intra, inter

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

        intra_edges, inter_edges = self._split_edges(learned_graph.causal_graph_)
        return {"graph": learned_graph.causal_graph_, "intra_edges": intra_edges, "inter_edges": inter_edges}

    def build_dbn(self, result, model_type="linear", order=None):

        inter_edges = result["inter_edges"]

        learned_order = max(
            (lag for _, _, lag in inter_edges),
            default=1,
        )

        order = max(1, learned_order if order is None else order)

        if order < learned_order:
            raise ValueError(
                f"order={order} is smaller than the largest learned lag "
                f"({learned_order})."
            )

        dbn = DynamicBayesianNetwork(model_type=model_type, order=order)

        for variable in self.variables:
            dbn.add_node(variable)         

        for parent, child in set(result["intra_edges"]):
            dbn.add_edge((parent, 0), (child, 0))

        for parent, child, lag in set(inter_edges):
            dbn.add_edge((parent, 0), (child, lag))

        return dbn