import typing
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.NeuralCPD import NeuralGaussianCPD

from pgmpy import config
from pgmpy.base import DAG

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork

@dataclass(eq=True, frozen=True)
class DynamicNode:
    """
    Class for representing the nodes of Dynamic Bayesian Networks.
    """

    node: str
    time_slice: int

    def __getitem__(self, idx: int) -> str | int:
        if idx == 0:
            return self.node
        elif idx == 1:
            return self.time_slice
        else:
            raise IndexError(f"Index {idx} out of bounds.")

    def __str__(self) -> str:
        return f"({self.node}, {self.time_slice})"

    def __repr__(self) -> str:
        return f"<DynamicNode({self.node}, {self.time_slice}) at {hex(id(self))}>"

    def __lt__(self, other) -> bool:
        if self.node < other.node:
            return True
        elif self.node > other.node:
            return False
        else:
            return self.time_slice < other.time_slice

    def __le__(self, other) -> bool:
        if self.node <= other.node:
            return True
        elif self.node > other.node:
            return False
        else:
            return self.time_slice <= other.time_slice

    def __eq__(self, other) -> bool:
        if isinstance(other, DynamicNode):
            return (self.node, self.time_slice) == (other.node, other.time_slice)
        elif isinstance(other, (list, tuple)):
            return (self.node, self.time_slice) == tuple(other)
        else:
            return False

    def to_tuple(self) -> tuple:
        """
        Returns a tuple representation as (node, time_slice) for DynamicNode object.
        """
        return (self.node, self.time_slice)

class DynamicBayesianNetwork(DAG):
    def __init__(self, ebunch=None, model_type="linear"):
        super().__init__()

        if model_type not in {"linear", "neural"}:
            raise ValueError("model_type must be 'linear' or 'neural'.")
        
        self.model_type = model_type
        
        if ebunch:
            self.add_edges_from(ebunch)

        self.cpds = []

    def add_node(self, node, **attr):
        super().add_node(DynamicNode(node, 0), **attr)

    def add_nodes_from(self, nodes, **attr):
        for node in nodes:
            self.add_node(node)

    def _nodes(self):
        return list({node for node, timeslice in super().nodes()})

    def _timeslices(self):
        return list({timeslice for node, timeslice in super().nodes()})

    def add_edge(self, start, end, **kwargs):
        try:
            if len(start) != 2 or len(end) != 2:
                raise ValueError("Nodes must be of type (node, time_slice).")
            elif not isinstance(start[1], int) or not isinstance(end[1], int):
                raise ValueError("Nodes must be of type (node, time_slice).")
            elif start[1] == end[1]:
                start = (start[0], 0)
                end = (end[0], 0)
            elif start[1] == end[1] - 1:
                start = (start[0], 0)
                end = (end[0], 1)
            elif start[1] > end[1]:
                raise NotImplementedError("Edges in backward direction are not allowed.")
            elif start[1] != end[1]:
                raise ValueError("Edges over multiple time slices is not currently supported")
        except TypeError:
            raise ValueError("Nodes must be of type (node, time_slice).")

        start = DynamicNode(*start)
        end = DynamicNode(*end)

        if start == end:
            raise ValueError("Self Loops are not allowed")
        elif start in super().nodes() and end in super().nodes() and nx.has_path(self, end, start):
            raise ValueError(f"Loops are not allowed. Adding the edge from ({str(start)} --> {str(end)}) forms a loop.")

        super().add_edge(start, end, **kwargs)

        if start[1] == end[1]:
            super().add_edge(DynamicNode(start[0], 1 - start[1]), DynamicNode(end[0], 1 - end[1]))
        else:
            super().add_node(DynamicNode(end[0], 1 - end[1]))

    def add_edges_from(self, ebunch, **kwargs):
        for edge in ebunch:
            self.add_edge(edge[0], edge[1])

    def get_intra_edges(self, time_slice=0):

        if not isinstance(time_slice, int) or time_slice < 0:
            raise ValueError("The timeslice should be a positive value greater than or equal to zero")

        return [
            tuple(DynamicNode(x[0], time_slice) for x in edge) for edge in self.edges() if edge[0][1] == edge[1][1] == 0
        ]

    def get_inter_edges(self):
        return [edge for edge in self.edges() if edge[0][1] != edge[1][1]]

    def get_interface_nodes(self, time_slice=0):
        return [DynamicNode(edge[time_slice][0], edge[time_slice][1]) for edge in self.get_inter_edges()]

    def get_slice_nodes(self, time_slice=0):
        if not isinstance(time_slice, int) or time_slice < 0:
            raise ValueError("The timeslice should be a positive value greater than or equal to zero")

        return [DynamicNode(node, time_slice) for node in self._nodes()]

    def add_cpds(self, *cpds):

        expected_type = (LinearGaussianCPD if self.model_type == "linear" else NeuralGaussianCPD)

        for cpd in cpds:
            if not isinstance(cpd, expected_type):
                raise ValueError(f"{self.model_type} model requires " f"{expected_type.__name__}.")

            if cpd.variable not in super().nodes():
                raise ValueError(f"CPD variable {cpd.variable} is not in the model.")

        self.cpds.extend(cpds)

    def check_model(self):

        expected_type = (LinearGaussianCPD if self.model_type == "linear" else NeuralGaussianCPD)

        for node in super().nodes():
            cpd = self.get_cpds(node=node)

            if cpd is None:
                raise ValueError(f"No CPD associated with {node}.")

            if not isinstance(cpd, expected_type):
                raise ValueError(
                    f"CPD associated with {node} must be " f"an instance of {expected_type.__name__}.")

            evidence = set(cpd.evidence)
            parents = set(self.get_parents(node))

            if evidence != parents:
                raise ValueError(
                    f"CPD associated with {node} " "doesn't have proper parents associated with it.")

        return True

    def get_cpds(self, node=None, time_slice=None):

        if time_slice is None:
            time_slices = self._timeslices()

        elif isinstance(time_slice, int) and time_slice >= 0:
            time_slices = [time_slice]

        elif isinstance(time_slice, typing.Iterable):

            if all(isinstance(n, int) for n in time_slice):
                time_slices = time_slice

            else:
                raise ValueError(
                    "At least one element inside time_slice iterable "
                    "is not a non-negative integer."
                )

        else:
            raise ValueError(
                "time_slice must be a non-negative integer "
                "or an iterable of integers."
            )

        if node is not None:

            if node not in super().nodes():
                raise ValueError(
                    "Node not present in the model."
                )

            for cpd in self.cpds:
                if cpd.variable == node:
                    return cpd

            return None

        return_cpds = []

        for time_slice in time_slices:

            for var in self.get_slice_nodes(
                time_slice=time_slice
            ):

                cpd = self.get_cpds(node=var)

                if cpd is not None:
                    return_cpds.append(cpd)

        return return_cpds

    def get_constant_bn(self, t_slice=0):

        edges = [
            (
                f"{u[0]}_{u[1] + t_slice}",
                f"{v[0]}_{v[1] + t_slice}",
            )
            for u, v in self.edges()
        ]

        bn = LinearGaussianBayesianNetwork(edges)

        new_cpds = []

        for cpd in self.cpds:

            new_variables = [
                f"{var[0]}_{var[1] + t_slice}"
                for var in cpd.variables
            ]

            variable = new_variables[0]
            evidence = new_variables[1:]

            new_cpd = LinearGaussianCPD(
                variable=variable,
                beta=cpd.beta.copy(),
                std=cpd.std,
                evidence=evidence if evidence else None,
            )

            new_cpds.append(new_cpd)

        bn.add_cpds(*new_cpds)

        return bn

    def _fit_linear(self,data,estimator="mle",std_estimator="unbiased",):

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pandas dataframe. Got: {type(data)}")

        if estimator.lower() != "mle":
            raise ValueError("Only Maximum Likelihood Estimation is supported.")

        if std_estimator not in {"mle", "unbiased"}:
            raise ValueError("std_estimator must be one of {'mle', 'unbiased'}")

        if not all(isinstance(col, tuple) and len(col) == 2 for col in data.columns):
            raise ValueError("Data column names must be of the form " "(variable, time_slice).")

        time_slices = sorted(set(col[1] for col in data.columns))

        if time_slices[0] != 0:
            raise ValueError("Data time slices must start from 0.")

        n_time_slices = len(time_slices)

        if n_time_slices < 2:
            raise ValueError("At least two time slices are required.")

        variables = self._nodes()

        initial_cpds = []

        initial_edges = [
            (u[0],v[0])
            for u, v in self.get_intra_edges(0)
        ]

        initial_bn = LinearGaussianBayesianNetwork(initial_edges)
        initial_bn.add_nodes_from(variables)
        initial_df = data.loc[:,[(node, 0) for node in variables]].copy()
        initial_df.columns = variables
        initial_bn.fit(initial_df)

        for cpd in initial_bn.get_cpds():

            evidence = [DynamicNode(parent, 0) for parent in cpd.evidence]
            initial_cpds.append(
                LinearGaussianCPD(variable=DynamicNode(cpd.variable,0),
                    beta=cpd.beta.copy(),
                    std=cpd.std,
                    evidence=evidence if evidence else [],
                )
            )

        transition_rows = []

        for t in range(n_time_slices - 1):

            current = data.loc[:,[(node, t) for node in variables]].copy()

            next_slice = data.loc[:,[(node, t + 1) for node in variables]].copy()

            current.columns = [
                f"{node}__0"
                for node in variables
            ]

            next_slice.columns = [f"{node}__1" for node in variables]

            transition_rows.append(pd.concat( [current, next_slice], axis=1))

        transition_df = pd.concat(transition_rows, ignore_index=True)

        transition_cpds = []

        for node in variables:

            intra_parents = [
                u[0]
                for u, v in self.get_intra_edges(0)
                if v[0] == node
            ]

            inter_parents = [
                u[0]
                for u, v in self.get_inter_edges()
                if v[0] == node
            ]

            current_columns = [f"{parent}__1" for parent in intra_parents]
            previous_columns = [f"{parent}__0" for parent in inter_parents]
            parents = (current_columns + previous_columns)

            target_column = f"{node}__1"
            y = transition_df[target_column]

            if parents:

                X = transition_df[parents]
                regression = LinearRegression()
                regression.fit(X, y)
                residuals = (y - regression.predict(X))

                if std_estimator == "mle":
                    ddof = 0
                else:
                    ddof = 1 + len(parents)

                std = residuals.std(ddof=ddof)
                beta = np.concatenate([[regression.intercept_], regression.coef_,])

            else:
                beta = np.array([y.mean()])
                ddof = (0 if std_estimator == "mle" else 1)
                std = y.std(ddof=ddof)

            evidence = []

            for parent in intra_parents:
                evidence.append(DynamicNode(parent,1))

            for parent in inter_parents:
                evidence.append( DynamicNode(parent, 0))

            transition_cpds.append(
                LinearGaussianCPD(
                    variable=DynamicNode(node, 1),
                    beta=beta,
                    std=std,
                    evidence=evidence if evidence else None,
                )
            )

        self.cpds = []
        self.add_cpds(*initial_cpds, *transition_cpds)
        self.check_model()

        return self

    
    def _fit_neural(self, data, hidden_dim=32, epochs=500, learning_rate=0.001, seed=None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pandas dataframe. Got: {type(data)}")

        if not all(isinstance(col, tuple) and len(col) == 2 for col in data.columns):
            raise ValueError("Data column names must be of the form (variable, time_slice).")

        time_slices = sorted(set(col[1] for col in data.columns))

        if not time_slices or time_slices[0] != 0 or time_slices != list(range(len(time_slices))):
            raise ValueError("Data time slices must be consecutive integers starting from 0.")

        if len(time_slices) < 2:
            raise ValueError("At least two time slices are required.")

        variables = self._nodes()
        required_columns = {(node, t) for node in variables for t in time_slices}
        missing = required_columns - set(data.columns)

        if missing:
            raise ValueError(f"Missing required data columns: {sorted(missing, key=str)}")

        if not all(pd.api.types.is_numeric_dtype(data[col]) for col in required_columns):
            raise ValueError("All model data columns must be numeric.")

        if not data.loc[:, list(required_columns)].notna().all().all():
            raise ValueError("Training data must not contain missing values.")

        if not np.isfinite(data.loc[:, list(required_columns)].to_numpy(dtype=float)).all():
            raise ValueError("Training data must contain only finite values.")

        if epochs < 1 or learning_rate <= 0:
            raise ValueError("epochs must be positive and learning_rate must be greater than zero.")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = torch.device("cpu")

        def train_cpd(variable, evidence, X, y):
            X = torch.as_tensor(X, dtype=torch.float32, device=device)
            y = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1)

            if len(y) < 2:
                raise ValueError(f"At least two observations are required to fit {variable}.")

            cpd = NeuralGaussianCPD(variable=variable, evidence=evidence if evidence else None, hidden_dim=hidden_dim).to(device)
            optimizer = torch.optim.Adam(cpd.parameters(), lr=learning_rate)
            cpd.train()

            for _ in range(epochs):
                optimizer.zero_grad()
                loss = -cpd.log_prob(y, X).mean()

                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite training loss for {variable}.")

                loss.backward()
                optimizer.step()

            cpd.eval()
            return cpd

        initial_cpds = []
        transition_cpds = []
        intra_edges = self.get_intra_edges(0)
        inter_edges = self.get_inter_edges()

        # Fit initial-slice CPDs: P(X(0)).
        for node in variables:
            parents = [u[0] for u, v in intra_edges if v[0] == node]
            evidence = [DynamicNode(parent, 0) for parent in parents]
            X = data.loc[:, [(parent, 0) for parent in parents]].to_numpy(dtype=np.float32) if parents else np.empty((len(data), 0), dtype=np.float32)
            y = data[(node, 0)].to_numpy(dtype=np.float32)
            initial_cpds.append(train_cpd(DynamicNode(node, 0), evidence, X, y))

        # Pool all adjacent transitions.
        transition_data = []

        for t in range(len(time_slices) - 1):
            current = data.loc[:, [(node, t) for node in variables]].copy()
            next_slice = data.loc[:, [(node, t + 1) for node in variables]].copy()
            transition_data.append((t, current, next_slice))

        # Fit transition CPDs: P(X(t+1) | parents).
        for node in variables:
            intra_parents = [u[0] for u, v in intra_edges if v[0] == node]
            inter_parents = [u[0] for u, v in inter_edges if v[0] == node]
            evidence = [DynamicNode(parent, 1) for parent in intra_parents] + [DynamicNode(parent, 0) for parent in inter_parents]
            X_parts = []
            y_parts = []

            for t, current, next_slice in transition_data:
                parent_values = [next_slice[(parent, t + 1)].to_numpy(dtype=np.float32) for parent in intra_parents]
                parent_values += [current[(parent, t)].to_numpy(dtype=np.float32) for parent in inter_parents]
                X_parts.append(np.column_stack(parent_values) if parent_values else np.empty((len(current), 0), dtype=np.float32))
                y_parts.append(next_slice[(node, t + 1)].to_numpy(dtype=np.float32))

            X = np.concatenate(X_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
            transition_cpds.append(train_cpd(DynamicNode(node, 1), evidence, X, y))

        self.cpds = []
        self.add_cpds(*initial_cpds, *transition_cpds)
        self.check_model()

        return self


    def fit(self,data,estimator="mle",std_estimator="unbiased",**kwargs,):

        if self.model_type == "linear":
            return self._fit_linear(data,estimator=estimator,std_estimator=std_estimator,**kwargs,)

        return self._fit_neural(data,**kwargs,)


    def simulate(self,n_samples=10,n_time_slices=2,seed=None,return_format="wide",):

        if n_time_slices < 1:
            raise ValueError("n_time_slices must be >= 1.")

        if not self.cpds:
            raise ValueError("Model must be fitted before simulation.")

        edges = []

        for t in range(n_time_slices):

            # Intra-slice edges
            for u, v in self.get_intra_edges(0):

                edges.append(
                    (
                        f"{u[0]}_{t}",
                        f"{v[0]}_{t}",
                    )
                )

        # Inter-slice edges
        for t in range(n_time_slices - 1):

            for u, v in self.get_inter_edges():

                edges.append(
                    (
                        f"{u[0]}_{t}",
                        f"{v[0]}_{t + 1}",
                    )
                )

        # ---------------------------------------------------------
        # Create static LGBN
        # ---------------------------------------------------------

        bn = LinearGaussianBayesianNetwork(
            edges
        )

        all_nodes = [
            f"{node}_{t}"
            for t in range(n_time_slices)
            for node in self._nodes()
        ]

        bn.add_nodes_from(all_nodes)

        # ---------------------------------------------------------
        # Replicate CPDs
        # ---------------------------------------------------------

        unrolled_cpds = []

        for t in range(n_time_slices):

            for node in self._nodes():

                # t = 0 uses initial CPD
                if t == 0:

                    source_cpd = self.get_cpds(
                        node=DynamicNode(node, 0)
                    )

                # t > 0 uses transition CPD
                else:

                    source_cpd = self.get_cpds(
                        node=DynamicNode(node, 1)
                    )

                if source_cpd is None:
                    raise ValueError(
                        f"No CPD found for {node}."
                    )

                variable = (
                    f"{node}_{t}"
                )

                evidence = []

                for parent in source_cpd.evidence:

                    if t == 0:
                        # Initial CPDs: all parents are in slice 0
                        parent_name = f"{parent[0]}_0"

                    elif parent[1] == 0:
                        # Transition CPD: time-0 parent means previous slice
                        parent_name = f"{parent[0]}_{t - 1}"

                    else:
                        # Transition CPD: time-1 parent means current slice
                        parent_name = f"{parent[0]}_{t}"

                    evidence.append(parent_name)

                unrolled_cpds.append(
                    LinearGaussianCPD(
                        variable=variable,
                        beta=source_cpd.beta.copy(),
                        std=source_cpd.std,
                        evidence=evidence,
                    )
                )

        bn.add_cpds(
            *unrolled_cpds
        )

        bn.check_model()

        # ---------------------------------------------------------
        # Sample
        # ---------------------------------------------------------

        sampled = bn.simulate(
            n_samples=n_samples,
            seed=seed,
        )

        # ---------------------------------------------------------
        # Convert columns back to
        # (variable, time_slice)
        # ---------------------------------------------------------

        sampled.columns = [
            (
                column.rsplit("_", 1)[0],
                int(column.rsplit("_", 1)[1]),
            )
            for column in sampled.columns
        ]

        # ---------------------------------------------------------
        # Return format
        # ---------------------------------------------------------

        if return_format == "wide":
            return sampled

        if return_format == "long":

            rows = []

            for _, row in sampled.iterrows():

                for time in range(n_time_slices):

                    observation = {
                        "time": time
                    }

                    for node in self._nodes():
                        observation[node] = row[
                            (node, time)
                        ]

                    rows.append(observation)

            return pd.DataFrame(rows)

        raise ValueError(
            "return_format must be either "
            "'wide' or 'long'."
        )