import typing
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from pgmpy_extensions.NeuralCPD import NeuralGaussianCPD

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

    def __init__(self, ebunch=None, model_type="linear", order=1):
        
        if not isinstance(order, int) or order < 1:
            raise ValueError("order must be an integer >= 1.")
        
        if model_type not in {"linear", "neural"}:
            raise ValueError("model_type must be 'linear' or 'neural'.")
 
        self.order = order 
        DAG.__init__(self)
        self.model_type = model_type
        self.cpds = []
 
        if ebunch:
            self.add_edges_from(ebunch)

    def add_node(self, node, **attr):
        
        for t in range(self.order + 1):
            DAG.add_node(self, DynamicNode(node, t), **attr)

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
            
            if not isinstance(start[1], int) or not isinstance(end[1], int):
                raise ValueError("Nodes must be of type (node, time_slice).")
            
        except TypeError:
            raise ValueError("Nodes must be of type (node, time_slice).")
 
        u, v = start[0], end[0]
        lag = end[1] - start[1]
        k = self.order
 
        if lag < 0:
            raise NotImplementedError("Edges in backward direction are not allowed.")
        
        if lag > k:
            raise ValueError(f"Edge spans {lag} time slices but model order is {k}.")
        
        if lag == 0:
            if u == v:
                raise ValueError("Self Loops are not allowed")
            
            a, b = DynamicNode(u, k), DynamicNode(v, k)

            if a in self and b in self and nx.has_path(self, b, a):
                raise ValueError(f"Loops are not allowed. Adding the edge ({a} --> {b}) forms a loop.")
 
        self.add_node(u)
        self.add_node(v)
 
        for t in range(lag, k + 1):
            DAG.add_edge(self, DynamicNode(u, t - lag), DynamicNode(v, t), **kwargs)

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

        k = self.order
        return [e for e in self.edges() if e[1][1] == k and e[0][1] < k]


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

    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parents_at(self, node, s):

        """[(parent_name, lag), ...] of `node` in CPD-slice s (0..order), deterministic order."""
        ps = [(p.node, s - p.time_slice) for p in self.predecessors(DynamicNode(node, s))]

        return sorted(ps, key=lambda x: (x[1], str(x[0])))
 
    def _cols(self, node, t):

        """Absolute data columns (name, time) that the CPD used for `node` at time t depends on."""
        s = min(t, self.order)
        cpd = self.get_cpds(node=DynamicNode(node, s))

        if cpd is None:
            raise ValueError(f"No CPD found for {node}.")
        
        return [(p[0], t - (s - p[1])) for p in cpd.evidence]
 
    def _xy(self, data, node, s, parents, T):

        """Training matrices for CPD-slice s. Template slice (s == order) pools all windows."""
        windows = [s] if s < self.order else range(self.order, T)
        X_parts, y_parts = [], []

        for t in windows:
            y_parts.append(data[(node, t)].to_numpy(dtype=float))

            if parents:
                X_parts.append(np.column_stack([data[(p, t - lag)].to_numpy(dtype=float) for p, lag in parents]))

            else:
                X_parts.append(np.empty((len(data), 0)))

        return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)
 
    def _check_frame(self, data):

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pandas dataframe. Got: {type(data)}")
        
        if not all(isinstance(c, tuple) and len(c) == 2 for c in data.columns):
            raise ValueError("Data column names must be of the form (variable, time_slice).")
        
        slices = sorted(set(c[1] for c in data.columns))
        if slices != list(range(len(slices))):
            raise ValueError("Data time slices must be consecutive integers starting from 0.")
        
        if len(slices) < self.order + 1:
            raise ValueError(f"At least {self.order + 1} time slices are required for order {self.order}.")
        
        variables = self._nodes()
        missing = {(n, t) for n in variables for t in slices} - set(data.columns)

        if missing:
            raise ValueError(f"Missing required data columns: {sorted(missing, key=str)}")
        
        return len(slices), variables


    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _fit_linear(self, data, estimator="mle", std_estimator="unbiased"):
        if estimator.lower() != "mle":
            raise ValueError("Only Maximum Likelihood Estimation is supported.")
        if std_estimator not in {"mle", "unbiased"}:
            raise ValueError("std_estimator must be one of {'mle', 'unbiased'}")
 
        T, variables = self._check_frame(data)
        cpds = []
 
        for s in range(self.order + 1):
            for node in variables:
                parents = self._parents_at(node, s)
                X, y = self._xy(data, node, s, parents, T)
 
                if parents:
                    reg = LinearRegression().fit(X, y)
                    resid = y - reg.predict(X)
                    ddof = 0 if std_estimator == "mle" else 1 + len(parents)
                    beta = np.concatenate([[reg.intercept_], reg.coef_])
                else:
                    resid = y - y.mean()
                    ddof = 0 if std_estimator == "mle" else 1
                    beta = np.array([y.mean()])
 
                std = np.sqrt((resid ** 2).sum() / (len(y) - ddof))
                cpds.append(
                    LinearGaussianCPD(
                        variable=DynamicNode(node, s),
                        beta=beta,
                        std=std,
                        evidence=[DynamicNode(p, s - lag) for p, lag in parents],
                    )
                )
 
        self.cpds = []
        self.add_cpds(*cpds)
        self.check_model()
        return self
 
    def _fit_neural(self, data, hidden_dim=32, epochs=500, learning_rate=0.001, seed=None):
        T, variables = self._check_frame(data)
        cols = [(n, t) for n in variables for t in range(T)]
        if not all(pd.api.types.is_numeric_dtype(data[c]) for c in cols):
            raise ValueError("All model data columns must be numeric.")
        if not np.isfinite(data.loc[:, cols].to_numpy(dtype=float)).all():
            raise ValueError("Training data must contain only finite values.")
        if epochs < 1 or learning_rate <= 0:
            raise ValueError("epochs must be positive and learning_rate must be greater than zero.")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
 
        def train_cpd(variable, evidence, X, y):
            X = torch.as_tensor(X, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
            if len(y) < 2:
                raise ValueError(f"At least two observations are required to fit {variable}.")
            cpd = NeuralGaussianCPD(variable=variable, evidence=evidence if evidence else None, hidden_dim=hidden_dim)
            opt = torch.optim.Adam(cpd.parameters(), lr=learning_rate)
            cpd.train()
            for _ in range(epochs):
                opt.zero_grad()
                loss = -cpd.log_prob(y, X).mean()
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite training loss for {variable}.")
                loss.backward()
                opt.step()
            cpd.eval()
            return cpd
 
        cpds = []
        for s in range(self.order + 1):
            for node in variables:
                parents = self._parents_at(node, s)
                X, y = self._xy(data, node, s, parents, T)
                evidence = [DynamicNode(p, s - lag) for p, lag in parents]
                cpds.append(train_cpd(DynamicNode(node, s), evidence, X, y))
 
        self.cpds = []
        self.add_cpds(*cpds)
        self.check_model()
        return self

    def fit(self, data, estimator="mle", std_estimator="unbiased", **kwargs,):

        if self.model_type == "linear":
            return self._fit_linear(data, estimator=estimator, std_estimator=std_estimator, **kwargs,)

        return self._fit_neural(data, **kwargs,)
 
    # ------------------------------------------------------------------
    # Unrolling (linear only)
    # ------------------------------------------------------------------
    def _unroll(self, n_time_slices):
        if not self.cpds:
            raise ValueError("Model must be fitted before unrolling.")
 
        nodes = self._nodes()
        edges, unrolled = [], []
 
        for t in range(n_time_slices):
            for node in nodes:
                s = min(t, self.order)
                src = self.get_cpds(node=DynamicNode(node, s))
                ev = [f"{n}_{tt}" for n, tt in self._cols(node, t)]
                edges.extend((e, f"{node}_{t}") for e in ev)
                unrolled.append(
                    LinearGaussianCPD(
                        variable=f"{node}_{t}",
                        beta=src.beta.copy(),
                        std=src.std,
                        evidence=ev,
                    )
                )
 
        bn = LinearGaussianBayesianNetwork(edges)
        bn.add_nodes_from([f"{n}_{t}" for t in range(n_time_slices) for n in nodes])
        bn.add_cpds(*unrolled)
        bn.check_model()
        return bn
 

    # ------------------------------------------------------------------
    # Log-likelihood (linear and neural)
    # ------------------------------------------------------------------
    def log_likelihood(self, data):
        """
        Per-row log p(x_0, ..., x_{T-1}) under the DBN.
        Data columns must be (node, time_slice).
        """
        T = len(set(c[1] for c in data.columns))
        total = np.zeros(len(data))

        for t in range(T):
            for node in self._nodes():
                cpd = self.get_cpds(node=DynamicNode(node, min(t, self.order)))

                if cpd is None:
                    raise ValueError(f"No CPD found for {node}.")

                # Map CPD parents to the columns they refer to at time t
                cols = self._cols(node, t)
                y = data[(node, t)].to_numpy(dtype=float)

                if self.model_type == "linear":
                    mean = np.full(len(data), float(cpd.beta[0]), dtype=float)
                    if cols:
                        mean += data[cols].to_numpy(dtype=float) @ np.asarray(cpd.beta[1:], dtype=float)
                    total += norm.logpdf(y, loc=mean, scale=float(cpd.std))
                else:
                    X = torch.as_tensor(
                        data[cols].to_numpy(dtype=np.float32) if cols
                        else np.empty((len(data), 0), dtype=np.float32)
                    )
                    with torch.no_grad():
                        lp = cpd.log_prob(torch.as_tensor(y, dtype=torch.float32), X)
                    total += lp.numpy().astype(float)

        return total

    # ------------------------------------------------------------------
    # Ancestral sampler for neural DBNs
    # ------------------------------------------------------------------
    def _simulate_neural(self, n_samples, n_time_slices, seed=None):
        gen = torch.Generator().manual_seed(0 if seed is None else int(seed))
        slice0 = self.subgraph([DynamicNode(n, 0) for n in self._nodes()])
        order = [v.node for v in nx.topological_sort(slice0)]
        out = {}

        with torch.no_grad():
            for t in range(n_time_slices):
                for node in order:
                    cpd = self.get_cpds(node=DynamicNode(node, min(t, self.order)))
                    cols = self._cols(node, t)
                    X = (
                        torch.stack([out[c] for c in cols], dim=1) if cols
                        else torch.empty((n_samples, 0))
                    )
                    mean, var = cpd(X)
                    out[(node, t)] = mean + var.sqrt() * torch.randn(n_samples, generator=gen)

        sampled = pd.DataFrame({k: v.numpy() for k, v in out.items()})
        sampled.columns = pd.MultiIndex.from_tuples(list(sampled.columns))
        return sampled

    # ------------------------------------------------------------------
    # Conditional forecasting (linear and neural CPDs)
    # ------------------------------------------------------------------
    @staticmethod
    def _cpd_mean_std(cpd, parents, shape):
        """Mean array and std (array or float) of a CPD given parent value arrays."""
        if isinstance(cpd, NeuralGaussianCPD):
            n = int(np.prod(shape))
            if parents:
                X = np.stack([p.reshape(-1) for p in parents], axis=1).astype(np.float32)
            else:
                X = np.empty((n, 0), dtype=np.float32)
            with torch.no_grad():
                mean, var = cpd(torch.as_tensor(X))
            mean = mean.numpy().astype(float).reshape(shape)
            std = np.sqrt(var.numpy().astype(float)).reshape(shape)
            return mean, std

        beta = np.asarray(cpd.beta, dtype=float).ravel()
        mean = np.full(shape, beta[0])
        for w, p in zip(beta[1:], parents):
            mean = mean + w * p
        return mean, float(cpd.std)

    def forecast(self, history, n_steps, n_samples=200, seed=None):
        """
        Sample the future given observed history.

        history : DataFrame with columns (node, t) for t = 0..k-1 (k >= 1).
        Returns (samples, variables) where samples has shape
        (n_rows, n_samples, n_steps, n_variables): slice k+h, variable j.
        """
        rng = np.random.default_rng(seed)
        k = len(set(c[1] for c in history.columns))
        if k < self.order:
            raise ValueError(f"history needs at least {self.order} time slices for order {self.order}.")

        N = len(history)
        shape = (N, n_samples)
        nodes = sorted(self._nodes())

        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from((u[0], v[0]) for u, v in self.get_intra_edges(0))
        order = list(nx.topological_sort(g))

        state = {
            (n, t): np.repeat(history[(n, t)].to_numpy(dtype=float)[:, None], n_samples, axis=1)
            for n in nodes for t in range(k)
        }

        for h in range(n_steps):
            t = k + h
            for node in order:
                cpd = self.get_cpds(node=DynamicNode(node, self.order))
                cols = self._cols(node, t)
                mean, std = self._cpd_mean_std(cpd, [state[c] for c in cols], shape)
                state[(node, t)] = mean + std * rng.standard_normal(shape)

        out = np.stack(
            [np.stack([state[(n, k + h)] for n in nodes], axis=-1) for h in range(n_steps)],
            axis=2,
        )  # (N, S, n_steps, d)
        return out, nodes

    def simulate(self, n_samples=10, n_time_slices=2, seed=None, return_format="wide",):

        if n_time_slices < 1:
            raise ValueError("n_time_slices must be >= 1.")

        if not self.cpds:
            raise ValueError("Model must be fitted before simulation.")

        if any(isinstance(c, NeuralGaussianCPD) for c in self.cpds):
            sampled = self._simulate_neural(n_samples, n_time_slices, seed)
        else:
            bn = self._unroll(n_time_slices)

            sampled = bn.simulate(
                n_samples=n_samples,
                seed=seed,
            )

            # Convert columns back to (variable, time_slice)
            sampled.columns = [
                (
                    column.rsplit("_", 1)[0],
                    int(column.rsplit("_", 1)[1]),
                )
                for column in sampled.columns
            ]

        if return_format == "wide":
            return sampled

        if return_format == "long":

            rows = []

            for _, row in sampled.iterrows():

                for time in range(n_time_slices):

                    observation = {"time": time}

                    for node in self._nodes():
                        observation[node] = row[(node, time)]

                    rows.append(observation)

            return pd.DataFrame(rows)

        raise ValueError(
            "return_format must be either "
            "'wide' or 'long'."
        )