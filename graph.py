from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field
import networkx as nx
import uuid
from enum import Enum


class NodeStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"


class DependencyType(str, Enum):
    hard = "hard"
    soft = "soft"


class Edge(BaseModel):
    source: str
    target: str
    dependency_type: DependencyType = DependencyType.hard
    data_transfer: Optional[Dict] = None


class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: NodeStatus = NodeStatus.pending
    inputs: Optional[Dict] = None
    outputs: Optional[Dict] = None


class TaskGraph(BaseModel):
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []
    _graph: Optional[nx.DiGraph] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """Build NetworkX graph from nodes and edges."""
        self._graph.clear()

        # Add nodes
        for node_id, node in self.nodes.items():
            self._graph.add_node(node_id, **node.dict(exclude={"id"}))

        # Add edges
        for edge in self.edges:
            self._graph.add_edge(
                edge.source,
                edge.target,
                dependency_type=edge.dependency_type,
                data_transfer=edge.data_transfer,
            )

    def add_node(self, description: str, **attributes) -> str:
        """Add a node to the graph and return its ID."""
        node_id = str(uuid.uuid4())
        node = Node(id=node_id, description=description, **attributes)
        self.nodes[node_id] = node
        self._graph.add_node(node_id, **node.dict(exclude={"id"}))
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        dependency_type: DependencyType = DependencyType.hard,
        data_transfer: Optional[Dict] = None,
    ):
        """Add an edge between two nodes."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Source or target node doesn't exist")

        edge = Edge(
            source=source,
            target=target,
            dependency_type=dependency_type,
            data_transfer=data_transfer,
        )
        self.edges.append(edge)
        self._graph.add_edge(
            source,
            target,
            dependency_type=dependency_type,
            data_transfer=data_transfer,
        )

    def update_node_status(self, node_id: str, status: NodeStatus):
        """Update the status of a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} doesn't exist")

        self.nodes[node_id].status = status
        self._graph.nodes[node_id]["status"] = status

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get predecessor node IDs for a given node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} doesn't exist")

        return list(self._graph.predecessors(node_id))

    def get_successors(self, node_id: str) -> List[str]:
        """Get successor node IDs for a given node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} doesn't exist")

        return list(self._graph.successors(node_id))

    def remove_node(self, node_id: str):
        """Remove a node and all its connected edges from the graph."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} doesn't exist")

        # Remove the node from the dictionary
        del self.nodes[node_id]

        # Remove edges involving this node
        self.edges = [
            e for e in self.edges if e.source != node_id and e.target != node_id
        ]

        # Update the NetworkX graph
        self._graph.remove_node(node_id)

    def is_dag(self) -> bool:
        """Check if the graph is a directed acyclic graph."""
        return nx.is_directed_acyclic_graph(self._graph)

    def get_critical_path(self) -> List[str]:
        """Get the critical path of the graph."""
        if not self.is_dag():
            raise ValueError("Graph must be a DAG to compute critical path")

        # Use topological sort to get the critical path
        return list(nx.topological_sort(self._graph))

    def get_leaf_nodes(self) -> List[str]:
        """Get all leaf nodes (nodes with no successors)."""
        return [n for n in self.nodes if self._graph.out_degree(n) == 0]

    def get_root_nodes(self) -> List[str]:
        """Get all root nodes (nodes with no predecessors)."""
        return [n for n in self.nodes if self._graph.in_degree(n) == 0]

    def to_networkx(self) -> nx.DiGraph:
        """Return the NetworkX graph representation."""
        return self._graph

    def to_dict(self) -> Dict:
        """Convert the graph to a dictionary for serialization."""
        return {
            "nodes": {k: v.dict() for k, v in self.nodes.items()},
            "edges": [e.dict() for e in self.edges],
        }
