import streamlit as st
import networkx as nx
from pydantic import BaseModel
from typing import List
import plotly.graph_objects as go

class Task(BaseModel):
    id: str
    name: str
    status: str = "pending"
    depends_on: List[str] = []

# Sample tasks
tasks = [
    Task(id="1", name="Fetch Data", status="done"),
    Task(id="2", name="Clean Data", status="in_progress", depends_on=["1"]),
    Task(id="3", name="Train Model", status="pending", depends_on=["2"]),
    Task(id="4", name="Evaluate Model", status="pending", depends_on=["3"]),
    Task(id="5", name="Deploy Model", status="pending", depends_on=["4"]),
]

# Build graph
graph = nx.DiGraph()
for task in tasks:
    graph.add_node(task.id, label=task.name, status=task.status)
for task in tasks:
    for dep in task.depends_on:
        graph.add_edge(dep, task.id)

st.title("Task Dependency Graph with Plotly")

# Status to color mapping
status_colors = {
    "done": "green",
    "in_progress": "orange",
    "pending": "red",
}
node_colors = [status_colors.get(graph.nodes[n]["status"], "gray") for n in graph.nodes()]

# Compute layout
pos = nx.spring_layout(graph)

# Create edge traces
edge_x = []
edge_y = []
for src, dst in graph.edges():
    x0, y0 = pos[src]
    x1, y1 = pos[dst]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Create node trace
node_x = []
node_y = []
node_text = []
for node in graph.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(graph.nodes[node]["label"] + f" ({graph.nodes[node]['status']})")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition='top center',
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color=node_colors,
        size=20,
        line_width=2
    )
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

st.plotly_chart(fig, use_container_width=True)
