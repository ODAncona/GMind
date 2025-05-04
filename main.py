import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import os
from typing import Dict, List, Optional
import json
import numpy as np

# Import our custom modules
from graph import TaskGraph, Node, NodeStatus, DependencyType
from agent import TaskPlannerAgent

from dotenv import load_dotenv
load_dotenv()

# Set page config
st.set_page_config(
    page_title="GMind - Task Planning",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if "graph" not in st.session_state:
    st.session_state.graph = TaskGraph()
if "selected_node" not in st.session_state:
    st.session_state.selected_node = None
if "agent" not in st.session_state:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        st.session_state.agent = TaskPlannerAgent(api_key=api_key)
    else:
        st.session_state.agent = None

# Functions for UI interactions
def update_node_status(node_id: str, new_status: str):
    """Update the status of a node in the graph."""
    st.session_state.graph.update_node_status(node_id, new_status)

def select_node(node_id: str):
    """Set the selected node in the session state."""
    st.session_state.selected_node = node_id

def create_graph_from_goal():
    """Create a graph from the user's goal using the agent."""
    if st.session_state.agent is None:
        st.error("Anthropic API key not set. Please set the ANTHROPIC_API_KEY environment variable.")
        return
    
    goal = st.session_state.goal_input
    if not goal:
        st.warning("Please enter a goal.")
        return
    
    with st.spinner("Generating plan..."):
        st.session_state.graph = st.session_state.agent.process_goal(goal)

def add_task_manually():
    """Add a task to the graph manually."""
    description = st.session_state.new_task_description
    if not description:
        st.warning("Please enter a task description.")
        return
    
    dependencies = st.session_state.new_task_dependencies.split(",") if st.session_state.new_task_dependencies else []
    dependencies = [dep.strip() for dep in dependencies if dep.strip()]
    
    node_id = st.session_state.graph.add_node(description=description)
    
    for dep_id in dependencies:
        if dep_id in st.session_state.graph.nodes:
            st.session_state.graph.add_edge(source=dep_id, target=node_id)

def advance_graph():
    """Progress in the graph by updating eligible nodes to in_progress."""
    graph = st.session_state.graph
    
    for node_id, node in graph.nodes.items():
        if node.status == "pending":
            predecessors = graph.get_predecessors(node_id)
            all_deps_completed = all(
                graph.get_node(pred_id).status == "completed" 
                for pred_id in predecessors
            )
            
            if all_deps_completed:
                graph.update_node_status(node_id, "in_progress")

# UI Layout
st.title("🧠 GMind - Task Planning Orchestrator")

# Sidebar for controls
with st.sidebar:
    st.header("Create New Plan")
    st.text_area("Enter your goal", key="goal_input", height=100, 
                placeholder="Describe what you want to accomplish...")
    st.button("Generate Plan", on_click=create_graph_from_goal)
    
    st.divider()
    
    st.header("Add Task Manually")
    st.text_input("Task Description", key="new_task_description")
    st.text_input("Dependencies (comma-separated IDs)", key="new_task_dependencies")
    st.button("Add Task", on_click=add_task_manually)
    
    st.divider()
    
    if st.session_state.graph.nodes:
        st.download_button(
            label="Export Graph",
            data=json.dumps(st.session_state.graph.to_dict(), indent=2),
            file_name="task_graph.json",
            mime="application/json"
        )

# Main area with tabs
tab1, tab2 = st.tabs(["Graph View", "Task List"])
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Avancer dans le graphe", key="advance_button"):
            advance_graph()
            st.rerun()
    
    # Graph visualization
    if not st.session_state.graph.nodes:
        st.info("No tasks in the graph. Create a plan or add tasks manually.")
    else:
        # Create NetworkX graph for visualization
        G = st.session_state.graph.to_networkx()
        
        try:
            # Compute layout
            try:
                if nx.is_planar(G):
                    pos = nx.planar_layout(G)
                else:
                    # Assurez-vous que pydot est installé: pip install pydot graphviz
                    pos = nx.nx_pydot.pydot_layout(G, prog='dot')
            except:
                # Fallback to spring layout if other methods fail
                pos = nx.spring_layout(G)
                
            # Node status colors
            status_colors = {
                "pending": "lightgray",  # Gris pour pending
                "in_progress": "blue",   # Vous pouvez ajuster cette couleur
                "completed": "green",    # Vert pour done  
                "failed": "red"          # Rouge pour fail
            }
            
            # Create edge traces
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                # Calculer la direction et réduire légèrement pour que la flèche ne touche pas le nœud
                dx, dy = x1 - x0, y1 - y0
                dist = (dx**2 + dy**2)**0.5
                if dist == 0:  # Éviter division par zéro
                    continue
                    
                # Réduire légèrement la longueur pour éviter de chevaucher les nœuds
                shrink_factor = 0.85
                end_x = x0 + dx * shrink_factor
                end_y = y0 + dy * shrink_factor
                
                edge_trace = go.Scatter(
                    x=[x0, end_x, None],
                    y=[y0, end_y, None],
                    line=dict(width=1.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'  # Retiré +markers pour éviter les flèches problématiques
                )
                edge_traces.append(edge_trace)
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_ids = []
            
            for node_id in G.nodes():
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                node_ids.append(node_id)
                
                node = st.session_state.graph.get_node(node_id)
                status = node.status if node else "pending"
                desc = node.description if node else "Unknown"
                
                node_text.append(f"{desc} ({status})")
                node_colors.append(status_colors.get(status, "gray"))
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition='top center',
                customdata=node_ids,
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    color=node_colors,
                    size=20,
                    line_width=2
                )
            )
            
            # Create figure - CORRECTION ICI: utiliser edge_traces et non edge_trace
            fig = go.Figure(
                data=[*edge_traces, node_trace],  # Utiliser edge_traces au lieu de edge_trace
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                    clickmode='event+select'
                )
            )
            
            # Display the graph
            selected_node = st.plotly_chart(fig, use_container_width=True, key="graph_plot")
            
            # Handle node selection
            if selected_node and isinstance(selected_node, dict) and 'points' in selected_node:
                for point in selected_node['points']:
                    if 'customdata' in point:
                        select_node(point['customdata'])
        
        except Exception as e:
            st.error(f"Error rendering graph: {str(e)}")
            # Fallback simple representation
            st.write("Task nodes:")
            for node_id, node in st.session_state.graph.nodes.items():
                st.write(f"- {node.description} ({node.status})")

with tab2:
    # Task list view
    if not st.session_state.graph.nodes:
        st.info("No tasks in the graph. Create a plan or add tasks manually.")
    else:
        st.subheader("All Tasks")
        
        # Sort nodes by status
        tasks_by_status = {
            "in_progress": [],
            "pending": [],
            "completed": [],
            "failed": []
        }
        
        for node_id, node in st.session_state.graph.nodes.items():
            tasks_by_status[node.status].append((node_id, node))
        
        # Display tasks grouped by status
        for status, color in [
            ("in_progress", "🟠 In Progress"),
            ("pending", "⚪ Pending"),
            ("completed", "🟢 Completed"),
            ("failed", "🔴 Failed")
        ]:
            st.write(f"### {color}")
            
            if not tasks_by_status[status]:
                st.write("No tasks in this status.")
                continue
            
            for node_id, node in tasks_by_status[status]:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{node.description}**")
                    st.write(f"ID: `{node_id}`")
                
                with col2:
                    new_status = st.selectbox(
                        "Status",
                        options=["pending", "in_progress", "completed", "failed"],
                        index=["pending", "in_progress", "completed", "failed"].index(node.status),
                        key=f"status_{node_id}"
                    )
                    
                    if new_status != node.status:
                        update_node_status(node_id, new_status)
                        st.rerun()
                
                # Show dependencies
                predecessors = st.session_state.graph.get_predecessors(node_id)
                if predecessors:
                    pred_nodes = [st.session_state.graph.get_node(p) for p in predecessors]
                    pred_text = ", ".join([f"{n.description}" for n in pred_nodes if n])
                    st.write(f"**Dependencies:** {pred_text}")
                
                st.divider()

# Detail panel for selected node
if st.session_state.selected_node:
    st.sidebar.header("Task Details")
    
    node = st.session_state.graph.get_node(st.session_state.selected_node)
    if node:
        st.sidebar.subheader(node.description)
        st.sidebar.write(f"**Status:** {node.status}")
        
        # Change status
        new_status = st.sidebar.selectbox(
            "Update Status",
            options=["pending", "in_progress", "completed", "failed"],
            index=["pending", "in_progress", "completed", "failed"].index(node.status)
        )
        
        if st.sidebar.button("Update Status"):
            update_node_status(st.session_state.selected_node, new_status)
            st.rerun()
        
        # Dependencies
        predecessors = st.session_state.graph.get_predecessors(st.session_state.selected_node)
        if predecessors:
            st.sidebar.subheader("Dependencies")
            for pred_id in predecessors:
                pred_node = st.session_state.graph.get_node(pred_id)
                if pred_node:
                    st.sidebar.write(f"- {pred_node.description} ({pred_node.status})")
        
        # Dependent tasks
        successors = st.session_state.graph.get_successors(st.session_state.selected_node)
        if successors:
            st.sidebar.subheader("Dependent Tasks")
            for succ_id in successors:
                succ_node = st.session_state.graph.get_node(succ_id)
                if succ_node:
                    st.sidebar.write(f"- {succ_node.description} ({succ_node.status})")
        
        # Delete node
        if st.sidebar.button("Delete Task", type="primary"):
            st.session_state.graph.remove_node(st.session_state.selected_node)
            st.session_state.selected_node = None
            st.rerun()