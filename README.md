
# GMind - Enhanced Specification

`streamlit run main.py`

## Project Overview

GMind is an AI-powered planning orchestration system that decomposes complex user goals into actionable sub-tasks, represents them as a dynamic graph structure, and facilitates execution tracking through an interactive visual interface.

## Core Components

### 1. Graph Engine

- **Data Structure**: Implements a directed acyclic graph (DAG) using a custom Pydantic model with NetworkX integration for graph algorithms
- **Node Schema**:
  - Unique identifier
  - Task description
  - Status (pending, in-progress, completed, failed)
  - Input/output schema definitions
- **Edge Properties**:
  - Dependency type (hard/soft)
  - Data transfer specification

### 2. User Interface

- **Dashboard Components**:
  - Interactive graph visualization with Plotly
  - Task details panel with CRUD operations
  - Progress tracking and status indicators
  - Timeline view for temporal dependencies
- **User Interactions**:
  - Zoom, pan, and node selection
  - Manual graph editing capabilities
  - Filtering and search functionality
  - Task detail inspection on node click
- **Visualization Features**:
  - Color-coding based on task status
  - Visual indicators for critical path
  - Collapsible sub-graphs for complex plans
  - Multiple layout algorithms for different view modes

### 3. Orchestrator Agent

- **Input Processing**:
  - Natural language goal parsing
- **Planning Capabilities**:
  - Multi-step reasoning using Claude 3.7's capabilities
  - Hierarchical task decomposition strategies
  - Dependency identification and validation
  - Resource allocation and constraint resolution
- **Integration with Graph**:
  - Real-time graph construction as planning proceeds
  - Modification of existing graphs based on new information
  - Handling of plan conflicts and dependency violations
- **Execution Monitoring**:
  - Progress tracking based on completed sub-tasks
  - Re-planning capabilities when obstacles are encountered
  - Automated status updates and reporting

## Technical Implementation

- **Backend**:
  - Pydantic for data validation and serialization
  - LangChain for orchestrating LLM interactions
  - Claude 3.7 for reasoning and planning capabilities
- **Frontend**:
  - Streamlit for rapid dashboard development
  - Plotly for interactive graph visualization
  - Custom CSS for enhanced UI/UX

## Development Roadmap

1. Core graph data structure implementation
2. Basic agent reasoning capabilities
3. Initial UI for graph visualization
4. Integration of all components
5. Testing and performance optimization
6. User feedback and iteration

This enhanced specification provides a comprehensive framework for implementing the GMind project, with clear definitions of components, their interactions, and technical requirements.
