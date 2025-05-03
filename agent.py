from typing import List, Dict, Any
from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMChain
from langchain.chat_models import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from graph import TaskGraph, Node, NodeStatus, DependencyType


class TaskPlannerAgent:
    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(model="claude-3-7-sonnet-latest", api_key=api_key)
        self.graph = TaskGraph()
        self._setup_chains()
        self._setup_agent()
    
    def _setup_chains(self):
        """Set up LangChain chains for different task planning steps."""
        # Task decomposition chain
        decomposition_template = """
        Your task is to break down a complex goal into smaller, actionable tasks.
        
        Goal: {goal}
        
        Instructions:
        1. Break down the goal into 3-7 main tasks
        2. For each task, provide:
           - A clear, concise description
           - Any dependencies on other tasks
        
        Output the tasks as a numbered list with dependencies noted in parentheses.
        Example:
        1. Research potential database solutions
        2. Design database schema (depends on 1)
        3. Implement database (depends on 2)
        4. Create API endpoints (depends on 3)
        5. Build user interface
        6. Connect UI to API (depends on 4, 5)
        7. Test end-to-end solution (depends on 6)
        """
        self.decomposition_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["goal"],
                template=decomposition_template
            )
        )
    
    def _setup_agent(self):
        """Set up the LangChain agent with custom tools."""
        tools = [
            Tool(
                name="DecomposeGoal",
                func=self._decompose_goal,
                description="Break down a complex goal into smaller, actionable tasks"
            ),
            Tool(
                name="CreateGraph",
                func=self._create_graph_from_decomposition,
                description="Create a task graph from the decomposed tasks"
            )
        ]
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def _decompose_goal(self, goal: str) -> str:
        """Break down a goal into tasks."""
        return self.decomposition_chain.run(goal=goal)
    
    def _parse_decomposition(self, decomposition: str) -> List[Dict[str, Any]]:
        """Parse the LLM's task decomposition into structured data."""
        lines = decomposition.strip().split('\n')
        tasks = []
        
        for line in lines:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
                
            # Extract task number and text
            parts = line.split('. ', 1)
            if len(parts) < 2:
                continue
                
            task_num = int(parts[0])
            task_text = parts[1]
            dependencies = []
            
            # Extract dependencies
            if '(depends on' in task_text:
                main_text, dep_text = task_text.split('(depends on', 1)
                task_text = main_text.strip()
                dep_text = dep_text.strip().rstrip(')')
                
                # Parse dependency numbers
                for dep_part in dep_text.split(','):
                    try:
                        dep_num = int(dep_part.strip())
                        dependencies.append(dep_num)
                    except ValueError:
                        continue
            
            tasks.append({
                "id": task_num,
                "description": task_text,
                "dependencies": dependencies
            })
            
        return tasks
    
    def _create_graph_from_decomposition(self, decomposition: str) -> str:
        """Create a task graph from the decomposed tasks."""
        tasks = self._parse_decomposition(decomposition)
        
        # Clear the existing graph
        self.graph = TaskGraph()
        
        # First pass: create all nodes
        node_map = {}  # Maps task IDs to node IDs
        for task in tasks:
            node_id = self.graph.add_node(description=task["description"])
            node_map[task["id"]] = node_id
        
        # Second pass: add dependencies
        for task in tasks:
            target_node_id = node_map.get(task["id"])
            if not target_node_id:
                continue
                
            for dep_id in task["dependencies"]:
                source_node_id = node_map.get(dep_id)
                if source_node_id:
                    self.graph.add_edge(
                        source=source_node_id,
                        target=target_node_id,
                        dependency_type="hard"
                    )
        
        return f"Created graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
    
    def process_goal(self, goal: str) -> TaskGraph:
        """Process a user goal and return the created task graph."""
        # Step 1: Decompose the goal into tasks
        decomposition = self._decompose_goal(goal)
        
        # Step 2: Create the graph from the decomposed tasks
        self._create_graph_from_decomposition(decomposition)
        
        return self.graph