import os
import json
from typing import List, Dict

import litellm
from pydantic import BaseModel, ValidationError
from graph import TaskGraph


class TaskPlannerAgent:
    def __init__(self, model: str = "gpt-4o-2024-08-06"):
        self.model = model
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        litellm.api_key = self.openai_api_key

    def _generate_graph(self, goal: str, max_tasks: int = 7) -> TaskGraph:
        schema = TaskGraph.model_json_schema(mode="validation")
        prompt = f"""
        Decompose the following goal into a task graph. Do not generate more than {max_tasks} tasks.

        Goal: {goal}
        Schema: {schema}
        output the task graph in JSON format
        """

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            graph_data = json.loads(
                response["choices"][0]["message"]["content"]
            )
            task_graph = TaskGraph.model_validate(graph_data)
            return task_graph
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(
                f"Error processing OpenAI response: {e}, Response: {response}"
            )
            return TaskGraph()  # Return empty graph on error
        except ValidationError as e:
            print(f"Validation error: {e}, Response: {response}")
            return TaskGraph()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return TaskGraph()

    def process_goal(self, goal: str, max_tasks: int = 5) -> TaskGraph:
        return self._generate_graph(goal, max_tasks)
