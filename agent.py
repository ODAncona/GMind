from cerebras.cloud.sdk import Cerebras
import json
from graph import TaskGraph, Node, Edge
from pydantic import BaseModel, ValidationError


class TaskPlannerAgent:
    def __init__(
        self, api_key: str, model: str = "llama-4-scout-17b-16e-instruct"
    ):
        self.client = Cerebras(api_key=api_key)
        self.model = model

    def _generate_graph(self, goal: str, max_tasks: int = 5) -> TaskGraph:
        schema = TaskGraph.model_json_schema()
        prompt = f"""
        Decompose the following goal into a task graph. Do not generate more than {max_tasks} tasks.

        Goal: {goal}

        Schema: {json.dumps(schema, indent=2)}

        Output: TaskGraph JSON No other text.
        """
        print(f"Prompt: {prompt}")
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "task_graph",
                    "strict": True,
                    "schema": schema,
                },
            },
        )

        try:
            graph_data = json.loads(completion.choices[0].message.content)
            task_graph = TaskGraph.model_validate(
                graph_data
            )  # Validate against your model
            return task_graph
        except json.JSONDecodeError as e:
            print(
                f"JSON decoding error: {e}, Raw response: {completion.choices[0].message.content}"
            )
            return TaskGraph()  # Return an empty graph on error
        except ValidationError as e:
            print(
                f"Validation error: {e}, Raw response: {completion.choices[0].message.content}"
            )
            return TaskGraph()  # Return an empty graph on error
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return TaskGraph()  # Return an empty graph on error

    def process_goal(self, goal: str, max_tasks: int = 5) -> TaskGraph:
        return self._generate_graph(goal, max_tasks)
