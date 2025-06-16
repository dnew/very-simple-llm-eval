"""
Core evaluation functionality for LLMs.
"""

import os
import json
from typing import Annotated, List, Optional, Any, Dict, Type
from pydantic import BaseModel, Field, create_model
from langchain_anthropic import ChatAnthropic
from langchain.tools import BaseTool, tool
from dotenv import load_dotenv
from dataclasses import dataclass
import types
from langchain_core.tools.structured import StructuredTool
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain_core.tools.simple import Tool
import pandas as pd
import asyncio
import csv

# Load environment variables from .env file
load_dotenv()

class EvaluationRule(BaseModel):
    """A single evaluation rule, containing a tool and its cost."""
    name: Annotated[str, "The name of the rule."]
    rule: Annotated[str, "A declarative statement expressing the rule to be evaluated."]
    cost: Annotated[int, "The number of points deducted from a starting score of 100 for this rule violation."]

    def to_tool(self) -> BaseTool:
        """Convert the rule to a LangChain tool."""
        return StructuredTool.from_function(
            func=lambda *args: None,
            name=self.name,
            description=(
                self.rule + "\n" +
                "When identifying violations you must select the starting and ending index so that you mark the whole of the logical whole of the offending text."
                "This means if the rule violation is a word, you must select the starting and ending index of the whole word."
                "If the rule violation is a phrase, you must select the starting and ending index of the whole phrase."
                "If the rule violation is a sentence, you must select the starting and ending index of the whole sentence."
                "If the rule violation is a paragraph, you must select the starting and ending index of the whole paragraph."
                "If the rule violation is a document, you must select the starting and ending index of the whole document."
            ),
            args_schema={
                "type": "object",
                "properties": {
                    "start_index": {
                        "type": "integer",
                        "description": "The character index in the model output where the rule violation starts."
                    },
                    "end_index": {
                        "type": "integer",
                        "description": "The character index in the model output where the rule violation ends."
                    }
                },
                "required": ["start_index", "end_index"]
            },
            infer_schema=False
        )

class RuleViolation(BaseModel):
    """A single rule violation."""
    name: str = Field(description="The name of the rule that was violated.")
    start_index: int = Field(description="The start index of the rule violation.")
    end_index: int = Field(description="The end index of the rule violation.")

class EvaluationScore(BaseModel):
    """Score for a single evaluation rule."""
    total_score: int = Field(description="The total score for the model output. Starts at 100 and is deducted for each rule violation.")
    rule_violations: List[RuleViolation] = Field(description="A list of rule violations. Each violation is a string that describes the rule violation.")

class Evaluation:
    """Wrapper around LangChain's Claude 4.0 Sonnet for rule-based evaluation."""

    def __init__(
        self,
        rules: List[EvaluationRule],
        api_key: Optional[str] = None,
        model_name: str = "claude-sonnet-4-20250514"
    ):
        """Initialize the evaluator with rules and optional API key.
        
        Args:
            rules: List of EvaluationRule objects
            api_key: Optional Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY env var
            model_name: Name of the Claude model to use
        """
        # Store rules
        self.rules = rules
        # Get API key from parameter or environment variable
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Please provide it either through the api_key parameter "
                "or set the ANTHROPIC_API_KEY environment variable."
            )
        # Bind all tools from rules
        tools = [rule.to_tool() for rule in rules]
        self.llm_model = ChatAnthropic(
            model=model_name,
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
            },
            anthropic_api_key=api_key
        ).bind_tools(tools)
        # Build an index of the rules by tool name for lookup during evaluation
        self.rule_index = {rule.name: rule for rule in rules}

    async def evaluate(
        self,
        model_output: str,
    ) -> EvaluationScore:
        """Evaluate the model output against all rules.
        
        Args:
            model_output: The output from the model to evaluate
            
        Returns:
            EvaluationScore object containing total score and rule violations
        """
        scoring_context = [
            SystemMessage(
                content=(
                    "You are a helpful assistant that evaluates the quality of the model output. "
                    "Please review the model output carefully and identify any rule violations using the tools provided."
                )
            ),
            HumanMessage(
                content=model_output
            )
        ]
        scoring_response = await self.llm_model.ainvoke(scoring_context)
        # Debug logging
        match scoring_response.content:
            case str():
                print(f"Debug - Raw response content: {scoring_response.content}")
            case list():
                print(f"Debug - Raw response content:")
                for item in scoring_response.content:
                    match item["type"]:
                        case "text":
                            print(f"text: {item['text']}")
                        case "thinking":
                            print(f"thinking: {item['thinking']}")
                        case _:
                            print(f"other - {item}")
            case _:
                print(f"Debug - Raw response content: {scoring_response.content}")

        print("\nDebug - Raw tool calls:")
        for tool_call in scoring_response.tool_calls:
            print(f"Tool: {tool_call['name']}")
            print(f"Args: {tool_call['args']}")
            print(f"Text at indices: {model_output[tool_call['args']['start_index']:tool_call['args']['end_index']]}")
            print("---")
        
        # Our response is a list of tool calls that record any rule violations in the model output.
        # We need to parse the response and return the scores for each rule.
        scope = 100
        rule_violations = []
        tool_calls = scoring_response.tool_calls
        for tool_call in tool_calls:
            rule = self.rule_index[tool_call["name"]]
            start_index = tool_call["args"]["start_index"]
            end_index = tool_call["args"]["end_index"]
            rule_violations.append(RuleViolation(
                name=rule.to_tool().name,
                start_index=start_index,
                end_index=end_index
            ))
            scope -= rule.cost
        return EvaluationScore(
            total_score=scope,
            rule_violations=rule_violations
        )

    async def evaluate_n(self, model_output: str, n: int) -> pd.DataFrame:
        """Run the evaluation n times concurrently and collect the results.
        
        Args:
            model_output: The output from the model to evaluate
            n: The number of times to run the evaluation
            
        Returns:
            DataFrame containing the results of each evaluation
        """
        # Use asyncio.gather to run evaluations concurrently
        tasks = [self.evaluate(model_output) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        
        # Convert results to a DataFrame for easier analysis
        df = pd.DataFrame([{
            'total_score': score.total_score,
            'rule_violations': score.rule_violations
        } for score in results])
        
        # Calculate statistical measures
        stats = {
            'mean': df['total_score'].mean(),
            'median': df['total_score'].median(),
            'standard_deviation': df['total_score'].std(),
            # Add more statistical measures as needed
        }
        
        print(f"Score statistics out of 100 for {n} evaluations:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Flatten and deduplicate rule violations
        all_violations = []
        for score in results:
            all_violations.extend(score.rule_violations)
        unique_violations = list({(v.name, v.start_index, v.end_index): v for v in all_violations}.values())
        
        return df

    def write_violations_to_csv(self, rule_violations: List[RuleViolation], model_output: str, output_file: str = "rule_violations.csv"):
        """Write rule violations to a CSV file.
        
        Args:
            rule_violations: List of RuleViolation objects.
            model_output: The text output from the model.
            output_file: The filename for the CSV report.
        """
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['rule_name', 'start_index', 'end_index', 'violation_text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for violation in rule_violations:
                writer.writerow({
                    'rule_name': violation.name,
                    'start_index': violation.start_index,
                    'end_index': violation.end_index,
                    'violation_text': model_output[violation.start_index:violation.end_index]
                })
        print(f"Rule violations written to {output_file}")
