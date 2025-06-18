import re
import json
from typing import Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer
import openai

from models import Navigator, InferenceLLM
from template_extractor import TemplateMatcher
from utilities import (
    TRAJECTORY_BUILDING_PROMPT,
    TRAJECTORY_ADJUST_PROMPT,  
    STEP_INSTANTIATION_PROMPT,
    INFERENCE_SYSTEM_PROMPT,
    TRAJECTORY_REQUIREMENT_PROMPT,
    print_reasoning_trajectory,
    print_solution_template,
    print_step
)


class ReasonFlux:
    """Main controller class coordinating the entire reasoning workflow.
    
    Orchestrates the interaction between navigation, template matching, 
    and inference components to solve complex problems through structured reasoning.
    """

    def __init__(
        self,
        navigator_path: str,
        template_matcher_path: str,
        inference_path: str,
        template_path: str,
        same_model: bool = False  # Marked for potential future use
    ) -> None:
        """Initialize the ReasonFlux components.
        
        Args:
            navigator_path: Path to navigator model/resources
            template_matcher_path: Path to template matching model/resources
            inference_path: Path to inference model/resources
            template_path: Path to template catalog file
            same_model: Flag indicating if components share the same base model
        """
        self.template_matcher = TemplateMatcher(template_matcher_path, template_path)
        self.navigator = Navigator(navigator_path)

        # Configure model inheritance if paths match
        inherit_model = navigator_path == inference_path
        if inherit_model:
            self.inference_llm = InferenceLLM(
                inference_path,
                inherit_model,
                self.navigator.model,
                self.navigator.tokenizer
            )
        else:
            self.inference_llm = InferenceLLM(inference_path)

    def reason(self, problem: str, max_iter: int = 10):
        """Execute the complete reasoning workflow.
        
        Args:
            problem: Input problem statement to solve
            max_iter: Maximum iterations for reasoning process
            
        Returns:
            Dict containing:
                - answer: Final solution
                - trajectory: Complete reasoning steps
                - evaluation: Performance metrics
        """
        initial_thought = self._plan_initial_trajectory(problem)
        print_reasoning_trajectory(self.navigator.template, initial_thought)
        
        retrieved_template = self.template_matcher.search_template(
            chapter_query=self.navigator.template['General Knowledge Category'],
            section_query=self.navigator.template['Specific Direction'],
            method_query=self.navigator.template['Applied Method']
        )
        
        print_solution_template(retrieved_template['method'])
        self._dynamic_adjustment(self.navigator.reasoning_flow, retrieved_template)
        
        print('Reasoning flow adjusted successfully!')
        print_reasoning_trajectory(self.navigator.template, initial_thought)

        for step_idx in range(self.navigator.reasoning_rounds):
            current_step = self.navigator.reasoning_flow[step_idx]
            current_instruction = self.navigator.initialize_reason_problem(
                problem, current_step
            )
            
            current_thought, current_reasoning = self.inference_llm.interplay(
                current_instruction,
                problem,
                self.navigator.reasoning_instructions,
                self.navigator.instantiation
            )
            
            # Update state
            self.navigator.reasoning_instructions.append(current_instruction)
            self.navigator.instantiation.append(current_reasoning)

            print_step(
                step_num=step_idx + 1,
                current_step=current_step,
                navigator_thought=None,
                current_problem=current_instruction,
                current_thought=current_thought,
                current_solution=current_reasoning
            )

    def _plan_initial_trajectory(self, problem: str) -> List[str]:
        """Generate initial reasoning trajectory.
        
        Args:
            problem: Input problem statement
            
        Returns:
            List of initial reasoning steps
        """
        return self.navigator.initializing_reasoning_trajectory(
            TRAJECTORY_BUILDING_PROMPT, problem
        )

    def _dynamic_adjustment(
        self, 
        trajectory: List[Dict], 
        retrieved_template: Dict
    ) -> None:
        """Dynamically adjust reasoning flow based on retrieved templates.
        
        Args:
            trajectory: Current reasoning steps
            retrieved_template: Matched template from catalog
        """
        adjustment_prompt = f"""
        {TRAJECTORY_ADJUST_PROMPT}
        
        Original Reason Flow:
        {json.dumps(trajectory, indent=2)}
        
        Standard Solution Template:
        {json.dumps(retrieved_template, indent=2)}
        
        {TRAJECTORY_REQUIREMENT_PROMPT}
        """
        
        self.navigator.dynamic_adjustment(
            [{'role': 'system', 'content': adjustment_prompt}]
        )

