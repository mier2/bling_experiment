TRAJECTORY_BUILDING_PROMPT = """
Please construct a reasoning trajectory in JSON format based on the following problem. This reasoning trajectory should include the following: problem description, general knowledge category, specific direction, applied method, examined knowledge, and reason_flow. Please output according to the given format:\n{\"Problem\": Here describes the problem you constructed, \"General Knowledge Category\": Here corresponds to the general category of mathematical knowledge to which the problem belongs, \"Specific Direction\": Here corresponds to the specific knowledge direction to which the problem belongs, \"Applied Method\": Here corresponds to the `template_name` of the input template, **please use its original name completely, do not refer, abbreviate or rewrite**, \"Examined Knowledge\": [Here is a list used to list the knowledge tags examined by this problem], \"reason_flow\": [This is a list, according to the `reason_flow` steps in the input template, to describe in detail the thinking process of solving the problem. Each step should be explained in conjunction with the specific situation of the problem, such as how to convert conditions, how to apply formulas, etc. But it should be noted that `reason_flow` is only a framework to guide students' thinking, and cannot directly give specific calculation results or answers, but should retain a certain degree of challenge for students to complete the specific calculations and derivations themselves.]}. Before providing a formal response, please carefully consider and analyze the question, and place your thoughts within <think></think> tags.
"""
TRAJECTORY_ADJUST_PROMPT = """
As a math problem-solving tutor, you need to optimize the original reasoning flow based on two inputs:
1. The user-provided **Original Reason Flow**
2. The retrieved **Standard Solution Template**

"""
TRAJECTORY_REQUIREMENT_PROMPT = """
Perform the following optimizations:
① **Step Consolidation**: Merge similar operations (e.g., multiple calculations → "Generate initial data")
② **Pattern Abstraction**: Convert concrete observations into methodological prompts (e.g., "Identify relationships with exponential terms")
③ **Logic Formalization**: Mark critical nodes using standard mathematical induction phases (base case/assumption/recursion)
④ **Gap Preservation**: Replace numerical computations with placeholders (e.g., "Analyze residual patterns")

Output Requirements:
- Maintain the logical framework from the standard template
- Each step contains only **1 core thinking instruction**
- Use methodological verbs (Observe/Hypothesize/Verify/Derive)
- Prohibit specific numerical values or algebraic operations
please carefully consider and analyze the question, and place your thoughts within <think></think> tags.
"""

STEP_INSTANTIATION_PROMPT = """You are an intelligent math teacher who are good at leading students step by step to solve difficult math problems. Now, based on the math problem and the current reasoning steps along with the results of previous steps, create an indepedent sub-problem for current step to guide the student solve the problem step by step. You should analyze the previous results and current step first, and then give clear guidance which could help to support solving the problem generated from current step, and stressed the error-prone parts of this generated problem. Please carefully consider and analyze the previous states and the current problems, and place your thoughts within <think></think> tags."""

FIRST_STEP_INSTANTIATION_PROMPT = """You are an intelligent math teacher who are good at leading students step by step to solve difficult math problems. Now, based on the math problem and the first reasoning steps, create an indepedent sub-problem for the first step to guide the student solve the problem step by step. You should analyze the previous results and current step first, and then give clear guidance which could help to support solving the problem generated from current step, and stressed the error-prone parts of this generated problem. Please carefully consider and analyze the previous states and the current problems, and place your thoughts within <think></think> tags."""

INFERENCE_SYSTEM_PROMPT ="""You are an intelligent student who are good at doing math, and you are now facing a math problem. Please strictly follow the teacher's instruction to solve the problem, and carefully analyze how to adopt the teacher's instruction to solve the problem, and pay attention to the error-prone the teacher mentioned. Pleased put your thoughts within <think></think> tags, and output the solution within <solution></solution>, and output answer within <answer></answer> tags."""


def print_reasoning_trajectory( trajectory, thought= None):
    colors = {
        'header': '\033[1;36m',    
        'problem': '\033[32m',     
        'category': '\033[34m',    
        'method': '\033[35m',      
        'knowledge': '\033[33m',   
        'flow': '\033[90m',        
        'highlight': '\033[1;93m', 
        'reset': '\033[0m'         
    }
    

    sections = [
        f"{colors['header']}╔{'═'*85}╗",
        f"║{' Reasoning Trajectory Analysis ':^85}║",
        f"╚{'═'*85}╝{colors['reset']}",
        
        f"\n{colors['method']}💡 Reasoning Process:{colors['reset']}",
        *[f"{'⎿' : <5}{line}" for line in thought.split('\n')],
        
        f"\n{colors['problem']}📌 Problem Statement:{colors['reset']}",
        *[f"   {line}" for line in trajectory['Problem'].split('\n')],
        
        f"\n{colors['category']}🧠 Knowledge Structure:{colors['reset']}",
        f"   ├─ General Category: {trajectory['General Knowledge Category']}",
        f"   └─ Specific Direction: {trajectory['Specific Direction']}",
        
        f"\n{colors['method']}🔧 Applied Methodology:{colors['reset']}",
        f"   ⮞ {trajectory['Applied Method']}",
        
        f"\n{colors['knowledge']}💎 Examined Knowledge Points:{colors['reset']}",
        *[f"   ▪ {point}" for point in trajectory['Examined Knowledge']],
        
        f"\n{colors['flow']}⛓️ Reasoning Flow:{colors['reset']}",
        *[f"{colors['highlight']}{i+1:>2}.{colors['flow']} {step}" 
        for i, step in enumerate(trajectory['reason_flow'])],
        
        f"\n{colors['header']}{' END OF ANALYSIS ':*^87}{colors['reset']}\n"
    ]
    
    print('\n'.join(sections))
    
def print_solution_template(template):

    colors = {
        'header': '\033[1;36m',  
        'title': '\033[1;93m',   
        'category': '\033[34m',  
        'keyword': '\033[32m',   
        'description': '\033[33m',  
        'step': '\033[90m',      
        'reset': '\033[0m'       
    }

    scenarios = template['application_scenario']
    if isinstance(scenarios, str):  
        scenarios = [scenarios]
    formatted_scenarios = "\n  ".join([f"• {scenario}" for scenario in scenarios])
 
    sections = [
        f"{colors['header']}╔{'═'*85}╗",
        f"║{' Solution Template Analysis ':^85}║",
        f"╚{'═'*85}╝{colors['reset']}",

        f"\n{colors['title']}🎯 Template Name: {colors['reset']}{template['template_name']}",
        f"\n{colors['category']}🏷️ Template Type: {colors['reset']}{template['template_type']}",

        f"\n{colors['keyword']}🔑 Knowledge Tags: {colors['reset']}" +
        ", ".join([f"{colors['keyword']}[{tag}]{colors['reset']}" for tag in template['knowledge_tag']]),

        f"\n{colors['description']}📝 Description: {colors['reset']}" +
        "\n  " + "\n  ".join(template['description'].split('\n')), 

        f"\n{colors['description']}✅ Application Scenario: {colors['reset']}" +
        "\n  " + formatted_scenarios,

        f"\n{colors['step']}⚙️ Reasoning Flow: {colors['reset']}",
        *[f"{colors['step']}{i + 1:>2}.{colors['reset']} {step}" for i, step in enumerate(template['reason_flow'])],

        f"\n{colors['header']}{' END OF ANALYSIS ':*^87}{colors['reset']}\n"
    ]
    print('\n'.join(sections))
    


def print_step(step_num, current_step, navigator_thought, current_problem, current_thought, current_solution):

    colors = {
        'header': '\033[1;36m',    
        'nav_think': '\033[34m',   
        'problem': '\033[32m',     
        'process': '\033[35m',     
        'solution': '\033[92m',    
        'reset': '\033[0m'         
    }
    
   
    output = [
        f"\n{colors['header']}╔{'═'*100}╗",
        f"║ Step {step_num:2}: {current_step.ljust(100)} ║",
        f"╚{'═'*100}╝{colors['reset']}",
        
        f"\n{colors['problem']}📝 Navigator Instruction:{colors['reset']}",
        f"{'⎿' : <5}{current_problem}",
        
        f"\n{colors['process']}💡 Reasoning Process:{colors['reset']}",
        *[f"{'⎿' : <5}{line}" for line in current_thought.split('\n')],
        
        f"\n{colors['solution']}✅ Solution:{colors['reset']}",
        f"{'⮞' : >3} {current_solution}",
        f"\n{'━'*65}\n"
    ]
    
    print('\n'.join(output))     
