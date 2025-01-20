"""
Heads or tails?

Toy setting for doing many rough guesses.
"""

from tasksolver.common import TaskSpec, ParsedAnswer, Question, KeyChain
from tasksolver.ollama import OllamaModel
from tasksolver.llama import LlamaModel
from tasksolver.exceptions import *
from tasksolver.utils import docs_for_GPT4
from tasksolver.claude import ClaudeModel
from tasksolver.gemini import GeminiModel
from tasksolver.qwen import QwenModel
from tasksolver.gpt4v import GPTModel
from tasksolver.phi import PhiModel
from tasksolver.minicpm import MiniCPMModel
from tasksolver.intern import InternModel
from PIL import Image
from pathlib import Path

# TODO: Import the class instance for your own model
# from tasksolver.your_model import YourModel

api_dict = KeyChain()
api_dict.add_key("openai_api_key", "BlenderAlchemyOfficial/credentials/openai_api.txt")
api_dict.add_key("claude_api_key", "BlenderAlchemyOfficial/credentials/anthropic_api.txt")
api_dict.add_key("gemini_api_key", "BlenderAlchemyOfficial/credentials/google_api.txt")

# TODO[optional]: Add API key for another model
# api_dict.add_key("your_api_key", "BlenderAlchemyOfficial/credentials/your_api.txt")

# Load images
image_path = 'TaskSolver/test_scripts/speed_limit.png'
image = Image.open(image_path)

class SpeedLimit(ParsedAnswer):
    def __init__(self, speed_limit:str):
        self.speed_limit = speed_limit

    @staticmethod    
    def parser(gpt_raw:str) -> "ReadSign":
        """
        @GPT4-doc-begin
            ONLY RETURN A NUMBER.
            
                For example,
                
                90

        @GPT4-doc-end
        """

        gpt_out = gpt_raw.strip().strip('.').strip(',').lower()

        if not gpt_out.isdigit():
            raise GPTOutputParseException("output should only contain a number!")

        return SpeedLimit(gpt_out)

    def __str__(self):
        return str(self.speed_limit)
    
read_speed_limit = TaskSpec(
    name="Read Speed Limit",
    description="You are given a picture on the right, which is about a speed limit sign in California . Please read it and find out the exact number of speed limit.",
    answer_type= SpeedLimit,
    followup_func= None,
    completed_func= None
)

read_speed_limit.add_background(
    Question([
        "ONLY RETURN A NUMBER. Read the following for the docs of the parser, which will parse your response, to guide the format of your responses:" , 
        docs_for_GPT4(SpeedLimit.parser) 
    ])
)


if __name__=='__main__':        
    question = Question(["Read the image now. What is the speed limit? ONLY RETURN THE NUMBER.", image])

    interface = GPTModel(api_key=api_dict['openai_api_key'], task=read_speed_limit)
    # interface = InternModel(task=read_speed_limit)

    # TODO: add your own model here. 
    # interface = YourModel(task=read_speed_limit)
    # Or if your model requires API:
    # interface = YourModel(api_key=api_dict['your_api_key'], task=read_speed_limit)

    '''
    Use rough_guess or many_rough_guesses(need first_question to process input)
    '''

    gpt_input = read_speed_limit.first_question(question)

    # Read for 5 times
    num_quries = 5
    out, _, _, _ = interface.many_rough_guesses(num_quries,gpt_input, max_tokens=2000, )
    outcomes = [str(el) for el in out if el is not None]
    print(outcomes)

    # # Read for a single time
    # out, _, _, _ = interface.rough_guess(gpt_input, max_tokens=2000, )
    # print(out)



