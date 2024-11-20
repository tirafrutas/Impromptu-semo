from openai import OpenAI
import json
from abc import abstractmethod
from enum import Enum

class PromptService:

    VALIDATOR_MODEL = 'gpt-4-vision-preview'

    __prompt = '''Provide a text that describes a new year beginning tomorrow. It is expected to be full of hope and happiness. The answer is written using a ironic register. The answer is written as a song'''

    __validators = [{"trait":"ironic","condition":"The answer is written using a ironic register"},{"trait":"song","condition":"The answer is written as a song"}]

    __media = 'text'
    MediaKind = Enum('Media', 'text image') # | 'audio' | 'video' | '3dobject')

    @property
    def prompt(self):
        return self.__prompt
    
    @property
    def media(self):
        return self.__media
    
    @property
    def response(self):
        if (self.__response): return self.__response
        return ''
    
    @property
    def validation(self):
        if (self.__validation): return self.__validation
        return []

    def __init__(self, openai_api_key: str):
        self.__validator = OpenAI(api_key=openai_api_key)
    
    def execute_prompt(self):
        self.__response = self.query_model()
        self.__validation = self.__validate_response()
    
    @abstractmethod
    def query_model(self):
        pass

    def __validate_response(self):
        if (self.media == self.MediaKind.text.name):
            validation_prompt = f'Given the PROMPT below and the RESPONSE given by an AI assistant as a response to the PROMPT. \
                Does the RESPONSE comply with the following LIST OF CONDITIONS (which are provided in JSON format, with keys "trait" and "condition")? \
                \
                Reply with a text in valid JSON format, that is: the content is embedded within an open and a closing bracket. \
                Do not include in your answer the term "json". Do not include in your answer any carry return, nor any special character other than brackets and curly brackets. \
                Your answer must include, for each item in the LIST OF CONDITIONS: \
                1. A key "trait" with the trait of the corresponding LIST OF CONDITIONS item. \
                2. A key "valid" only with "True" if the corresponding condition of the LIST OF CONDITIONS item is fulfilled by the RESPONSE; or "False" otherwise. \
                \
                PROMPT: ```{self.prompt}``` \
                \
                RESPONSE: ```{self.response}``` \
                \
                LIST OF CONDITIONS: {self.__validators}'
            message_payload = [{
                "role": "user",
                "content": validation_prompt
            }]
        # yet, we only consider text and image media outputs
        else:
            validation_prompt = f'Given the PROMPT below and the image provided by an AI assistant as a response to the PROMPT. \
                Does the image comply with the following LIST OF CONDITIONS (which are provided in JSON format, with keys "trait" and "condition")? \
                \
                Reply with a text in valid JSON format, that is: the content is embedded within an open and a closing bracket. \
                Do not include in your answer the term "json". Do not include in your answer any carry return, nor any special character other than brackets and curly brackets. \
                Your answer must include, for each item in the LIST OF CONDITIONS: \
                1. A key "trait" with the trait of the corresponding LIST OF CONDITIONS item. \
                2. A key "valid" only with "True" if the corresponding condition of the LIST OF CONDITIONS item is fulfilled by the image; or "False" otherwise. \
                \
                PROMPT: ```{self.prompt}``` \
                \
                LIST OF CONDITIONS: {self.__validators}'
            message_payload = [{
                "role": "user",
                "content": [
                    { "type": "text", "text": validation_prompt },
                    { "type": "image_url", "image_url": { "url": self.response } }
                ]
            }]
        validation = self.__query_validation(message_payload)
        return validation
            
    def __query_validation(self, message_payload):
        completion = self.__validator.chat.completions.create(
            model = self.VALIDATOR_MODEL,
            # if no max_tokens provided, default seems to be 16
            # 16 is insufficient for generating a valid json response
            # TODO: decide!
            max_tokens = 30,
            n = 1,
            messages = message_payload)
        return json.dumps(completion.choices[0].message.content)


class OpenAIService(PromptService):

    # Invoke with...
    # 1) Your own OpenAI's API key
    # 2) A valid OpenAI's chat model, e.g.:
    #       gpt-3.5-turbo-1106  # snapshot November 6th, 2023
    #       gpt-3.5-turbo
    #       gpt-3.5-turbo-16k
    #       gpt-4-0613          # snapshot June 13th, 2023
    #       gpt-4
    def __init__(self, openai_api_key: str, text_model: str):
        self.__client = OpenAI(api_key=openai_api_key)
        self.__model = text_model
        self.__image_model = 'dall-e-3'
        super().__init__(openai_api_key)

    def query_model(self):
        if (self.media == self.MediaKind.text.name): return self.__query_text()
        # yet, we only consider text and image media outputs
        else: return self.__query_image()
    
    def __query_text(self):
        completion = self.__client.chat.completions.create(
            model = self.__model,
            n = 1,
            # max_tokens = 10,
            messages = [{
                "role": "user",
                "content": self.prompt,
            }])
        return completion.choices[0].message.content
    
    def __query_image(self):
        response = self.__client.images.generate(
            model = self.__image_model,
            prompt = self.prompt,
            #size="1024x1024",
            #quality="standard",
            n = 1,
        )
        return response.data[0].url