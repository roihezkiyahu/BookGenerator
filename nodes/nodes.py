import json
import os
import logging
from tools.agent_tools import AgentState
from litellm import completion, image_generation
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
# from dotenv import load_dotenv
from .node_utils import *
from time import sleep
# load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_story(state: AgentState) -> AgentState:
    """Generate a story based on the state.
    
    Args:
        state: AgentState containing story parameters including title, 
               number of pages, description, and characters.
        
    Returns:
        AgentState: Updated state with the generated story.
    """
    if not state.get('validation_attempts', False):
        state['validation_attempts'] = 0
    if state.get('story_validation', {}).get('is_valid', False):
        return state
    
    if state.get("title", None) is None:
        response = completion(
            model="gemini/gemini-2.0-flash", 
            messages=[{"role": "user", "content": "Generate a title for a story"}],
            response_format={"type": "string"},
            temperature=0.7
        )
        state['title'] = response['choices'][0]['message']['content']

    prompt = f"""
    You are a storyteller. 
    You are given a title, number of pages, an optional story description, and an optional list of characters.
    You need to generate a story based on the input.

    When creating the story follow these guidelines:
    - Include all the characters in the story and add new characters if needed.
    - Supply a detailed description of each character including name and physical description inside the characters dictionary only.
    - Do not include the characters descriptions in the pages of the story.
    - Create a story that is engaging, interesting and coherent.
    - Create a story that is suitable for a children's book.
    - Pages should be short and to the point, each page should be about 2 - 3 sentences.
    - There should be no character descriptions in the story.

    Input:
    - title: {state['title']}
    - number_of_pages: {state['number_of_pages']}
    - story_description: {state['story_description']}
    - characters_metadata: {state['characters_metadata']}

    Your output should be a dictionary structured as follows:
    {{
        "story_outline": summary of the story plot/plan/characters/scenes,
        "characters": {{character_1: detailed description including name, physical description, clothing, hair, eyes, etc.,
          character_2: detailed description including name, physical description, clothing, hair, eyes, etc., ...}},
        "story": {{page_1: "text for page", page_2: "text for page", ...}},
        
    }}
    """

    # Define the response schema for structured output
    response_schema = {
        "type": "object",
        "properties": {
            "story_outline": {
                "type": "string",
                "description": "Summary of the story plot/plan/characters"
            },
            "characters": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                    "description": "Detailed character description including name and physical description"
                }
            },
            "story": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                    "description": "Text content for each page"
                }
            },
            
        },
        "required": ["story_outline", "characters", "story"]
    }
    prompt_validation = ''
    if state.get('story_validation', {}).get('is_valid', None) is not None:
        validation_errors = f"{state['story_validation']['validation_errors']}"
        prompt_validation = f"""You were given this task: {prompt}
        You returned the following story: {state['generated_story']}
        You need to fix the following errors: {validation_errors}
        Please refine the story to fix the errors. 
        Make sure to:
        - to include all the characters in the story and add new characters if needed.
        - to supply a detailed description of each character including name and physical description.
        - to create a story that is engaging, interesting and coherent.
        - to create a story that is suitable for a children's book.
        - pages should be short and to the point, each page should be a about 2 - 3 sentences.
        """
    response = completion(
        model="gemini/gemini-2.0-flash", 
        messages=[{"role": "user", "content": prompt if not prompt_validation else prompt_validation}],
        response_format={"type": "json_object", "schema": response_schema},
        temperature=0.7
    )

    state['generated_story'] = json.loads(response['choices'][0]['message']['content'])
    return state

def validate_story(state: AgentState) -> AgentState:
    """Validates the generated story against specified criteria.
    
    Args:
        state: AgentState containing the generated story and metadata.
        
    Returns:
        AgentState: Updated state with validation results.
    """
    if state.get('validation_attempts', 0) > 5:
        state['story_validation'] = {
            "is_valid": True,
            "validation_errors": "Validation attempts limit reached"
        }
        return state
    
    prompt = f"""
    You are a meticulous story validator.

    You are given story metadata and a generated story.
    Your task is to validate if the story meets all the following criteria:
    
    1. Page structure:
       - The story must have exactly {state['number_of_pages']} pages
       - Each page should have 2-4 lines of text maximum
       - Pages must flow logically in sequence
       - Each page should make sense of the previous pages and by itself.
       - Keep it in simple language and avoid complex words.
    
    2. Character requirements:
       - Each character listed in characters_metadata must appear in the story
       - Each character from characters_metadata must have detailed descriptions in the characters_dictionary
       - The characters_dictionary can include additional characters not in characters_metadata (this is valid and encouraged!)
       - All character descriptions should be in the characters_dictionary and not in the story
       - Each character description should be detailed and include name, physical description, clothing, hair, eyes, etc.
    
    IMPORTANT EXAMPLE FOR CLARITY:
    If characters_metadata = ["Lion", "Zebra"] and characters_dictionary includes {{"Lion": "...", "Zebra": "...", "Monkey": "..."}}, 
    this is VALID because all required characters (Lion, Zebra) are included, and additional characters (Monkey) are allowed.

    Metadata:
    - title: {state['title']}
    - number_of_pages: {state['number_of_pages']}
    - story_description: {state['story_description']}
    - characters_metadata: {state['characters_metadata']}

    Generated Story:
    - story_outline: {state['generated_story']['story_outline']}
    - characters_dictionary: {state['generated_story']['characters']}
    - story: {state['generated_story']['story']}

    Provide a structured output with the following fields:
    - validation_errors: detailed explanation of all validation errors found, or "No errors found" if valid
    - is_valid: boolean indicating if the story passes all criteria
    """

    response_schema = {
        "type": "object",
        "properties": {
            "validation_errors": {"type": "string"},
            "is_valid": {"type": "boolean"}
        },
        "required": ["validation_errors", "is_valid"]
    }

    response = completion(
        model="gemini/gemini-2.0-flash", 
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object", "schema": response_schema},
        temperature=0
    )
    response_json = json.loads(response['choices'][0]['message']['content'])
    state['story_validation'] = response_json
    state['validation_attempts'] += 1
    return state

def image_prompt_generator(state: AgentState) -> AgentState:
    """Generates an image prompt based on the story.
    
    Args:
        state: AgentState containing the generated story.
        
    Returns:
        AgentState: Updated state with the generated image prompt.
    """

    if not state.get('image_prompt_validation_attempts', False):
        state['image_prompt_validation_attempts'] = 0
    if state.get('images_prompts_validation', {}).get('is_valid', False):
        os.makedirs(f"{state['title']}/images", exist_ok=True)
        save_txt_file(state["images_prompts"], f"{state['title']}/images_prompts.json")
        save_txt_file(state["cover_image_prompt"], f"{state['title']}/cover_image_prompt.json")
        return state
    
    prompt = f"""
    You are an image prompt generator.
    you are given details about the story and the characters.
    you need to generate a detailed image prompt for each page of the story and the cover image.
    You are aware that the generation is image by image, so when describing a scene or a character make sure its coherent with the previous pages and metadata.
    The image generator you are using is is imagen-3 and it preducoes one image at a time, so when describing a scene or a character make sure its coherent with the previous pages and metadata.

    The image prompt should follow these instructions:
    - The image prompt should be detailed and include all the details about the scene and the character.
    - When a character is in the page, it should be described in detail including all its metadata from the characters_dictionary, so when the images are genereted it will look the same.
    - Scene description should be detailed and the same across all paged it appears in, with similar details and style.
    - The cover image must include the title of the story in like a real book cover WITHOUT AUTHOR NAME.
    - The image style is {state['image_style']}
    - If a character is saying something, do not include it in the image prompt.
    - Use best practices for image generation specifically for imagen-3.
        - Start with the style then the subject and lastly the context.
            e.g: A sketch (style) of a modern apartment building (subject) surrounded by skyscrapers (context and background)
        - Use descriptive language to describe the scene and the character.
        - Provide context: If necessary, include background information to aid the AI's understanding.
        - Use prompt engineering techniques to improve the quality of the image.
        - When making a cover specify the placement of the title and font size (e.g: medium, large, small).



    Story details:
        - story_outline: {state['generated_story']['story_outline']}
        - characters_dictionary: {state['generated_story']['characters']}
        - story: {state['generated_story']['story']}
    Provide a structured output with the following fields:
    {{
        "images_prompts": {{page_1: "image prompt", page_2: "image prompt", ...}},
        "cover_image_prompt": "image prompt for the cover image, needs to explictly say its a book cover",
    }}
    """
    response_schema = {
        "type": "object",
        "properties": {
            "images_prompts": {"type": "object", "additionalProperties": {"type": "string", "description": "Image prompt for each page"}},
            "cover_image_prompt": {"type": "string", "description": "Image prompt for the cover image"}
        },
        "required": ["images_prompts", "cover_image_prompt"]
    }

    prompt_validation = ''
    if state.get('images_prompts_validation', {}).get('is_valid', None) is not None:
        validation_errors = f"{state['images_prompts_validation']['validation_errors']}"
        prompt_validation = f"""
        You were given this task:
        {prompt}

        You returned the following images_prompts: {state['images_prompts']}
        You returned the following cover_image_prompt: {state['cover_image_prompt']}
        You need to fix the following errors: {validation_errors}
        Please refine the images_prompts and cover_image_prompt to fix the errors. 
        Make sure to:
        - to include all the characters in the story and add new characters if needed.
        - to supply a detailed description of each character including name and physical description.
        - to create a story that is engaging, interesting and coherent.
        - to create a story that is suitable for a children's book.
        - pages should be short and to the point, each page should be a about 2 - 3 sentences.
        """
    response = completion(
        model="gemini/gemini-2.0-flash", 
        messages=[{"role": "user", "content": prompt if not prompt_validation else prompt_validation}],
        response_format={"type": "json_object", "schema": response_schema},
        temperature=0.7
    )

    state['images_prompts'] = json.loads(response['choices'][0]['message']['content'])['images_prompts']
    state['cover_image_prompt'] = json.loads(response['choices'][0]['message']['content']).get('cover_image_prompt', "no prompt created")
    return state

def validate_image_prompts(state: AgentState) -> AgentState:
    """Validates the generated image prompts against specified criteria.
    
    Args:
        state: AgentState containing the generated image prompts.
        
    Returns:
        AgentState: Updated state with validation results.
    """
    if state.get('image_prompt_validation_attempts', 0) > 5:
        state['images_prompts_validation'] = {
            "is_valid": True,
            "validation_errors": "Validation attempts limit reached"
        }
        return state
    
    prompt = f"""
    You are a strict and meticulous images prompts validator.
    You are aware that image generation is very expensive and we are on a tight budget, so we need to make sure the prompts are as good as possible before generating the images.
    You are given a story_outline, characters_dictionary, story (of each page) and a list of image prompts.
    Your task is to validate if the image prompts are valid and coherent with the story and the characters.

    Validate the following:
    - All prompts are detailed and include all the details about the scene and the character.
    - Characters are described in detail including all its metadata from the characters_dictionary.
    - If a character is saying something, do not include it in the image prompt.
    - Each prompt is a "standalone" prompt that can be used to generate a single image without any knowlage of the other prompts or pages.
    - Each prompt is coherent with the story and the characters.
    - Each prompt is suitable for the image style {state['image_style']}.
    - Each 2 different prompts will end up with similar looking images: character, scene, etc.
    - Cover image prompt must be coherent with the story and the characters if listed in it, and must contain the title of the story to be generated like a real book cover, NO AUTHOR NAME.
    - Cover image prompt need to spesificlly include that this is a book cover with the title of the story listed in the prompt.
    

    Story details:
        - story_outline: {state['generated_story']['story_outline']}
        - characters_dictionary: {state['generated_story']['characters']}
        - story: {state['generated_story']['story']}
    Image prompts:
        - images_prompts: {state['images_prompts']}
        - cover_image_prompt: {state['cover_image_prompt']}

    Provide a structured output with the following fields:
    {{
        "validation_errors": "detailed explanation of all validation errors found, or "No errors found" if valid",
        "is_valid": "boolean indicating if the image prompts pass all criteria"
    }}
    """

    response_schema = {
        "type": "object",
        "properties": {
            "validation_errors": {"type": "string"},
            "is_valid": {"type": "boolean"}
        },
        "required": ["validation_errors", "is_valid"]
    }

    response = completion(
        model="gemini/gemini-2.0-flash", 
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object", "schema": response_schema},
        temperature=0
    )
    response_json = json.loads(response['choices'][0]['message']['content'])
    state['images_prompts_validation'] = response_json
    state['image_prompt_validation_attempts'] += 1
    return state

def merge_images(state: AgentState) -> AgentState:
    """Merge images and text into a single image.
    
    Args:
        state: AgentState containing the generated images and story text.
        
    Returns:
        AgentState: Updated state with the merged images that combine 
                   the original image and text side by side.
    """
    merged_dir = create_merged_image_directory(state['title'])
    if os.path.exists(f"{state['title']}/story.json"):
        with open(f"{state['title']}/story.json", "r") as f:
            state['generated_story']['story'] = json.load(f)
    
    if 'merged_images' not in state:
        state['merged_images'] = {}
    if not state['images_prompts']:
        with open(f"{state['title']}/images_prompts.json", "r") as f:
            state['images_prompts'] = json.load(f)
    if not state['cover_image_prompt']:
        with open(f"{state['title']}/cover_image_prompt.json", "r") as f:
            state['cover_image_prompt'] = json.load(f)
    if not state['generated_images']:
        for image_name in list(state['images_prompts'].keys()) + ["cover"]:
            if os.path.exists(f"{state['title']}/images/{image_name}.jpg"):
                state['generated_images'][image_name] = Image.open(f"{state['title']}/images/{image_name}.jpg")
    
    font = get_font(font_size=72)
    
    for page_key in state['images_prompts'].keys():
        if page_key in state['generated_images']:
            logger.info(f"Processing page: {page_key}")
            
            original_img = state['generated_images'][page_key]
            page_text = state['generated_story']['story'].get(page_key, "")
            img_width, img_height = original_img.size
            
            text_img = render_text_image(page_text, img_width, img_height, font)
            
            combined_width = img_width + text_img.width
            combined_img = Image.new('RGB', (combined_width, img_height))
            combined_img.paste(original_img, (0, 0))
            combined_img.paste(text_img, (img_width, 0))
            
            combined_path = f"{merged_dir}/{page_key}.jpg"
            save_image(combined_img, combined_path)
            
            state['merged_images'][page_key] = combined_img
    
    if 'cover' in state['generated_images']:
        logger.info("Processing cover image")
        cover_img = state['generated_images']['cover']
        state['merged_images']['cover'] = cover_img
        cover_path = f"{merged_dir}/cover.jpg"
        save_image(cover_img, cover_path)
    
    return state

def generate_images(state: AgentState) -> AgentState:
    """Generate images from the prompts.
    
    Args:
        state: AgentState containing the prompts.

    Returns:
        AgentState: Updated state with the generated images.
    """    
    save_txt_file(state['generated_story']['story'], f"{state['title']}/story.json")
    for image_name, prompt in zip(["cover"] + list(state['images_prompts'].keys()), [state['cover_image_prompt']] + list(state['images_prompts'].values())):
        cover_image_addition = "\n --book cover title: " + state['title'] if 'cover' in image_name else ""
        final_prompt = f"""Create an image a square image of aspect ratio 1:1 with size 1024x1024 based on the following prompt :
                         High Quality, 
                         {prompt}
                         {cover_image_addition}"""
        if state["generated_images"]:
            img = list(state['generated_images'].items())[-1]
            final_prompt += """Make sure the final image is similar in drawing style, colors, lightning, etc.
                                If Any charaters repeat themselfs make sure they look similar"""
            contents = [img, final_prompt]
        else:
            contents = [final_prompt]
        if state.get('use_imagen_3', False):
            client = genai.Client(api_key=os.getenv('GEMINI_API_KEY_PAID'))
            
            response = client.models.generate_images(
                model='imagen-3.0-generate-002',
                prompt=final_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images= 1
                )
            )
            generated_image = response.generated_images[0]
            image = Image.open(BytesIO(generated_image.image.image_bytes))
            state['generated_images'][image_name] = image
            image.save(f"{state['title']}/images/{image_name if 'cover' not in image_name else 'cover'}.jpg")
        else:
            client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
            for i in range(3):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-exp-image-generation",
                        contents=contents,
                        config=types.GenerateContentConfig(
                        response_modalities=["Text",'Image']
                        )
                    )
                except Exception as e:
                    logger.error(f"Error generating image: {e}, retrying in 10 seconds")
                    if i < 2:
                        sleep(10)
                        continue
                    else:
                        raise e
                break
                
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    state['generated_images'][image_name] = image
                    image.save(f"{state['title']}/images/{image_name if 'cover' not in image_name else 'cover'}.jpg")
                    break
    return state


def is_valid_image_prompts(state: AgentState) -> str:
    valid_image_prompts = "valid prompts" if state.get('images_prompts_validation', {}).get('is_valid', False) else "invalid image prompts"
    if valid_image_prompts == "valid prompts":
        valid_image_prompts += ", " + check_images_existance(state)
    return valid_image_prompts

def is_story_valid(state: AgentState) -> str:
    return "valid story" if state.get('story_validation', {}).get('is_valid', False) else "invalid story"

def check_images_existance(state: AgentState) -> str:
    files = [image for image in state['images_prompts'].keys()] + ["cover"]
    return "images exist" if all([os.path.exists(f"{state['title']}/images/{file}.jpg") for file in files]) else "images do not exist"
