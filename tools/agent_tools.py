from typing import List, Dict, Any, Optional
from typing import Annotated
from langgraph.graph.message import add_messages
from PIL import Image
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    State for the agent
    """
    title: str
    number_of_pages: int
    story_description: str
    characters_metadata: List[Dict[str, str]]
    image_style: str

    generated_story: Dict[str, str]
    story_validation: Dict[str, Any]
    images_prompts: List[Dict[str, str]]
    cover_image_prompt: str
    images_prompts_validation: Dict[str, Any]
    images_urls: List[Dict[str, str]]
    generated_text_images_urls: List[Dict[str, str]]
    merged_images: Dict[str, Image.Image]
    generated_images: Dict[str, Image.Image]

    messages: Annotated[list, add_messages]
    model: str
    response: str
    error: Optional[str] = None

    validation_attempts: int
    image_prompt_validation_attempts: int
    use_imagen_3: bool
    
